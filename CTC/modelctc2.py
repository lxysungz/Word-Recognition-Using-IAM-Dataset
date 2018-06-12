from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import random
import math
import glob
import util
from imp import reload

import cv2
import numpy as np
import time
import json

from tensorflow.python import debug as tf_debug
import argparse
import sys


# In[2]:


def cnn(x, kernel_size, num_neurons, num_conv1, num_conv2, num_conv3, num_conv4, num_conv5,keep_prob):
    # VARIABLES
    # filters and bias of CNNs
    w_conv1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1,num_conv1], stddev=0.01))
    b_conv1 = tf.Variable(tf.random_normal([num_conv1], stddev=0.01))
    w_conv2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv1, num_conv2], stddev=0.01))
    b_conv2 = tf.Variable(tf.random_normal([num_conv2], stddev=0.01))
    w_conv3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv2,num_conv3], stddev=0.01))
    b_conv3 = tf.Variable(tf.random_normal([num_conv3], stddev=0.01))
    w_conv4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv3, num_conv4], stddev=0.01))
    b_conv4 = tf.Variable(tf.random_normal([num_conv4], stddev=0.01))
    w_conv5 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv4, num_conv5], stddev=0.01))
    b_conv5 = tf.Variable(tf.random_normal([num_conv5], stddev=0.01))


    # Normalize input
    #x = tf.nn.l2_normalize(x, [1, 2])

    #LAYER 1 CNN-MAXPOOL-DROP
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #h_pool1=tf.nn.l2_normalize(h_pool1,[1,2])
    h_pool1=tf.nn.dropout(h_pool1,keep_prob=keep_prob)

    # LAYER 2 CNN-MAXPOOL-DROP
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #h_pool2=tf.nn.l2_normalize(h_pool2,[1,2])
    h_pool2=tf.nn.dropout(h_pool2,keep_prob=keep_prob)

    # LAYER 3 CNN-MAXPOOL-DROP
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #h_pool3=tf.nn.l2_normalize(h_pool3,[1,2])
    h_pool3=tf.nn.dropout(h_pool3,keep_prob=keep_prob)

    # LAYER 4 CNN-MAXPOOL-DROP
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)
    h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #h_pool4=tf.nn.l2_normalize(h_pool4,[1,2])
    h_pool4=tf.nn.dropout(h_pool4,keep_prob=keep_prob)

    # LAYER 5 CNN-MAXPOOL-DROP
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
    h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #h_pool5=tf.nn.l2_normalize(h_pool5,[1,2])
    h_pool5=tf.nn.dropout(h_pool5,keep_prob=keep_prob)
    
    return h_pool5


    # In[3]:


def lstm(x, num_neurons):
    # weights RNN
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_neurons, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_neurons, forget_bias=1.0)

    outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x ,dtype=tf.float32)
    print("lstm outputs 0 shape: ", outputs[0].shape)
    print("lstm outputs 0 shape: ", outputs[1].shape)
    outputs=tf.concat(outputs,2) #(b,time_step, num_neurons*2)


    return outputs
            


    # In[4]:


def fc(x, num_fc, num_classes):
    # weights FULLCONNECT
    num_inputs = int(x.get_shape()[1])
    w1 = tf.Variable(tf.random_normal([num_inputs, num_fc], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([num_fc], stddev=0.01))
    w2 = tf.Variable(tf.random_normal([num_fc,num_classes], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([num_classes], stddev=0.01))
    logits = tf.matmul(x, w1) + b1
    logits =  tf.matmul(logits, w2) + b2
    
    return logits


    # In[5]:


def ctc(inputs, labels, seq_len, num_classes, learning_rate=0.01):
    loss = tf.nn.ctc_loss(labels, inputs, seq_len, preprocess_collapse_repeated=False)
    cost = tf.reduce_mean(loss)
    
    # minimizing the error
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(cost, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)        
    optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params))

    # decode to extract the sequence of charcaters
    decoded, log_prob = tf.nn.ctc_greedy_decoder(inputs, seq_len)
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), labels)
    # Error: Label Error Rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

    #condition = tf.greater(distance, tf.zeros(distance.shape))
    
    wer = tf.reduce_mean(tf.sign(distance))

    return optimizer, decoded, cost, ler, wer


# In[6]:


def model(config):
    learning_rate = config['learning_rate']
    kernel_size = config['kernel_size']
    num_conv1 = config['num_conv1']
    num_conv2 = config['num_conv2']
    num_conv3 = config['num_conv3']
    num_conv4 = config['num_conv4']
    num_conv5 = config['num_conv5']
    num_classes = config['num_classes']
    num_fc = config['num_neurons_hidden_layer']
    img_height = config['img_height']
    img_width = config['img_width']
    
    #number of output neurons in each RNN cell
    num_neurons = config['num_neuron_per_rnn_cell']
    graph = tf.Graph()
    with graph.as_default():

        # PLACEHOLDERS
        # x: input Tensor [batch_size, height, width, channel] 
        X = tf.placeholder(tf.float32, [None, img_height, img_width,1])
        seq_len = tf.placeholder(tf.int32, [None])
        keep_prob = tf.placeholder(tf.float32)
        # y: labels sparse_tensor (indices, values, [batch_size, word_len])
        Y = tf.sparse_placeholder(tf.int32)
        
       # Normalization inputs
        X = tf.nn.l2_normalize(X, [1, 2])
        
        cnn_output = cnn(X, kernel_size, num_neurons,num_conv1, num_conv2, num_conv3, num_conv4, num_conv5,keep_prob)
        
        # reshape cnn_output with shape (batch_size, rows, colums, channels) to fit 
        # LSTM input shape (batch_size, time_steps, data)
        #batch_size = cnn_output.shape[0]
        rows = cnn_output.shape[1]
        time_step = cnn_output.shape[2]
        channels = cnn_output.shape[3]     
        print("cnn_output shape: ", cnn_output.shape)
 
        with tf.name_scope("adapt_shape"):
            outputs = tf.transpose(cnn_output, (0,2,1,3))
            #outputs = tf.reshape(outputs, (time_step,-1,rows*channels))
            outputs = tf.reshape(outputs, (-1, math.ceil(img_width/(2**5)), num_conv5*math.ceil(img_height/(2**5))))
            #outputs = tf.transpose(outputs, (1,0,2))
        print("input to lstm shape: ", outputs.shape)
        lstm_outputs = lstm(outputs, num_neurons)  #lstm outputs (batch, time, num_neurons*2)
        print("lstm_outputs shape: ", lstm_outputs.shape)   
        #outputs=tf.transpose(outputs, (1,0,2))   #(time_step, b, num_neurons*2)      
        lstm_outputs=tf.reshape(lstm_outputs, (-1,num_neurons*2))  #flatten lstm outputs to (batch*time, num_neurons*2)         
        logits = fc(lstm_outputs, num_fc, num_classes)    # FULL CONNECT
        print("logits shape: ", logits.shape)
        logits = tf.reshape(logits, (-1, math.ceil(img_width/(2**5)),num_classes))  #(batch, time, num_classes)
        logits = tf.transpose(logits, (1,0,2))
        print("input to ctc shape: ", logits.shape)
        
        optimizer, decoded, cost, ler, wer = ctc(logits, Y, seq_len, num_classes, learning_rate)    #CTC
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('ler', ler)
    
    return graph, X, Y, keep_prob, seq_len, optimizer, cost, ler, decoded, wer


# In[7]:


def validation(val_dir, batch_size, ctc_input_len, word_len, level, X, Y, keep_prob, seq_len, session, merged, cost, ler, wer):

    """
        
        Args:

          - ctc_input_len: input length of ctc (Int)
          - batch_size:  (Int)
          - im_path: images path (String)
          - csv_path: images dataset path (Int)
          - inputs: Placeholder of input (placeholder)
          - targets: Placeholder of output (placeholder)
          - keep_prob: Placeholder of dropout probability (placeholder)
          - seq_len: Placeholder for the length of the CTC input (placeholder)
          - session: TensorFlow session. (Session)
          - cost:CTC cost. (Tensor: [1])
          - ler: Tensor for LER. (Tensor:[1])

        reuturn:

          - val_tuple: Result of the validation. (Tuple: {'step','cost','LER'})
    """
 
    val_set = util.dataset(val_dir, batch_size, ctc_input_len, word_len,0, level)
    total_val_cost = 0
    total_val_ler = 0
    total_val_wer = 0
    cont = 1
    count = 0
    while cont > 0:
  
        val_inputs, val_targets, val_original, val_seq_len = val_set.extract_data_batch()
        if len(val_inputs)>0:
            val_feed = {X: val_inputs,
                    Y: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            summary, val_cost, val_ler, val_wer = session.run([merged, cost, ler, wer], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            total_val_wer += val_wer
            count += 1
            #print("val step: ", count, "total cost: ", total_val_cost, "total ler: ", total_val_ler)
        else:
            cont = 0

    val_tuple = (summary, total_val_cost/count, total_val_ler/count, total_val_wer/count)
    return val_tuple


    # In[8]:
    
def main(_):

    DEBUG = 1
    SUMMARY_DIR = os.path.curdir + '/summary2/'
    TRAIN_DIR = os.path.curdir+'/trainImgPad1024/'
    VAL_DIR = os.path.curdir+'/testImgPad1024/'
    CHECKPOINT_DIR = os.path.curdir + '/checkpoint_10/'
    print("summary: ", SUMMARY_DIR)

    BASE_CHANNELS = 4
    TRAIN_STEPS = 2
    PRINT_STEPS = 1
    VAL_STEPS = 1
    BATCH_SIZE = 64
    KEEP_PROB = 1
    IMG_HEIGHT = 300
    IMG_WIDTH = 1024

    # After 5 times of max pool of CNN model, the output image resolution is 1/x**5 of original, 
    # therefor the input to CTC is (ctc_input_len =  math.ceil(IMG_WIDTH/(2**5)), batch_size, num_classes*IMG_HEIGHT/2**5)
    #ctc_input_len = math.ceil(IMG_WIDTH/(2**5)/2)
    ctc_input_len = 10
    WORD_LEN = 10

    # 自定义参数词典，大多数是模型需要的超级参数
    params = { 
               'learning_rate' : 0.0001,
               'num_conv1': BASE_CHANNELS,
               'num_conv2': BASE_CHANNELS*2,
               'num_conv3': BASE_CHANNELS*4,
               'num_conv4': BASE_CHANNELS*8,
               'num_conv5': BASE_CHANNELS*16,
               'kernel_size': 3,
               'num_classes' : 53, #26 大写， 26 小写字母， 空白
               'num_neuron_per_rnn_cell': 1024,
               'num_neurons_hidden_layer': 512,
               'img_height': IMG_HEIGHT,
               'img_width': IMG_WIDTH,
               'ctc_input_len': ctc_input_len,
               'word_len': WORD_LEN
               }
    params_json = json.dumps(params)
    f = open("config_4.json","w")
    f.write(params_json)
    f.close()
    random.seed(2)

    # In[ ]:


    # train the model
    net = model(params)
    graph=net[0]
    X=net[1]
    Y=net[2]
    keep_prob=net[3]
    seq_len=net[4]
    optimizer=net[5]
    cost=net[6]
    ler=net[7]
    decoded=net[8]
    wer=net[9]



    with tf.Session(graph=graph) as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses", dump_root='C:/data/debug_tmp')

        train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test') 

        saver=tf.train.Saver(max_to_keep=20) 
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path: #Continue training with previous saved variable values
            print(" restore variables from ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1])
            print("global_step: ", global_step)
        else: #start a new train if no saved checkpoints to restore variables
            tf.global_variables_initializer().run()
            global_step = 0
 
        merged = tf.summary.merge_all()
        
        train_dataset = util.dataset(TRAIN_DIR, BATCH_SIZE, ctc_input_len, WORD_LEN, 1, 1)
        start_time = time.time()
        for curr_step in range(TRAIN_STEPS):
            
            train_inputs, train_labels, original, train_seq_len = train_dataset.extract_data_batch()                             
            train_feed = {X: train_inputs, Y: train_labels, keep_prob: 0.9, seq_len: train_seq_len}        
            _ = sess.run([optimizer], train_feed)

            if curr_step % PRINT_STEPS == 0:
                summary, train_cost, train_ler = sess.run([merged, cost, ler], train_feed)
                train_writer.add_summary(summary, curr_step+global_step)
                elapse_time = time.time() - start_time
                print("Step:", curr_step+global_step, "Train cost:", train_cost, "Train ler:", train_ler, "Duration(ms): ", elapse_time)
                print("---------------------------------------------------------------------")
                start_time = time.time()
            if curr_step>0 and curr_step % VAL_STEPS == 0:
                val_start_time = time.time()
                summary, val_cost, val_ler, val_wer = validation(VAL_DIR, BATCH_SIZE, ctc_input_len, 
                                    WORD_LEN, 1, X, Y, keep_prob, seq_len, sess, merged, cost, ler, wer)
                test_writer.add_summary(summary, curr_step+global_step)
                elapse_val_time = time.time() - val_start_time
                print("---------------------------------------------------------------------")
                print("Step:", curr_step+global_step, "Val cost:", val_cost, "Val ler:", val_ler, "Val wer: ", val_wer, "Duration(ms): ", elapse_val_time)
                
                save_path = saver.save(sess, CHECKPOINT_DIR+"model"+'.ckpt_'+str(curr_step+global_step))
                print("Model saved in file: " +str(save_path))
                print("=====================================================================")
                
        save_path = saver.save(sess, CHECKPOINT_DIR+"model"+'.ckpt_'+str(curr_step+global_step))
        print("Model saved in file: " +str(save_path))
        print("=====================================================================")        
        print("THE TRAINING IS OVER")
        train_writer.close()
        test_writer.close()
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. "
            "Mutually exclusive with the --tensorboard_debug_address flag.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  
    



