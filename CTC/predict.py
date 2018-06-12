import os
import glob
import sys
from random import randint
import tensorflow as tf
import util
import pandas as pd
import modelctc1
import json
import ast
import numpy as np

def convert_word(indices, codes, shape):
    words = []
    word = []
    i = 0
    j=0
    for index in indices:
        if i!=index[0]:  
            words.append(word)
            i = index[0]
            word = []
        if codes[j]<26:
            word.append(chr(codes[j]+65))
        elif codes[j]<52:
            word.append(chr(codes[j]+97-26))
        j += 1
    words.append(word)
    return words
    
def run_ctc():
    print("cur dir: ", os.path.curdir)
    ckpt = tf.train.get_checkpoint_state('./checkpoint_10/')
    checkpoint_file = ckpt.model_checkpoint_path
    config_file = str('./config_4.json')
    img_dir = str('./predictImg/')
    print("len arg: ", len(sys.argv))
    if len(sys.argv) == 1:
        print("Execution without arguments, default arguments")
        print("checkpoints_file=",checkpoint_file)
        print("config_file=", config_file)
        print("img_dir=", img_dir)
    elif len(sys.argv) == 2:       
        print("Execution without some arguments, default arguments")
        print("checkpoints_file=",checkpoint_file)
        print("config_file=", config_file)        
        img_dir = str(sys.argv[1])

    elif len(sys.argv) == 3:
        print("Execution without some arguments, default arguments")
        print("config_file=", config_file)
        print("img_dir=", img_dir)
        img_dir = str(sys.argv[1])
        checkpoint_file = str(sys.argv[2])
        
    elif len(sys.argv) == 4:
        img_dir = str(sys.argv[1])
        checkpoint_file = str(sys.argv[2])
        config_file = str(sys.argv[3])
        
    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 predict.py [checkpoint_file] [config_file] [img_dir]")
        print("e.g. python predict.py ./checkpoints/model.ckpt_1000 config.json ./img_to_predict/")
        exit(1)

    try:
        config = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)



    BATCH_SIZE = 4
    std_height = 300
    std_width = 1024
    ctc_input_len = int(config['ctc_input_len'])
    word_len = int(config['word_len'])
    
    
    net = modelctc1.model(config)
    graph=net[0]
    X=net[1]
    Y=net[2]
    keep_prob=net[3]
    seq_len=net[4]
    optimizer=net[5]
    cost=net[6]
    ler=net[7]
    decoded=net[8]
    wer = net[9]

    #result_test = pd.DataFrame()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allocator_type = 'BFC'


    with tf.Session(graph=graph,config = sess_config) as session:
    #with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file) 
        print("Loaded Model")
        
        predict_set = util.dataset(img_dir, BATCH_SIZE, ctc_input_len, word_len,0, 0)
        cont = 1
        while cont > 0: 
            outputs = []
            pre_inputs, pre_seq_len, img_list = predict_set.extract_predict_data_batch(std_height,std_width)
            #print("img list: ", img_list)
            if len(pre_inputs)>0:
                predict_feed = {X: pre_inputs, keep_prob: 1, seq_len: pre_seq_len}
                result = session.run(decoded[0], predict_feed)
                #print("result: ", result.values)
                #print("result.indices: ", result.indices)
                output = convert_word(result.indices, result.values, result.dense_shape)
                #print("val step: ", count, "total cost: ", total_val_cost, "total ler: ", total_val_ler)
            else:
                cont = 0 
            print("outputs: ", outputs)
            for img_file, word in zip(img_list, output):
                print("image: "+img_file + "predict: "+ str(word))
 
        
        return outputs

if __name__ == '__main__':
    run_ctc()
