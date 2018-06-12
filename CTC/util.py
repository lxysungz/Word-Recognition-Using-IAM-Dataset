import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import glob
import cv2
import math
import random
import preprocessImg

class dataset:
    def __init__(self, path, batch_size, ctc_input_len, word_len, repeat, level=1):
        #self.path = path
        self.__batch_size = batch_size
        self.__ctc_input_len = ctc_input_len
        self.__word_len = word_len
        self.__repeat = repeat
        self.__index = 0
        self.__level = level
        self.__sample_set = self.get_sample_filenames(path, level=level)
        if repeat==1:
            random.shuffle(self.__sample_set)
        

        
    def get_words(self, batch_set):

        #return np.asarray([np.int32(filename.split(os.sep)[-2]) for filename in filenames])
        words=[]
        #words = np.asarray([filename.split(os.sep)[-1].split('.')[-2].split('-')[-2] for filename in filenames])
        for filename in batch_set:
            word = filename.split(os.sep)[-1]
            #print("first spit: ", word)
            word = word.split('.')[-2].split('-')[-2]
            if len(word)==0:
                word = filename.split(os.sep)[-1].split('.')[0].split('-')[-1]
                #print("special word: ", word)
            words.append(word)
     
        return words
        
    def extract_data_batch(self):

        """
            Return batch of images and labels in a random way to train model

            Args:

              - ctc_input_len: input length of ctc (int)
              - batch_size:  (Int)
              - train_dir: train image path (String)        

            return:

              - batchx: Tensor with images as matrices
                (Array of Floats: [batch_size, height, width, 1])
              - sparse: SparseTensor with labels (SparseTensor: indice,values,shape)
              - transcriptions: Arraywith labels of "batchx". (Array de Strings: [batch_size])
              - seq_len: Array with input length for CTC, "ctc_input_len". (Array of Ints: [batch_size])
        """

        batchx = []
        transcriptions = []
        indice = []
        seq_len=[]
        values = []
        i = 0
        
        if self.__repeat and self.__index +self.__batch_size > len(self.__sample_set):
            random.shuffle(self.__sample_set)
            self.__index = 0        
        elif not self.__repeat and self.__index + self.__batch_size > len(self.__sample_set):  
            return batchx, None, transcriptions, seq_len
            
        batch_set = self.__sample_set[self.__index:self.__index+self.__batch_size]
        words = self.get_words(batch_set)
        i =0
        for file, word in zip(batch_set, words):
            if len(word)==0:
                print("word is empty for image ", file, "return empyt batch")
                return None, None, transcriptions, seq_len
            else:
                img = cv2.imread(file,0)

                height = img.shape[0]
                width = img.shape[1]
                result=img.reshape(height, width,1)
                result = result/255.0
                batchx.append(result)

                # extract labels, here we only get images with english letters, 
                j = 0
                for ch in list(str(word)):
                    if ord(ch)<97: 
                        values.append(ord(ch)-65) 
                        indice.append([i,j])
                    else:
                        values.append(ord(ch)-97+26) 
                        if j>self.__ctc_input_len:
                            print("Error: Word", word, " length", j, " > ctc_input_len", self.__ctc_input_len, " return empty batch")
                            return None, None, transcriptions, seq_len
                        indice.append([i,j])
                    
                    j = j + 1
                            
                transcriptions.append(word)
                seq_len.append(self.__ctc_input_len)
                i +=1
        self.__index += self.__batch_size
        batchx = np.stack(batchx, axis=0)   
        shape=[self.__batch_size,self.__word_len]
        sparse=indice,values,shape
        #print ("labels values: ", values) 
        #print ('batch set: ', batch_set)
        return batchx, sparse, transcriptions, seq_len
        
    def extract_predict_data_batch(self, std_height, std_width):

        batchx = []
        indice = []
        seq_len=[]
 
        i = 0
        if self.__index >= len(self.__sample_set):           
            return batchx, seq_len, []
        if self.__index + self.__batch_size > len(self.__sample_set):  
            batch_set = self.__sample_set[self.__index:]           
        else:
            batch_set = self.__sample_set[self.__index:self.__index+self.__batch_size]

        for file in batch_set:
            img = preprocessImg.normalizeImg(filename=file, img=None, stdRow=std_height, stdCol=std_width)
            
            #img = cv2.imread(file,0)
            height = img.shape[0]
            width = img.shape[1]
  
            result=img.reshape(height, width,1)
            result = result/255.0
            batchx.append(result)
            seq_len.append(self.__ctc_input_len)
            i +=1
        self.__index += len(batch_set)
        batchx = np.stack(batchx, axis=0)   
 
        return batchx, seq_len, batch_set

    def get_sample_filenames(self, directory, level):
        if level==1:
            file_set = sorted(glob.glob(directory+'*/*.png'))
            
        else:
            file_set_png = sorted(glob.glob(directory+'*.png'))
            file_set_jpg = sorted(glob.glob(directory+'*.jpg'))
            file_set = file_set_png + file_set_jpg
           # if len(file_set) == 0:
           #     file_set=sorted(glob.glob(directory+'*.jpg'))
 
        return file_set
    


