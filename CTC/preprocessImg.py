
# coding: utf-8

# # 预处理待识别的图片
# 
# ## Lucia
# ```
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ```

# ### 概要
# 图片要求： 行与行之间必须有水平的空行间隔
# 预处理步骤：
#     -将图片转为黑底白字https://docs.opencv.org/3.3.0/d7/d4d/tutorial_py_thresholding.html
#     -消除噪点https://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
#     -分行，并且截取每一行的图片
#     -对行图片分词， 并且截取每一个单词的图片
#     
#     


import os
import sys
import glob
from random import shuffle
import shutil
import csv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from random import randint
import numpy as np
import cv2
import math
from scipy import ndimage
import pandas as pd
import tensorflow as tf
import statistics


# ### cv2.adaptiveThreshold
# 自适应阈值将图片转为二值图片，自适应阈值可以看成一种局部性的阈值，通过规定一个区域大小，比较这个点与区域大小里面像素点的计算值（或者其他特征）的大小关系确定这个像素点是属于黑或者白（如果是二值情况）。使用的函数为：cv2.adaptiveThreshold（） 
# 该函数需要填6个参数：
# 
#     第一个原始图像
#     第二个像素值上限
#     第三个自适应方法Adaptive Method: 
#         — cv2.ADAPTIVE_THRESH_MEAN_C ：区域内均值 
#         —cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：区域内像素点加权和，权 重为一个高斯窗口
#     第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
#     第五个Block size:规定区域大小（一个正方形的领域）
#     第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） 
# 这种方法在实际图片上的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用同一个阈值。

# In[3]:


def blackwhiteImg(img, inverse):
    """将原始彩色图片，或者处理过的灰度图片二值化为黑白图片，注意调用前，灰度图片必须以灰度方式打开，如果灰度图片以彩色方式打开转换的结果会有问题
    参数:
        img: 2维灰度图，或者原始3维彩色图
    返回:
        二值化的黑底白字图
    """    
    # 双边滤波消噪同时保留清晰的边缘，比其他滤波算法慢一些，noise removal while keeping edges sharp
    img = cv2.bilateralFilter(img,10,75,75) 
    #print(len(img.shape))
    if len(img.shape)==3:
        #对于原始的彩色图片先转为灰度图片，因为光线明暗的因素，用自适应阈值转为黑白的二值化图片效果最佳
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        #将灰度图转换为黑白图，转换后像素值为0（黑）       
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    else:
        #对于已经经过黑白转换，然后因为调用correctSkew 和correctSlant而变为灰度的图片，用Otsu’s二值化转为黑白图片远比自适应阈值更合适
        ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
    #如果是白底黑字的图片，需要转为黑底白字，
    if inverse==1:
        img = 255-img
 
    return img


def cropImg(img):
    
    #print ("cropImg sum:", np.sum(img), "  shape: ", img.shape)
    if np.sum(img) > 0:
        row, col = img.shape
        while np.sum(img[0]) == 0:
            
            if row<=1:
                return img
            else:
                img = img[1:] 
                row = row - 1
        
        while np.sum(img[:,0]) == 0:
            
            if col<=1:
                return img
            else:
                img = np.delete(img,0,1)
                col = col - 1
        
           
        while np.sum(img[-1]) == 0:
            
            if row<=1:
                return img
            else:
                img = img[:-1] 
                row = row - 1
            #row, col = img.shape
            #print ("cropImg sum:", np.sum(img), "  shape: ", img.shape)
        while np.sum(img[:,-1]) == 0:
            
            if col<=1:
                return img
            else:
                img = np.delete(img,-1,1)
                col = col -1
            
        return img
    else:        
        return np.zeros(1)



#convert jpg to MNIST format using OpenCV
def convertByCV(gray):
    
    gray = cv2.GaussianBlur(gray,(5,5),1)
    (thresh, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    #(thresh, gray) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
      
    gray = 255 - gray
    
    #print("origin{}".format(gray.shape))
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)        
    while np.sum(gray[-1]) == 0:
        gray = gray[:-2]
    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)
    

    print (gray.shape)
    rows,cols = gray.shape
    
    if rows>cols:
        colsPadding = (int(math.ceil((rows-cols)/2.0)+2),int(math.floor((rows-cols)/2.0)+2))
        rowsPadding = (2,2)
    else:
        rowsPadding = (int(math.ceil((cols-rows)/2.0)+2),int(math.floor((cols-rows)/2.0)+2))
        colsPadding = (2,2)
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    print("squareGray{}".format(gray.shape))    
    
    interpolation = cv2.INTER_CUBIC

    gray = cv2.resize(gray, (28, 28),interpolation)

    print(gray.shape)
    return gray


# ### 纠正倾斜的字体
#不改变高度的前提下旋转
def shear(img,shear_range):

    rows, cols = img.shape
    pts1 = np.float32([[cols/4,rows/4],[cols/2,rows/4],[cols/4,rows/2]])

    pt1 = cols/4+shear_range
    pt2 = cols/2+shear_range

    pts2 = np.float32([[pt1,rows/4],[pt2,rows/4],[cols/4,rows/2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    shearedImg = cv2.warpAffine(img,shear_M,(cols,rows))

    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(shearedImg),plt.title('Sheared Image')
    #plt.show()
    return shearedImg



def projection(img):
    """projection calculate the vertical projection value of a black/white image
    Args:
        img: 2D black/white image
    Returns:
        vertical projection value
    """
    # array to store projection value of each column
    img = cropImg(img)
    w = [0]*img.shape[1]

    # 对每一列计算投影值
    # 把每一列上像素值加起来，然后返回像数值最大的那一列的值
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            w[y] = w[y] + img[x][y]

    return max(w)



#修正文本倾斜的角度
def correctSlant(gray):
    """correctSlant correct the slant of image of word, e.g. italic font
    Args:
        image: 2D image black and White image
    Returns:
        A 2D  imaged with slant corrected, background is black and foreground is white
    """
    #convert image to black background and while foreground image
    #gray = blackwhiteImg(image)
    
    #填充图片外框，避免矫正倾斜后文字信息丢失
    rows, cols = gray.shape
    if rows>20 and cols>20:
        rowsPadding = (math.ceil(gray.shape[0]/20),math.ceil(gray.shape[0]/20))
        colsPadding = (math.ceil(gray.shape[0]/2),math.ceil(gray.shape[0]/2))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
        #shear the image with different angles, calculate the vertical projection value
        #the angle with maximum projection value is the text slant angle
        #shear the image with the slant angle to correct slant 

        startAngle = - int(math.ceil(rows/2))
        endAngle = int(math.ceil(rows/2))
        step = math.floor(rows/20)
        maxValue = 0
        slantAngle = 0
        #print ("step: ", step)
        for angle in range(startAngle,endAngle,step):
            shearedImg = shear(gray, angle)
            projectValue = projection(shearedImg)
            if (maxValue < projectValue):
                maxValue = projectValue
                slantAngle = angle

        finalImg = shear(gray, slantAngle)
    else:
        finalImg = gray
    return finalImg


# ### 水平矫正
# ![水平矫正示意图](document/CorrectSkew.png)

# In[11]:


#水平矫正
def correctSkew(gray):
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    # 计算不为0的像素的坐标
    coords = np.column_stack(np.where(gray > 0))
    # 求能包围所有>0的像素的面积最小的长方形的旋转角度，顺时针方向为正，逆时针为负
    # 参考https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    # 函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
    # 注意：旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。
    #      并且这个边的边长是宽width，另一条边边长是长height。也就是说，在这里，width与height不是按照长短来定义的。
    #在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。在这里，θ∈（-90度，0]。
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    
    #填充图片，避免旋转矫正后文字信息丢失   
    rowsPadding = (math.ceil(gray.shape[0]/4),math.ceil(gray.shape[0]/4))
    colsPadding = (math.ceil(gray.shape[1]/4),math.ceil(gray.shape[1]/4))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    
    # cv2.minAreaRect返回角度的值范围[-90, 0)，
    # 而下面旋转用的仿射函数角度为正时逆时针转，角度为负时顺时针转
    # 所以需要把cv2.minAreaRect返回的角度转换
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # 旋转图片进行水平矫正，这里正好和前面相反，顺时针旋转为负，逆时针为正
    # 注意： 图形旋转之后，边缘有渐变平滑的锯齿
    # 所以旋转后，本图不再是二值图像，有渐变灰度
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cropImg(rotated)
    
    return rotated
    


# In[12]:


def segmentation(img):
    w = [0]*img.shape[1]
    segList = []
    count = 1;
    # 对每一列计算投影值
    # 把每一列上像素值加起来，然后返回像数值最大的那一列的值
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if (img[x][y])==255 and (count==1):
                w[y] = w[y] + img[x][y]
                count = 0;
            elif (img[x][y]!=255):
                count = 1;
    leftSlop = 0
    for y in range(1,img.shape[1]-1):  
        if (w[y] == 0):
            segList.append(y)
        #elif ((w[y]<w[y+1]) and w[y]<w[y-1]):
         #   segList.append(y)
       
        
    return segList


# In[13]:


def isBinaryImg(img):
    count =0
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            if img[i][j] !=0 and img[i][j]!=255:
                count = count +1

    return count


# In[14]:


def cropWordImg(lineGrayImg, threshold):
    """
    参数
        lineGrayImg: 黑底白字二值一行英文文字图片
        threshold： 单词间的间隔最小列数
    返回
        分词，返回一个list，list里的元素是每个单词的(起始列,结束列)的元组，排列顺序按单词在句子中的顺序
    """
   # img = correctSlant(lineGrayImg)
    
    
    #img = cropImg(lineGrayImg)
    #rowsPadding = (2,2)
    #colsPadding = (2,2)
    #img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')
    img = lineGrayImg
    w = [0]*img.shape[1]
    ImgsOfWords = []
    segList = []
    count = 1;
    
    # 对每一列计算投影值
    # 把每一列上像素值加起来， 由于是黑底白字而黑色的像素值为0，如果连续几列上像素值和为0，就意味着这是单词间隔
    for y in range(img.shape[1]):
        for x in range(img.shape[0]):             
                w[y] = w[y] + img[x][y]
               
    
    textFound = 0
    lastBlank = 0
    y=0
    while y < img.shape[1]: 
        blankCount = 0
        #print("img.shape:", img.shape[1])
        while y < img.shape[1] and w[y]==0:   
            blankCount = blankCount+1
            y = y + 1
        #print("blankCount:", str(blankCount))
        if blankCount >=threshold and blankCount > 4: 
            
            if textFound:
                segList.append((lastBlank,y-1))   
                textFound = 0
            lastBlank = y-1                                       
        else:
            textFound = 1
            while y < img.shape[1] and w[y]>0:
                y = y +1
    if textFound ==1:
        segList.append((lastBlank, y-1))
    #print("words segment:", segList)
    for (y0,y1) in segList:
        if y1-y0>4:
            cropped = img[:, y0:y1]
            if np.sum(cropped)!=0:
                ImgsOfWords.append(cropped)
                print("Word cord: ", y0, y1)
    return ImgsOfWords


# In[15]:


def cropLinesImg(img,threshold = 3):
    """
    参数
        img: 黑底白字二值包含多行英文文字图片
        threshold： 判断是空行的阈值
    返回： 
        分割文字行，返回一个list，list里的元素是单个文字行的图片，排列顺序按文字行由上至下顺序
    正确分割的先决条件： 
        文字行之间必须有至少一水平空行， 字体的高度至少5个像素，字体大小大体相等
    """
    ImgOfLines = []
    segList = []
    count = 1;
    #img = correctSkew(img)
    w = [0]*img.shape[0]
    #if (isBinaryImg(img)>0):
        #print("cropLinesImg wrong")
        
    # 对每一行计算投影值
    # 把每一行上像素值加起来，由于是黑底白字而黑色的像素值为0白色为255，考虑到噪点和个别特别长的笔画，
    # 设定一个阈值，如果连续几行上像素值和小于阈值，而且文字行高度大于5就意味着这是文字行间隔
    # 对初始计算出来的文字行间隔的高度比较，
    # 超过2个标准方差的文字行间隔很可能包含多行文字，再重新水平矫正后分割文字行
    # 而小于中位值2个标准方差的文字行间隔很可能是噪点，放弃
    for x in range(img.shape[0]):
        
        for y in range(img.shape[1]): 
                w[x] = w[x] + img[x][y]
            

    textFound = 0
    lastBlank = 0
    #max_value = max(w)
    extreme_min_value = 255*threshold
    x = 0

    while x < img.shape[0]:  
        if (w[x] < 255*threshold): 
            if textFound:   
                extreme_min_value = w[x]
                x = x + 1
                while x < img.shape[0] and w[x]<extreme_min_value:
                    extreme_min_value = w[x]
                    x = x + 1
                segList.append((lastBlank,x-1))
                
                while x < img.shape[0] and w[x]==0:
                    x = x + 1
                lastBlank = x
                while x < img.shape[0] and w[x]<255*threshold:
                    x = x + 1 
                
                textFound = 1
                #extreme_min_value = 255*threshold
        else:                   
            textFound = 1
            #extreme_min_value = 255*threshold
        x = x + 1
    if textFound and lastBlank<x-1-5:
        segList.append((lastBlank,x-1))
 
    heightList = []
    for (x0,x1) in segList:
        
        if x1-x0>5:
            heightList.append(x1-x0)
    minimal = 5
    if (len(heightList)>2):
        stdev = statistics.stdev(heightList)
        median = statistics.median(heightList)        
        maximum = median + 2*stdev
    else:        
        maximum = img.shape[1]
    for (x0,x1) in segList:
        
        cropped = img[x0:x1,:]
        if x1-x0>minimal and x1-x0<maximum:
            
            cropped = cropImg(cropped)               
            cropped = correctSkew(cropped)
            #cropped = correctSlant(cropped)
            #cropped = correctSkew(cropped)
            cropped = cropImg(cropped)
            print("crop lines: ", x0,x1)
            if np.sum(cropped)!=0:
                ImgOfLines.append(cropped)
       
        elif x1-x0>=maximum:   
            #cropped = cropImg(cropped)            
            #cropped = correctSkew(cropped)            
            ImgOfLines = ImgOfLines+cropLinesImg(cropped, threshold)
            
            
    return ImgOfLines
        #elif ((w[y]<w[y+1]) and w[y]<w[y-1]):
         #   segList.append(y)


# In[16]:


def normalizeImg(filename, img,stdRow, stdCol):
    """规范化图片，将图片转换为统一大小的黑底白字图片
    Args:
        filename: 图片文件全路径名
        stdRow: 标准尺寸的行数
        stdCol: 标准尺寸的列数
    Returns:
        统一大小的黑底白字图片
    """

    if not filename==None:
        img = cv2.imread(filename)
 
    if len(img.shape)>2:
        #彩色图片
        grey = blackwhiteImg(img,1)
        #plt.imshow(grey)
    else:
        #灰度图片
        grey = blackwhiteImg(img, 0)
        #plt.imshow(grey)
    if np.sum(grey)!=0:
        grey = cropImg(grey)
    
    #补足因笔墨浓淡不均匀在转为黑的白字后笔画里的小黑点 https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #kernel = np.ones((2,2),np.uint8)
    #grey = cv2.morphologyEx(grey,cv2.MORPH_CLOSE,kernel, iterations = 1)
    #plt.imshow(grey)
    rows, cols = grey.shape
    rowRatio = stdRow/rows
    colRatio = stdCol/cols
    if (rows<=stdRow) and (cols<=stdCol):
        if rowRatio < colRatio:
            grey = cv2.resize(grey, None, fx=rowRatio, fy=rowRatio, interpolation=cv2.INTER_LINEAR) 
        else:
            grey = cv2.resize(grey,None,fx=colRatio,fy=colRatio, interpolation=cv2.INTER_LINEAR)
        grey = blackwhiteImg(grey, 0)
 
    elif (rows>=stdRow) and (cols>=stdCol):
        if rowRatio < colRatio:
            grey = cv2.resize(grey, None, fx=rowRatio, fy=rowRatio, interpolation=cv2.INTER_LINEAR)
  
        else:
            grey = cv2.resize(grey, None, fx=colRatio, fy=colRatio, interpolation=cv2.INTER_LINEAR)
            
    elif (rows>=stdRow) and (cols<=stdCol):
        grey = cv2.resize(grey, None, fx=rowRatio, fy=rowRatio, interpolation=cv2.INTER_LINEAR)
        
    elif (rows<=stdRow) and (cols>=stdCol):
        grey = cv2.resize(grey, None, fx=colRatio, fy=colRatio, interpolation=cv2.INTER_LINEAR)
        
    rows, cols = grey.shape
    #print(rows,cols)
    if stdRow>=rows:
        rowsPadding = (math.ceil((stdRow-rows)/2),math.floor((stdRow-rows)/2))
    if stdCol>=cols:
        colsPadding = (math.ceil((stdCol-cols)/2),math.floor((stdCol-cols)/2))
    grey = np.lib.pad(grey,(rowsPadding,colsPadding),'constant')
    
    
    return grey


# ### 去除图片的噪点
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def deleteOutlier(image):
    print("original shape:", image.shape)
    #补足笔画里的小黑点 
    kernel = np.ones((5,5),np.uint8)
    image1 = cv2.dilate(image,kernel,iterations = 2)
    #plt.imshow(image1)
    image1 = cv2.morphologyEx(image1,cv2.MORPH_CLOSE,kernel, iterations = 2)
    
    kernel1 = np.ones((3,3),np.uint8)
    image1 = cv2.morphologyEx(image1,cv2.MORPH_OPEN,kernel, iterations = 2)
    rowStart=0
    colStart = 0
    
    rowEnd,colEnd = image1.shape
 
    cv2.imwrite('C:/Data/AI/IAM/deleteOutlier.jpg', image1)
    while np.sum(image1[rowStart]) == 0 and rowStart<rowEnd-1:        
        rowStart +=1
    while np.sum(image1[rowEnd-1]) == 0 and rowEnd > rowStart+1:
        rowEnd -=1
    while np.sum(image1[:,colStart])==0 and colStart<colEnd-1:
        colStart +=1

    while np.sum(image1[:,colEnd-1])==0 and colEnd >colStart+1:
        colEnd -=1
    #print("rowStart:", rowStart)          
    img = image[rowStart:rowEnd-1, colStart:colEnd-1]
    
    coords = np.column_stack(np.where(img > 0))
    rows,cols = coords.shape
    
    distance = []
    for i in range(rows):
        distance.append(math.pow(coords[i][0]-rows/2,2)+math.pow(coords[i][1]-cols/2,2))
    stdev = statistics.stdev(distance)
    mean = statistics.mean(distance)
    for i in range(rows):
        if distance[i]>(mean+2*stdev):
            img[coords[i][0]][coords[i][1]] = 0
    return img


def splitWords(filename, path, stdRow, stdCol):
    """分割图片里的单词，规范化单词图片，返回单词图片列表
    参数：
        filename： 原始图片全路径名
        path：单词图片存储目录
        stdRow: 单词图片的行数
        stdCol: 单词图片的列数
    返回：规范的单词图片列表
        
    """
    
    grey = cv2.imread(filename)    
    grey = blackwhiteImg(grey,1)
    
    grey = deleteOutlier(grey)
    print("splitWords: ",grey.shape)
    #grey = cropImg(grey)
    #grey = correctSkew(grey)
  
    #补足笔画里的小黑点 https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    #kernel = np.ones((2,2),np.uint8)
    #grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
 
    #kernel = np.ones((1,1),np.uint8)
    #binaryImg = cv2.morphologyEx(binaryImg,cv2.MORPH_OPEN,kernel, iterations = 2)
    #cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
    #cv2.imshow('original image', grey)
    
    #分割行，返回行图片集
    cv2.imwrite('grey.jpg',grey)
    imgOfLines = cropLinesImg(grey,4)
    
    #如果目录不存在就创建存储行图片和单词图片的目录
    if not os.path.exists(path+'/lines'):
        os.makedirs(path+'/lines')
    if not os.path.exists(path+'/words'):
        os.makedirs(path+'/words')
    
    #对每张行图片，切词并存储行图片和单词图片
    lineCount = 0
    wordCount = 0    
    
    for lineImg in imgOfLines:
        #plt.imshow(lineImg)
        wordCount = 0
        #lineImg = blackwhiteImg(lineImg, 0)
        
        #print("lineImg shape:", lineImg.shape[0])
        imgOfWords = cropWordImg(lineImg,lineImg.shape[0]/6)
        
        cv2.imwrite(path+'/lines/'+'Line'+str(lineCount)+'.jpg', lineImg)
        
        for img in imgOfWords:
            #规范化单词图片，使之符合识别模型的要求
            cv2.imwrite(path+'/beforenorm/'+'Line'+str(lineCount)+'Word'+str(wordCount)+'.jpg',img)
            img = normalizeImg(filename=None, img=img,stdRow=stdRow, stdCol=stdCol)
            cv2.imwrite(path+'/words/'+'Line'+str(lineCount)+'Word'+str(wordCount)+'.jpg',img)
            wordCount = wordCount + 1
        lineCount = lineCount + 1
    #testlines = []
    #testlines = cropLinesImg(lineImg,3)
    #print("final image lines:", len(testlines))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == '__main__': 
    path = "."
    if len(sys.argv) == 1:
        print("Missing image file to process: Python preprocessImg.py filename_image path_store_processed_images")
        
    elif len(sys.argv) == 2:
        print("Start to process image:" + str(sys.argv[1]))
        file = str(sys.argv[1])
        splitWords(str(sys.argv[1]), path, 300, 1024)
    elif len(sys.argv) == 3:
        file = str(sys.argv[1])
        path = str(sys.argv[2])
        splitWords(str(sys.argv[1]), path, 300, 1024)