# Word-Recognition-Using-IAM-Dataset
## Prepare Data


Before execute IAM_Dataset_Preparation.ipynb, Unzip IAM dataset to get all word images and word.txt file
What IAM_Dataset_Preparation.ipynb does:
- pick up those images that meet below conditions
  - stated in word.txt that the word segment result is ok 
  - Images of English word 
  - number of images >50
  - for each word, 
     - convert its images to black/white binary image
     - resize its images to 100 x 300
     - randomly pick up 10% of its images for test, and the rest 90% for training  

As a result, there are total 39618 images for training and 4487 images for test, and total 166 different words.


## Packages to import: 

+ tensorflow==1.3 
+ pillow
+ matplotlib
+ opencv
+ numpy
+ matplotlib
  
## Model based on CNN 

cnn/CNN_Classification_IAM_smallsize.ipynb 
Use Tensorflow的Estimator API 和 Dataset API to simplify model


Estimator API 
https://www.tensorflow.org/get_started/custom_estimators

Dataset API 
https://www.tensorflow.org/get_started/datasets_quickstart

## Model based on CNN LSTM CTC

## reference：

1. U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002
