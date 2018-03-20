# Word-Recognition-Using-IAM-Dataset
环境和数据要求

IAM 的word数据集和word.txt需要提前下载解压 IAM 的word数据我们进行了清洗和预处理，只选出了满足一下条件的单词图片用于训练，

    标注正确的图片，IAM 说明文件word.txt里注明了哪些图片的标注是正确的

    标注为英语单词的图片

    符合以上两个条件并且图片数量大于50张的单词

    对每个单词，随机选取10%的图片用于测试，其他90%用于训练

对于入选的单词图片，确定统一的图片分辨率（100，300），把图片转换为黑底白字100x300的图片

清洗后一共39618张训练图片，4487张测试图片，只有166个不同的单词，每个单词的图片数量都不一样，有的多，有的少

同样待识别的图片必须是黑底白字100x300的图片

依赖的Python包: 

    tensorflow==1.3
  
    pillow 
  
    matplotlib 
  
    opencv
  
    numpy
  
    matplotlib
  
  
此程序使用Tensorflow的Estimator API 和 Dataset API，这两个新的API可以简化模型的创建

Estimator API 入门指南
https://www.tensorflow.org/get_started/custom_estimators

Dataset API 入门指南
https://www.tensorflow.org/get_started/datasets_quickstart

参考：

1. U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002
