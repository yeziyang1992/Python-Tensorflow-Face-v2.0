# Python-Tensorflow-Face-v2.0
1. 本文以Siamese网络为基础，改进了原有的通过训练分类器来检测人脸的方法，大大提高了识别的准确度。   
2. 本人使用数据集为LFW,在temp/lfw/lfw中。     
3. 程序运行流程：          
  3.1. 首先运行get_align_face.py文件。生成对齐过的人脸训练数据。  
  3.2. 运行run_new.py文件，会出现 “我们发现模型，是否需要预训练 [yes/no]?”第一次训练输入no，按回车，即开始训练。这样做的目的是可以间断训练，每次训练完一段时间退出，可以接着训练。   
  3.3. 训练完成后，首先改lfw_test.py中的Flag=0,然后运行，运行完成后改Falg=3,再运行，即可看到准确率的表格。
4. 文件夹结构：
    - face_lib
         * ...
    - model
         - random
             * ...
         - traversal
             * ...
    - out
         - random
             * ...
         - traversal
             * ...
    - temp
         - lfw
              - lfw
              - lfw_test
              - result
    - train_faces
         * ...

