# Python-Tensorflow-Face-v2.0
1. 本文以Triplet网络为基础，改进了原有的通过训练分类器来检测人脸的方法，大大提高了识别的准确度。   
2. 本人使用训练数据集为微软MS-Celeb-1M数据集,[下载地址](https://www.msceleb.org/download/aligned)   
3. 下载的数据集为.tsv格式,需要还原成jpg格式图片。需要把extracted.py拷贝到tsv文件同目录下，然后运行：  
    $ python extracted.py --outputDir=data FaceImageCroppedWithAlignment.tsv  
    尽量留有足够大的硬盘空间。
4. 还需要对数据集进行清理，可到这[链接](https://pan.baidu.com/s/1JfqPCL6vMABbX71WpGDUpg) (密码：9ykp)下载干净的列表，并把下载的txt文件、faceshutil.py和data文件夹放在同一目录下，运行py文件即可
5. 把get_align_face.py放到out同目录下，进行最后的人脸对齐。得到的train_faces文件夹放在项目同级文件夹下。
6. 程序运行流程：          
  3.1. 首先运行makefile.py文件，生成一些目录。 
  3.2. 在[链接](https://pan.baidu.com/s/1F6w8JIzg6o61D2sNmJ9tLw)( 密码：8tgk)下载LFW数据集，放在 ./temp/lfw/文件夹下，最后的目录是./tmp/lfw/lfw。这是为了测试准确率。
  3.2. 运行run.py文件，如果想要继续接着训练,需要把new改为True.这样做的目的是可以间断训练，每次训练完一段时间退出，可以接着训练。   
  3.3. 训练完成后，可以运行lfw_test.py文件查看准确率。
7. 如有问题探讨，邮箱联系：yeziyang1992@163.com

