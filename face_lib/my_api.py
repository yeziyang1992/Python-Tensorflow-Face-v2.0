# import system things
import csv
import itertools
import os
import random
import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

from face_lib import align_dlib, inference

size = 96


class GetAlignedFace:
    def __init__(self, input_dir, output_dir):
        self.PREDICTOR_PATH = './face_lib/shape_predictor_68_face_landmarks.dat'  # 关键点提取模型路径
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.pic_names = self.read_pic_names()

    # 读取文件夹中的图片目录
    def read_pic_names(self):
        pic_names = []
        for filename in os.listdir(self.input_dir):
            pic_names.append(filename)
        return pic_names

    def photo_read(self, path, num):
        # 使用dlib自带的frontal_face_detector作为我们的特征提取器
        detector = align_dlib.AlignDlib(self.PREDICTOR_PATH)
        path = self.input_dir + '/' + path
        print(path + " 正在处理...")
        name_file = str(num) + '_' + path.split('/')[-1]
        name_file = self.output_dir + '/' + name_file
        # 如果不存在目录 就创造目录
        if not os.path.exists(name_file):
            os.makedirs(name_file)
        index = 1

        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                img_path = path + '/' + filename
                print(img_path)
                # 从文件读取图片
                img_bgr = cv2.imread(img_path)  # 从文件读取bgr图片
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转为RGB图片
                face_align = detector.align(size, img_rgb)
                if face_align is None:
                    pass
                else:
                    face_align = cv2.cvtColor(face_align, cv2.COLOR_RGB2BGR)  # 转为BGR图片
                    # 保存图片
                    cv2.imwrite(name_file + '/' + str(index) + '.jpg', face_align)
                    index += 1


class Random:
    @staticmethod
    #输出图片数据，归一化
    def get_image_array(path):
        img = cv2.imread(path)
        img = np.array(img)
        return img.astype('float32') / 255.0

    @staticmethod
    # 读取一个文件夹 返回标签数(文件夹数)、图片路径
    # def get_triplet_data(path):
    #     max_num = len(os.listdir(path))  # 获取人脸标签数
    #     face_array = [[] for n in range(max_num)]  # 初始化二维数组 
    #     for i, filename in enumerate(os.listdir(path)):
    #         for _, img_name in enumerate(os.listdir(path + filename)):
    #             if img_name.endswith('.jpg'):
    #                 path_name = path + filename + '/' + img_name
    #                 face_array[i].append(path_name)
    #     return max_num, face_array
    
    def get_triplet_data(path):
        max_num = len(os.listdir(path))  # 获取人脸标签数
        face_array = []  # 初始化二维数组 
        id_array=[]
        for i, filename in enumerate(os.listdir(path)):
            for _, img_name in enumerate(os.listdir(path + filename)):
                if img_name.endswith('.jpg'):
                    path_name = path + filename + '/' + img_name
                    id_array.append(path_name)
            face_array.append(id_array)
            id_array=[]
        return max_num, face_array
    
    @staticmethod
    # 以随机方式生成3元组训练数据
    def generate_train_data(image_path, num):
        per_data = []
        p = None
        train_data = []
        for i in range(num):
            per_data.clear()
            temp_x = image_path[i]
            random.shuffle(temp_x)    #打乱排序

            if len(temp_x) == 1:
                per_data.append(str(temp_x[0]))
                per_data.append(str(temp_x[0]))   # 加2次 防止图片只有一张时，无法形成三元组
            elif len(temp_x) >= 2:
                per_data.append(str(temp_x[-1]))
                per_data.append(str(temp_x[0]))   # 取2张相同照片
            else:
                continue

            flag = True
            while flag:
                p = random.randint(0, num-1)      # 取第3张图片,且与前2张不属于同一类
                if p != i and len(image_path[p]) != 0:
                    flag = False
            temp_y = image_path[p]
            random.shuffle(temp_y)
            per_data.append(str(temp_y[0]))
            train_data.append(tuple(per_data))
        random.shuffle(train_data)
        return train_data


class LfwTest:
    def __init__(self):
        self.PREDICTOR_PATH = './face_lib/shape_predictor_68_face_landmarks.dat'  # 关键点提取模型路径
        self.path = './temp/lfw/lfw/'
        self.filename_neg = './temp/lfw/lfw_test/negative_pairs.txt'
        self.filename_path = './temp/lfw/lfw_test/Path_lfw2.txt'
        self.filename_pos = './temp/lfw/lfw_test/postive_pairs.txt'
        self.path_array, self.neg_array, self.pos_array = self.get_file()

    def get_file(self):
        neg_array = []
        pos_array = []
        path_array = []
        with open(self.filename_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                path_array.append(lines.replace("\n", ""))
        with open(self.filename_neg, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                p_tmp, e_tmp = [int(i) for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                neg_array.append((p_tmp, e_tmp))
        with open(self.filename_pos, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                p_tmp, e_tmp = [int(i) for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                pos_array.append((p_tmp, e_tmp))
        return path_array, neg_array, pos_array

    def get_pair_image(self, pair):
        detector = align_dlib.AlignDlib(self.PREDICTOR_PATH)
        x, y = pair
        path_name_x = self.path + self.path_array[x-1]
        path_name_y = self.path + self.path_array[y-1]
        x_img = self.get_one_image(x, detector)
        y_img = self.get_one_image(y, detector)
        return x_img, y_img

    def get_one_image(self, x, detector):
        path_name_x = self.path + self.path_array[x-1]
        try:
            img_x = cv2.imread(path_name_x)
        except IndexError:
            print(path_name_x)
            print('error')
        else:
            img_x_rgb = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)  # 转为RGB图片
            face_align_rgb_x = detector.align(size, img_x_rgb)
            if face_align_rgb_x is None:
                det = dlib.get_frontal_face_detector()
                gray_img = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
                # 使用detector进行人脸检测
                dets = det(gray_img, 1)
                if len(dets) > 0:
                    x1 = dets[0].top() if dets[0].top() > 0 else 0
                    y1 = dets[0].bottom() if dets[0].bottom() > 0 else 0
                    x2 = dets[0].left() if dets[0].left() > 0 else 0
                    y2 = dets[0].right() if dets[0].right() > 0 else 0
                    face = img_x[x1:y1, x2:y2]
                else:
                    face = cv2.resize(img_x, (size, size))
                face_align_x = cv2.resize(face, (size, size))
            else:
                face_align_x = cv2.cvtColor(face_align_rgb_x, cv2.COLOR_RGB2BGR)  # 转为BGR图片
            x_img = np.array(face_align_x)
            x_img = x_img.astype('float32') / 255.0
            return x_img


class LfwPlot:
    def __init__(self):
        self.margin = 5.0
        self.x_data=[]
        self.y1_data = []
        self.y2_data = []
        self.y3_data = []

    def plot(self):
        for i in range(1000):
            x = 0.002 * i
            test_num, neg = self.__neg_reader(x)
            train_num, pos = self.__pos_reader(x)
            try:
                p1 = neg / test_num  # 总的准确率
                p2 = pos / train_num  # 总的准确率
                p3 = 2*p1*p2/(p1+p2)
            except ZeroDivisionError:
                p1 = 0
                p2 = 0
                p3 = 0

            self.x_data.append(x)
            self.y1_data.append(p1)
            self.y2_data.append(p2)
            self.y3_data.append(p3)
        # 画图
        plt.plot(self.x_data, self.y1_data, 'r-', lw=1, label="$negative pairs$")
        plt.plot(self.x_data, self.y2_data, 'b-', lw=1, label="$positive pairs$")
        f1score = round(max(self.y3_data),4)
        plt.plot(self.x_data, self.y3_data, 'g-', lw=3, label="$F1 score: $"+str(f1score))
        plt.legend(loc='best') 
        plt.xlabel('margin')
        plt.ylabel('p')
        plt.title("lfw_test")
        plt.grid()
        plt.show()

    def calulate(self):
        data = []
        for i in range(1000):
            x = 0.004 * i
            test_num, neg = self.__neg_reader(x)
            train_num, pos = self.__pos_reader(x)
            try:
                p1 = neg / test_num  # 总的准确率
                p2 = pos / train_num  # 总的准确率
                p3 = 2*p1*p2/(p1+p2)
            except ZeroDivisionError:
                p1 = 0
                p2 = 0
                p3 = 0
            data.append(p3)
        f1score = round(max(data),4)
        return f1score

    @staticmethod
    def __neg_reader(margin):
        csv_test_reader = csv.reader(open('./temp/lfw/result/neg.csv', encoding='utf-8'))
        test_num = 0  # 测试集
        test = 0  # 超出阈值的个数
        for row in csv_test_reader:
            test_num += 1
            x1 = row[0]
            if float(x1) > margin:
                test += 1
        return test_num, test

    @staticmethod
    def __pos_reader(margin):
        csv_test_reader = csv.reader(open('./temp/lfw/result/pos.csv', encoding='utf-8'))
        test_num = 0  # 测试集
        test = 0  # 超出阈值的个数
        for row in csv_test_reader:
            test_num += 1
            x1 = row[0]
            if float(x1) > margin:
                test += 1
        return test_num, test_num-test


