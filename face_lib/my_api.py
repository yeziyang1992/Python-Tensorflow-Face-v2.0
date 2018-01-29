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


class Traversal:

    @staticmethod
    # 读取一个文件夹 返回标签数(文件夹数)、图片数组和图片id
    def get_triplet_data(path):
        max_num = len(os.listdir(path))  # 获取人脸标签数
        face_array = [[] for n in range(max_num)]  # 初始化二维数组
        id_array = [[] for n in range(max_num)]  # 初始化二维数组
        for i, filename in enumerate(os.listdir(path)):
            for j, img_name in enumerate(os.listdir(path + filename)):
                if img_name.endswith('.jpg'):
                    path_name = path + filename + '/' + img_name
                    img = cv2.imread(path_name)
                    img = np.array(img)
                    face_array[i].append(img.astype('float32') / 255.0)   # 归一化
                    id_array[i].append(j)
        return max_num, face_array, id_array

    @staticmethod
    # 以遍历方式生成3元组训练数据
    def generate_train_data(image_array, num, csv_file):
        per_data = []
        out = open(csv_file, 'w', newline='')
        csv_writer = csv.writer(out, dialect='excel')
        for i in range(num):
            if len(image_array[i]) == 1:            # 一个标签只有一种照片的情况
                per_data.clear()
                per_data.append(str(i)+'_'+str(image_array[i][0]))
                per_data.append(str(i)+'_'+str(image_array[i][0]))   # 加2次 防止图片只有一张时，无法形成三元组
                for n in range(num):
                    if n == i:                       # 防止出现3张图片都属于同一类
                        break
                    for data in image_array[n]:
                        per_data.append((str(n)+'_'+str(data)))
                        csv_writer.writerow(tuple(per_data))
                        per_data.pop()

            elif len(image_array[i]) >= 2:              # 一个标签有多种照片的情况
                for data_list in itertools.combinations(image_array[i], 2):
                    per_data.clear()
                    per_data.append(str(i)+'_'+str(list(data_list)[0]))
                    per_data.append(str(i)+'_'+str(list(data_list)[1]))
                    for n in range(num):
                        if n != i:                       # 防止出现3张图片都属于同一类
                            for data in image_array[n]:
                                per_data.append((str(n)+'_'+str(data)))
                                csv_writer.writerow(tuple(per_data))
                                per_data.pop()


class Random:
    @staticmethod
    def generate_train_data(image_array, num):
        per_data = []
        p = None
        train_data = []
        for i in range(num):
            per_data.clear()
            temp_x = image_array[i]
            random.shuffle(temp_x)
            if len(temp_x) == 1:
                per_data.append(str(i)+'_'+str(temp_x[0]))
                per_data.append(str(i)+'_'+str(temp_x[0]))   # 加2次 防止图片只有一张时，无法形成三元组
            elif len(temp_x) >= 2:
                per_data.append(str(i)+'_'+str(temp_x[1]))
                per_data.append(str(i)+'_'+str(temp_x[0]))   # 加2次 防止图片只有一张时，无法形成三元组
            else:
                continue

            flag = True
            while flag:
                p = random.randint(0, num-1)      # 取第3张图片,且与前2张不属于同一类
                if p != i and len(image_array[p]) != 0:
                    flag = False
            temp_y = image_array[p]
            random.shuffle(temp_y)
            per_data.append(str(p) + '_' + str(temp_y[0]))
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
        print(x)
        print(y)
        path_name_x = self.path + self.path_array[x-1]
        path_name_y = self.path + self.path_array[y-1]
        print(path_name_x)
        print(path_name_y)
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
        plt.figure(1)

    def neg_cac(self):
        x_data = []
        y1_data = []
        margin = 5
        for i in range(500):
            x = margin + 0.2 * i
            test_num, test = self.__neg_reader(x)
            try:
                p1 = test / test_num  # 总的准确率
            except ZeroDivisionError:
                p1 = 0

            x_data.append(x)
            y1_data.append(p1)
        # 画图
        plt.subplot(211)  # the first subplot in the first figure
        plt.plot(x_data, y1_data, 'r-', lw=1, label="$p1$")
        plt.title("negative_pairs")
        plt.ylabel("p")
        plt.grid(True)
        plt.legend()

    def pos_cac(self):
        x_data = []
        y1_data = []
        margin = 5
        for i in range(500):
            x = margin + 0.2 * i
            train_num, pos = self.__pos_reader(x)
            try:
                p1 = pos / train_num  # 总的准确率
            except ZeroDivisionError:
                p1 = 0
            x_data.append(x)
            y1_data.append(p1)

        # 画图
        plt.subplot(212)  # the second subplot in the first figure
        plt.plot(x_data, y1_data, 'r-', lw=1, label="$p1$")
        plt.title("positive_pairs")
        plt.xlabel("margin")
        plt.ylabel("p")
        plt.grid(True)
        plt.legend()
        plt.show()

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


