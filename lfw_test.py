from face_lib import my_api, inference
import csv
import multiprocessing
import tensorflow as tf


class LFW_TEST:
    def __init__(self,sia,se):
        self.sia = sia
        self.se = se

    def csv_write(self,path, data):
        out_test = open(path, 'w', newline='')
        csv_test_writer = csv.writer(out_test, dialect='excel')
        for n in data:
            x1, x2 = n.get()
            # 返回图片的128维编码信息
            res = self.se.run(self.sia.look_like, feed_dict={self.sia.x1: [x1],self.sia.x2: [x2],self.sia.keep_f: 1.0})
            csv_test_writer.writerow([res])
        out_test.close()

    def write(self, Flag,n=3000):
        lfw_test = my_api.LfwTest()     # 获取对象
        neg_array = lfw_test.neg_array[:n]
        pos_array = lfw_test.pos_array[:n]
        # path_array = lfw_test.path_array

        neg_data = []
        pos_data = []
        if Flag == 1 or Flag == 0:
            # 开启多进程
            pool = multiprocessing.Pool(processes=6)
            for index in neg_array:
                result = pool.apply_async(lfw_test.get_pair_image, (index,))
                neg_data.append(result)
            pool.close()  # 调用join之前，先调用close函数，否则会出错。
            pool.join()  # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
            print("Negative Down!")
            self.csv_write('./temp/lfw/result/neg.csv', neg_data)
            print('./temp/lfw/result/neg.csv  写入成功!')
        if Flag == 2 or Flag == 0:
            # 开启多进程
            pool = multiprocessing.Pool(processes=6)
            for index in pos_array:
                result = pool.apply_async(lfw_test.get_pair_image, (index,))
                pos_data.append(result)
            pool.close()  # 调用join之前，先调用close函数，否则会出错。
            pool.join()  # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
            print("Positive Down!")
            self.csv_write('./temp/lfw/result/pos.csv', pos_data)
            print('./temp/lfw/result/pos.csv  写入成功!')

    @staticmethod
    def plot():
        # 画图
        print('正在绘制图形')
        lfw_plot = my_api.LfwPlot()
        lfw_plot.plot()

    @staticmethod
    def calculate():
        lfw_plot = my_api.LfwPlot()
        return lfw_plot.calulate()

if __name__ == '__main__':
    size = my_api.size  # 图片大小
    model_file = 'model/train_faces.model'  # 模型存放目录
    # # setup siamese network
    siamese = inference.Siamese(size)
    sess = tf.Session()
    saver = tf.train.Saver()
    # 全局参数初始化
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_file)
    print('模型重载成功')
    test = LFW_TEST(siamese,sess)
    test.write(Flag=0)
    test.plot()
    print(test.calculate())
    sess.close()

 








