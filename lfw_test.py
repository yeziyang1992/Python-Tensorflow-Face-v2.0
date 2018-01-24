from face_lib import my_api, inference
import csv
import multiprocessing
import tensorflow as tf


def csv_write(path, data):
    out_test = open(path, 'w', newline='')
    csv_test_writer = csv.writer(out_test, dialect='excel')
    i = 0
    for n in data:
        i += 1
        print(i)
        x1, x2 = n.get()
        print(x1.shape, x2.shape)
        # 返回图片的128维编码信息
        res = sess.run(siamese.look_like, feed_dict={
            siamese.x1: [x1],
            siamese.x2: [x2],
            siamese.keep_f: 1.0})
        print(res)
        csv_test_writer.writerow([res])
    out_test.close()

if __name__ == '__main__':

    Flag = 3
    size = my_api.size  # 图片大小
    model_file = 'model/random/train_faces.model'  # 模型存放目录
    # setup siamese network
    siamese = inference.Siamese(size)
    sess = tf.Session()
    saver = tf.train.Saver()
    # 全局参数初始化
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_file)
    print('模型重载成功')

    lfw_test = my_api.LfwTest()     # 获取对象
    neg_array = lfw_test.neg_array
    pos_array = lfw_test.pos_array
    path_array = lfw_test.path_array

    print(len(neg_array))
    print(len(pos_array))
    print(len(path_array))

    neg_data = []
    pos_data = []
    if Flag == 1 or Flag == 0:
        # 开启多进程
        pool = multiprocessing.Pool(processes=4)
        for index in neg_array:
            result = pool.apply_async(lfw_test.get_pair_image, (index,))
            neg_data.append(result)
        pool.close()  # 调用join之前，先调用close函数，否则会出错。
        pool.join()  # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        print("Negative Down!")
        csv_write('./temp/lfw/result/neg.csv', neg_data)
        print('./temp/lfw/result/neg.csv  写入成功!')
    if Flag == 2 or Flag == 0:
        # 开启多进程
        pool = multiprocessing.Pool(processes=4)
        for index in pos_array:
            result = pool.apply_async(lfw_test.get_pair_image, (index,))
            pos_data.append(result)
        pool.close()  # 调用join之前，先调用close函数，否则会出错。
        pool.join()  # 执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
        print("Positive Down!")
        csv_write('./temp/lfw/result/pos.csv', pos_data)
        print('./temp/lfw/result/pos.csv  写入成功!')

    sess.close()
    if Flag == 3:
        # 画图
        print('正在绘制图形')
        plot = my_api.LfwPlot()
        plot.neg_cac()
        plot.pos_cac()




