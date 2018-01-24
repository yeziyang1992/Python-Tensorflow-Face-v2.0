# import system things
from time import ctime
import numpy as np
import tensorflow as tf
import csv

# import helpers
from face_lib import my_api, inference


def get_row_train_data(train_step, saver, step, image_array, train_data,  num):
    loss_step = 0   # 代表进行loss训练的第几批  loss_step = step_i / batch_size
    losses = 0      # 总的loss初始化为0
    image_x1 = []
    image_x2 = []
    image_x3 = []
    # 每次取(batch_size)张图片
    for step_i, data in enumerate(train_data):
        if step_i > 1 and step_i % batch_size == 0:
            loss_step += 1
            if loss_step >= 00:
                train_anc = np.array(image_x1)
                train_pos = np.array(image_x2)
                train_neg = np.array(image_x3)

                _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: train_anc,
                            siamese.x2: train_pos,
                            siamese.x3: train_neg,
                            siamese.keep_f: 1.0})
                losses = losses + loss_v
                print('time %s  step %d, %d: loss %.4f  losses %.4f' % (ctime(), step, loss_step, loss_v, losses))
                if loss_step % 100 == 0 and loss_v <= 0.002:
                    saver.save(sess, model_file)
                    print('保存成功')
            image_x1.clear()
            image_x2.clear()
            image_x3.clear()

        x1 = data[0]
        x2 = data[1]
        x3 = data[2]
        id_x1 = x1.split('_')[0]
        id_y1 = x1.split('_')[1]
        id_x2 = x2.split('_')[0]
        id_y2 = x2.split('_')[1]
        id_x3 = x3.split('_')[0]
        id_y3 = x3.split('_')[1]
        for i in range(num):
            if i == int(id_x1):
                for j, img in enumerate(image_array[i]):
                    if j == int(id_y1):
                        image_x1.append(img)
            if i == int(id_x2):
                for j, img in enumerate(image_array[i]):
                    if j == int(id_y2):
                        image_x2.append(img)
            if i == int(id_x3):
                for j, img in enumerate(image_array[i]):
                    if j == int(id_y3):
                        image_x3.append(img)
    return losses


def cnn_train():
    l_rate = 1e-5  # 学习率
    train_step = tf.train.AdamOptimizer(l_rate).minimize(siamese.loss)
    saver = tf.train.Saver()
    # 全局参数初始化
    sess.run(tf.global_variables_initializer())
    # if you just want to load a previously trainmodel?
    new = True
    input_var = input("我们发现模型，是否需要预训练 [yes/no]?")
    if input_var == 'yes':
        new = False
    if not new:
        saver.restore(sess, model_file)
        print('模型重载成功')
    loss_sum = 0
    for step in range(5000):
        # 每次取128(batch_size)张图片
        # 生成训练数据
        train_data = my_api.Random.generate_train_data(id_array, max_num)
        losses = get_row_train_data(train_step, saver, step, np_img, train_data, max_num)
        print('step：%s    losses： %s   rate: %s' % (step, losses, l_rate))
        loss_sum = loss_sum + losses
        if step > 1 and step % 3 == 0:
            # 向csv中写入数据
            with open('./out/random/print_result.csv', 'a+', newline='') as csv_w:
                csv_print_writer = csv.writer(csv_w, dialect='excel')
                csv_print_writer.writerow([ctime(), 'step:', step, '    loss_sum:', loss_sum, '   rate:', l_rate])
            l_rate = l_rate * 0.8
            loss_sum = 0
            saver.save(sess, model_file)
            print('保存成功')
    saver.save(sess, model_file)
    print('保存成功')
    sess.close()

if __name__ == '__main__':
    my_faces_path = './train_faces/'                # 人脸数据集目录
    out_train_faces_path = './out/train_data.csv'   # 训练数据csv文件存放目录
    size = my_api.size                              # 图片大小
    model_file = 'model/random/train_faces.model'          # 模型存放目录
    # 读取一个文件夹 返回标签数和图片数组
    max_num, face_array, id_array = my_api.Traversal.get_triplet_data(my_faces_path)
    print('标签数： ' + str(max_num))
    # 将图片数据转换成np数组
    np_img = np.array(face_array)
    # 图片块，每次取100张图片
    batch_size = 16
    # setup siamese network
    siamese = inference.Siamese(size)
    sess = tf.Session()
    cnn_train()




