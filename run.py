# import system things
from time import ctime
import numpy as np
import tensorflow as tf
import csv
import os
import cv2

# import helpers
from face_lib import my_api, inference
import lfw_test


def get_row_train_data(train_step, saver, step, train_data,  num):
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
                # if loss_step % 10 == 0 :
                #     saver.save(sess, model_file)
                #     print('保存成功')
            image_x1.clear()
            image_x2.clear()
            image_x3.clear()

        x1 = data[0]
        x2 = data[1]
        x3 = data[2]
        image_x1.append(my_api.Random.get_image_array(x1))
        image_x2.append(my_api.Random.get_image_array(x2))
        image_x3.append(my_api.Random.get_image_array(x3))
    return losses


def cnn_train():
    l_rate = 1e-6  # 学习率
    train_step = tf.train.AdamOptimizer(l_rate).minimize(siamese.loss)
    saver = tf.train.Saver()
    # 全局参数初始化
    sess.run(tf.global_variables_initializer())
    # if you just want to load a previously trainmodel?
    new = True  # 第一次训练需要把new改为True
    # if os.path.exists(model_file)
    #if os.path.isfile(model_file):
    # input_var = input("我们发现模型，是否需要预训练 [yes/no]?")
    # if input_var == 'yes':
    #     new = False
    if not new:
        saver.restore(sess, model_file)
        print('模型重载成功')

    loss_sum = 0
    p=0
    for step in range(5000):
        # 计算准确率
        if step%1==0:
            test = lfw_test.LFW_TEST(siamese,sess)
            test.write(0,240)
            p = test.calculate()
            print("准确率： " +str(p))
        # 每次取128(batch_size)张图片
        # 生成训练数据
        train_data = my_api.Random.generate_train_data(face_path, max_num)
        losses = get_row_train_data(train_step, saver, step, train_data, max_num)
        print('step：%s    losses： %s   rate: %s  accuracy: %s' % (step, losses, l_rate, p))
        loss_sum = loss_sum + losses
        # 向csv中写入数据
        with open('./out/random/print_result.csv', 'a+', newline='') as csv_w:
            csv_print_writer = csv.writer(csv_w, dialect='excel')
            csv_print_writer.writerow([ctime(), 'step:', step, '  loss_sum:', loss_sum, '  rate:', l_rate,'  accuracy:',p])
        l_rate = l_rate * 0.95
        loss_sum = 0
        saver.save(sess, model_file)
        print('保存成功')

    saver.save(sess, model_file)
    print('保存成功')
    sess.close()

if __name__ == '__main__':

    my_faces_path = './train_faces/'             
    size = my_api.size                              
    model_file = './model/train_faces.model'        
    # 读取一个文件夹 返回标签数和图片数组
    max_num, face_path = my_api.Random.get_triplet_data(my_faces_path)
    print('标签数： ' + str(max_num))
    # 图片块，每次取100张图片
    batch_size = 128
    # setup siamese network
    siamese = inference.Siamese(size)
    sess = tf.Session()
    cnn_train()




