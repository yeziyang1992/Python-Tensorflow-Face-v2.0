import tensorflow as tf


class Siamese:

    # Create model
    def __init__(self, size):
        self.x1 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.x2 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.x3 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.d1 = tf.placeholder(tf.float32, [None, 128])
        self.d2 = tf.placeholder(tf.float32, [None, 128])
        self.d3 = tf.placeholder(tf.float32, [None, 128])
        self.keep_f = tf.placeholder(tf.float32)

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1, self.keep_f)
            scope.reuse_variables()
            self.o2 = self.network(self.x2, self.keep_f)
            scope.reuse_variables()
            self.o3 = self.network(self.x3, self.keep_f)

        # Create loss
        self.loss = self.loss_with_spring()
        self.look_like = self.look_like()

    def network(self, x, keep_f):
        with tf.variable_scope("conv1"):
            # 第一层卷积，此时输入图片大小为96*96,输出图像大小为96*96
            conv1 = self.cnn_layer(x, [3, 3, 3, 64], [64])
        with tf.variable_scope("conv2"):
            # 第二层卷积，此时输入图片大小为96*96,输出图像大小为48*48
            conv2 = self.cnn_layer(conv1, [3, 3, 64, 64], [64])
            pool1 = self.pool_layer(conv2, 1.0)

        with tf.variable_scope("conv3"):
            # 第三层卷积，此时输入图片大小为48*48,输出图像大小为48*48
            conv3 = self.cnn_layer(pool1, [3, 3, 64, 128], [128])
        with tf.variable_scope("conv4"):
            # 第四层卷积，此时输入图片大小为48*48,输出图像大小为24*24
            conv4 = self.cnn_layer(conv3, [3, 3, 128, 128], [128])
            pool2 = self.pool_layer(conv4, 1.0)

        with tf.variable_scope("conv5"):
            # 第五层卷积，此时输入图片大小为24*24,输出图像大小为24*24
            conv5 = self.cnn_layer(pool2, [3, 3, 128, 256], [256])
        with tf.variable_scope("conv6"):
            # 第六层卷积，此时输入图片大小为24*24,输出图像大小为24*24
            conv6 = self.cnn_layer(conv5, [3, 3, 256, 256], [256])
        with tf.variable_scope("conv7"):
            # 第七层卷积，此时输入图片大小为24*24,输出图像大小为12*12
            conv7 = self.cnn_layer(conv6, [3, 3, 256, 256], [256])
            pool3 = self.pool_layer(conv7, 1.0)

        with tf.variable_scope("conv8"):
            # 第八层卷积，此时输入图片大小为12*12,输出图像大小为12*12
            conv8 = self.cnn_layer(pool3, [3, 3, 256, 512], [512])
        with tf.variable_scope("conv9"):
            # 第九层卷积，此时输入图片大小为12*12,输出图像大小为12*12
            conv9 = self.cnn_layer(conv8, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv10"):
            # 第十层卷积，此时输入图片大小为12*12,输出图像大小为6*6
            conv10 = self.cnn_layer(conv9, [3, 3, 512, 512], [512])
            pool4 = self.pool_layer(conv10, 1.0)

        with tf.variable_scope("conv11"):
            # 第十一层卷积，此时输入图片大小为6*6,输出图像大小为6*6
            conv11 = self.cnn_layer(pool4, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv12"):
            # 第十二层卷积，此时输入图片大小为6*6,输出图像大小为6*6
            conv12 = self.cnn_layer(conv11, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv13"):
            # 第十三层卷积，此时输入图片大小为6*6,输出图像大小为3*3
            conv13 = self.cnn_layer(conv12, [3, 3, 512, 512], [512])
            pool5 = self.pool_layer(conv13, 1.0)

        with tf.variable_scope("full_layer1"):
            # 全连接层1，此时输入图片大小为7*7
            f1 = self.full_layer(pool5, [3 * 3 * 512, 1024], [1024], keep_f, True)
        with tf.variable_scope("full_layer2"):
            # 全连接层2，此时输入2048
            f2 = self.full_layer(f1, [1024, 512], [512], keep_f)
        with tf.variable_scope("full_layer3"):
            # 全连接层2，此时输入2048
            f3 = self.full_layer(f2, [512, 128], [128], 1.0)
        return f3

    @staticmethod
    def cnn_layer(input_image, kernel_shape, bias_shape):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=kernel_shape,
                                  initializer=init)

        biases = tf.get_variable("cnn_biases", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))
        conv = tf.nn.conv2d(input_image, weights,
                            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + biases)

    @staticmethod
    def pool_layer(input_image, keep):
        pool = tf.nn.max_pool(input_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        drop = tf.nn.dropout(pool, keep)
        return drop

    @staticmethod
    def full_layer(input_image,  kernel_shape, bias_shape, keep, reshape=False):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=kernel_shape,
                                  initializer=init)

        biases = tf.get_variable("cnn_biases", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))
        if reshape:
            input_image = tf.reshape(input_image, [-1, 3*3*512])
        dense = tf.nn.relu(tf.matmul(input_image, weights) + biases)
        drop = tf.nn.dropout(dense, keep)
        return drop

    def loss_with_spring(self):

        margin = 5.0
        anchor_output = self.o1     # shape [None, 128]
        positive_output = self.o2   # shape [None, 128]
        negative_output = self.o3   # shape [None, 128]

        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1, name="d_pos")
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1, name="d_neg")

        losses = tf.maximum(0., margin + d_pos - d_neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")

        return loss

    def look_like(self):
        anchor_output = self.o1  # shape [None, 128]
        positive_output = self.o2  # shape [None, 128]

        d_look = tf.reduce_sum(tf.square(anchor_output - positive_output), 1, name="d_look")
        distance = tf.reduce_mean(d_look, name="distance")

        return distance





