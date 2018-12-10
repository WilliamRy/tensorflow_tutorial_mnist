import tensorflow as tf

adam_beta1 = 0.9
adam_beta2 = 0.999
learning_rate = 0.003

class BasicDeepLearningModel:
    def __init__(self):
        self.build()

    def build(self):
        self.add_placeholders()
        self.add_body()
        self.add_loss()
        self.add_optimizer()
        self.add_stat()

    def add_placeholders(self):
        self.input = tf.placeholder(tf.float32, [None, 28, 28])
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        self.step = tf.placeholder(tf.int32)

    def add_body(self):
        with tf.variable_scope('model'):
            input_ = tf.expand_dims(self.input, axis=-1)  # 28,1
            conv = tf.layers.conv2d(input_,
                                    filters=8,
                                    kernel_size=(5, 5),
                                    padding='same',
                                    activation=tf.nn.tanh,
                                    name='conv_layer1')  # 28,32
            pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling1')  # 14,8

            conv = tf.layers.conv2d(pooling,
                                    filters=4,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation=tf.nn.tanh,
                                    name='conv_layer2')  # 14,4
            pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling2')  # 7,4

            conv = tf.layers.conv2d(pooling,
                                    filters=10,
                                    kernel_size=(7, 7),
                                    padding='valid',
                                    activation=tf.nn.tanh,
                                    name='conv_final')  # 1,10
            self.logits = tf.squeeze(conv, axis=(1, 2))  # [None, 10]
            self.predict_proba = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(self.predict_proba, 1)

    def add_loss(self):
        with tf.variable_scope('my_loss') as scope:
            self.loss_each = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_, logits=self.logits)
            self.loss = tf.reduce_mean(self.loss_each)

    def add_optimizer(self):
        self.learning_rate = 0.0001 + tf.train.exponential_decay(learning_rate=0.03, global_step=self.step,
                                                                 decay_steps=300, decay_rate=2 / 2.713)
        # self.learning_rate = _learning_rate_decay(0.01, self.step)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, adam_beta1, adam_beta2)
        self.train_step = self.optimizer.minimize(self.loss)

    def add_stat(self):
        with tf.variable_scope('stats'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('lr', self.learning_rate)
            self.stat = tf.summary.merge_all()


class Official_CNN(BasicDeepLearningModel):

    # neural network structure for this sample:
    #
    # · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 28, 28, 1]
    # @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 6x6x1=>24 stride 1      W1 [5, 5, 1, 24]        B1 [24]
    # ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 28, 28, 6]
    #   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>48 stride 2      W2 [5, 5, 6, 48]        B2 [48]
    #   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
    #     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>64 stride 2     W3 [4, 4, 12, 64]       B3 [64]
    #     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
    #      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
    #       · · · ·                                                    Y4 [batch, 200]
    #       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
    #        · · ·                                                     Y [batch, 10]

    def add_placeholders(self):
        # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
        self.input = tf.placeholder(tf.float32, [None, 28, 28])
        # correct answers will go here
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        # test flag for batch norm
        self.is_training = tf.placeholder(tf.bool)
        self.step = tf.placeholder(tf.int32)
        # dropout probability
        self.pkeep = tf.placeholder(tf.float32)
        self.pkeep_conv = tf.placeholder(tf.float32)

    def add_body(self):
        self.X = tf.expand_dims(self.input, axis=-1)
        # three convolutional layers with their channel counts, and a
        # fully connected layer (tha last layer has 10 softmax neurons)
        K = 24  # first convolutional layer output depth
        L = 48  # second convolutional layer output depth
        M = 64  # third convolutional layer
        N = 200  # fully connected layer

        with tf.variable_scope('model'):

            W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
            self.W1 = W1
            B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
            W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
            B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
            W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
            B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

            W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
            B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
            W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
            B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

            # The model
            # batch norm scaling is not useful with relus
            # batch norm offsets are used instead of biases
            stride = 1  # output is 28x28
            Y1l = tf.nn.conv2d(self.X, W1, strides=[1, stride, stride, 1], padding='SAME')
            Y1bn, update_ema1 = self.batchnorm(Y1l, self.is_training, self.step, B1, convolutional=True)
            Y1r = tf.nn.relu(Y1bn)
            Y1 = tf.nn.dropout(Y1r, self.pkeep_conv, self.compatible_convolutional_noise_shape(Y1r), seed = 0)
            stride = 2  # output is 14x14
            Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
            Y2bn, update_ema2 = self.batchnorm(Y2l, self.is_training, self.step, B2, convolutional=True)
            Y2r = tf.nn.relu(Y2bn)
            Y2 = tf.nn.dropout(Y2r, self.pkeep_conv, self.compatible_convolutional_noise_shape(Y2r), seed = 0)
            stride = 2  # output is 7x7
            Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
            Y3bn, update_ema3 = self.batchnorm(Y3l, self.is_training, self.step, B3, convolutional=True)
            Y3r = tf.nn.relu(Y3bn)
            Y3 = tf.nn.dropout(Y3r, self.pkeep_conv, self.compatible_convolutional_noise_shape(Y3r), seed = 0)

            # reshape the output from the third convolution for the fully connected layer
            YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

            Y4l = tf.matmul(YY, W4)
            Y4bn, update_ema4 = self.batchnorm(Y4l, self.is_training, self.step, B4)
            Y4r = tf.nn.relu(Y4bn)
            Y4 = tf.nn.dropout(Y4r, self.pkeep, seed = 0)
            self.logits = tf.matmul(Y4, W5) + B5
            self.output = tf.nn.softmax(self.logits)

            self.predict = tf.argmax(self.output, 1)
            self.predict_onehot = tf.one_hot(self.predict, depth=10)
            self.confuse = tf.reduce_sum(tf.matmul(tf.expand_dims(self.predict_onehot, -1),
                                                   tf.expand_dims(self.Y_, 1)), 0)
            self.accuracy = tf.reduce_sum(tf.trace(self.confuse))

            self.update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

    def add_stat(self):
        with tf.variable_scope('stats') as scope:
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accu', self.accuracy)
            tf.summary.scalar('lr', self.learning_rate)
            for i in range(24):
                image = self.W1[:, :, :, i:i + 1]
                image = tf.reshape(image, (1, 6, 6, 1))
                tf.summary.image('image_' + str(i), image)
            return tf.summary.merge_all()

    def batchnorm(self, Ylogits, is_training, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_training, lambda: mean, lambda: exp_moving_avg.average(mean))
        v = tf.cond(is_training, lambda: variance, lambda: exp_moving_avg.average(variance))
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def no_batchnorm(self, Ylogits, is_test, iteration, offset, convolutional=False):
        return Ylogits, tf.no_op()

    def compatible_convolutional_noise_shape(self, Y):
        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
        return noiseshape

class MyCNN(BasicDeepLearningModel):

    def add_body(self):
        input_ = tf.expand_dims(self.input, axis=-1)  # 28,1
        conv = self.incept(input_, tf.nn.tanh, 'incept1')  # 28,32
        conv = self.incept(conv, tf.nn.tanh, 'incept2', filter_list=[8, 4, 4], kernel_size_list=[1, 3, 5])  # 28,16

        pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling1')  # 14,16

        conv = self.incept(pooling, tf.nn.tanh, 'incept3', filter_list=[8, 4, 4], kernel_size_list=[1, 3, 5])  # 14,16
        pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling2')  # 7,16

        conv = tf.layers.conv2d(pooling,
                                filters=32,
                                kernel_size=(1, 1),
                                padding='same',
                                activation=tf.nn.tanh,
                                name='dense_layer')  # 7,16

        conv = tf.layers.conv2d(conv,
                                filters=10,
                                kernel_size=(7, 7),
                                padding='valid',
                                activation=tf.nn.tanh,
                                name='conv_final')  # 1,10

        self.logits = tf.squeeze(conv, axis=(1, 2))  # [None, 10]
        self.output = tf.nn.softmax(self.logits)
        self.predict = tf.argmax(self.output, 1)
        self.predict_onehot = tf.one_hot(self.predict, depth=10)
        self.confuse = tf.reduce_sum(tf.matmul(tf.expand_dims(self.predict_onehot, -1),
                                               tf.expand_dims(self.Y_, 1)), 0)
        self.accuracy = tf.reduce_sum(tf.trace(self.confuse))

    def add_stat(self):
        with tf.variable_scope('stats'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accu', self.accuracy)
            tf.summary.scalar('lr', self.learning_rate)
            self.stat = tf.summary.merge_all()

    def incept(self, input, activation, scope_name, filter_list=[16, 8, 8], kernel_size_list=[3, 5, 7]):
        assert len(filter_list) == len(kernel_size_list)
        with tf.variable_scope(scope_name) as scope:
            convs = []
            for filter, kernel_size in zip(filter_list, kernel_size_list):
                convs.append(tf.layers.conv2d(input,
                                              filters=filter,
                                              kernel_size=(kernel_size, kernel_size),
                                              padding='same',
                                              activation=activation))
            res = tf.concat(convs, axis=-1)
        return res


class Model:
    def __init__(self, sess = None, summary_writer = None):
        self.sess = sess
        self.summary_writer = summary_writer
        self.nn = BasicDeepLearningModel()
        self.nstep_summary = 0
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, y, k=0, batch_size=31, nepoch=100):
        for X_, Y_ in zip(batch_iterator(X, batch_size=batch_size, nepoch=nepoch),
                          batch_iterator(y, batch_size=batch_size, nepoch=nepoch)):
            loss, _ = self.sess.run([self.nn.loss, self.nn.train_step],
                                      feed_dict={self.nn.input: X_, self.nn.Y_: Y_, self.nn.step: k})

            if self.summary_writer is not None:
                stat = self.sess.run(self.nn.stat,
                                     feed_dict={self.nn.input: X_, self.nn.Y_: Y_, self.nn.step: k})
                self.summary_writer.add_summary(stat, self.nstep_summary)

            self.nstep_summary += 1
            k += 1
        return None

    def predict(self, X):
        pred = self.sess.run(self.nn.predict, feed_dict = {self.nn.input: X})
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        real = y
        score = sum(pred == real) * 1.0 / len(real)
        return score

    def predict_proba(self, X):
        pred = self.sess.run(self.nn.predict_proba, feed_dict={self.nn.input: X})
        return pred



def batch_iterator(lst, batch_size=100, nepoch=15000):
    n = 0
    for i in range(nepoch):
        res = []
        for k in range(batch_size):
            if n >= len(lst):
                n = 0
            res.append(lst[n]);
            n += 1
        yield res

def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 1000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

