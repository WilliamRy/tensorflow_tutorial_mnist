import tensorflow as tf

adam_beta1 = 0.9
adam_beta2 = 0.999
learning_rate = 0.003


class CNN:
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 28, 28])
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        self.step = tf.placeholder(tf.int32)
        self.build()
        self.add_loss()
        self.add_optimizer()
        self.stat = self.add_stat()
        self.is_training = False
        self.nstep_summary = 0

    def build(self):
        input_ = tf.expand_dims(self.input, axis=-1)  # 28,1
        conv = self.incept(input_, 'tanh', 'incept1')  # 28,32
        conv = self.incept(conv, 'tanh', 'incept2', filter_list=[8, 4, 4], kernel_size_list=[1, 3, 5])  # 28,16

        pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling1')  # 14,16

        conv = self.incept(pooling, 'tanh', 'incept3', filter_list=[8, 4, 4], kernel_size_list=[1, 3, 5])  # 14,16
        pooling = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), name='pooling2')  # 7,16

        conv = tf.layers.conv2d(pooling,
                                filters = 32,
                                kernel_size=(1,1),
                                padding = 'same',
                                activation='tanh',
                                name = 'dense_layer') #7,16


        conv = tf.layers.conv2d(conv,
                                filters=10,
                                kernel_size=(7, 7),
                                padding='valid',
                                activation='tanh',
                                name='conv_final')  # 1,10

        self.logits = tf.squeeze(conv, axis=(1, 2))  # [None, 10]
        self.output = tf.nn.softmax(self.logits)
        self.predict = tf.argmax(self.output, 1)
        self.predict_onehot = tf.one_hot(self.predict, depth = 10)
        self.confuse = tf.reduce_sum(tf.matmul(tf.expand_dims(self.predict_onehot, -1),
                                               tf.expand_dims(self.Y_, 1)), 0)
        self.accuracy = tf.reduce_sum(tf.trace(self.confuse))

    def add_loss(self):
        with tf.variable_scope('my_loss') as scope:
            self.loss_each = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_, logits=self.logits)
            self.loss = tf.reduce_mean(self.loss_each)

    def add_optimizer(self):
        self.learning_rate = 0.0001 + tf.train.exponential_decay(learning_rate = 0.03, global_step=self.step,
                                   decay_steps = 300, decay_rate= 2/2.713)
        #self.learning_rate = _learning_rate_decay(0.01, self.step)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, adam_beta1, adam_beta2)
        self.train_step = self.optimizer.minimize(self.loss)

    def add_stat(self):
        with tf.variable_scope('stats') as scope:
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accu', self.accuracy)
            tf.summary.scalar('lr', self.learning_rate)
            return tf.summary.merge_all()

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


    def train_and_test(self, train_data, test_data, sess, batch_size=100, nepoch=15000, summary_writer=None):

        sess.run(tf.global_variables_initializer())
        losses = []
        k = 0
        for X, Y in zip(batch_iterator(train_data['X'],batch_size=batch_size, nepoch=nepoch),
                        batch_iterator(train_data['Y'],batch_size=batch_size, nepoch=nepoch)):
            stat, loss, step = sess.run([self.stat, self.loss, self.train_step],
                                        feed_dict={self.input: X, self.Y_: Y, self.step: k})
            if summary_writer is not None:
                summary_writer.add_summary(stat, self.nstep_summary)
            losses.append(loss)
            self.nstep_summary += 1
            k += 1

        pred = sess.run(self.predict, feed_dict={self.input: test_data['X']})
        real = test_data['Y']
        score = sum(pred == real)*1.0/len(real)
        return score

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
