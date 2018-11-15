import os
import tensorflow as tf


from model import CNN
from official_model import Official_CNN
from feeder import feeder_mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #仅使用第一块显卡
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 #限定使用比例


#from keras.backend.tensorflow_backend import set_session
#config.gpu_options.allow_growth = True  #或者自适应分配显存
#set_session(tf.Session(config=config))
# #当使用keras时调用



def random_try(model):
    sess = tf.Session(config=config)
    if os.path.exists('./log'):
        os.system('rm -rf log')
    summary_writer = tf.summary.FileWriter('./log', sess.graph)

    feeder = feeder_mnist()
    for i in range(20):
        train_data = feeder.train_data(n=200)
        test_data = feeder.test_data(onehot=False)
        scores = []
        for j in range(5):
            score = model.train_and_test(train_data, test_data, sess=sess,
                                         batch_size=100, nepoch=2000,
                                         summary_writer=summary_writer)
            scores.append(score)
        print(scores)

def balance_vs_random(model):
    sess = tf.Session(config=config)
    if os.path.exists('./log'):
        os.system('rm -rf log')
    summary_writer = tf.summary.FileWriter('./log', sess.graph)

    feeder = feeder_mnist()
    print('This is result of balanced sample training:')
    for i in range(10):
        train_data = feeder.train_data(n=1000, index=feeder.class_balanced_index(), seed = i)
        test_data = feeder.test_data(onehot=False)
        scores = []
        for j in range(5):
            score = model.train_and_test(train_data, test_data, sess=sess,
                                         batch_size=100, nepoch=500,
                                         summary_writer=summary_writer)
            scores.append(score)
        print(scores)

    print('This is result of random sample training:')
    for i in range(10):
        train_data = feeder.train_data(n=1000, seed=i)
        test_data = feeder.test_data(onehot=False)
        scores = []
        for j in range(5):
            score = model.train_and_test(train_data, test_data, sess=sess,
                                         batch_size=100, nepoch=500,
                                         summary_writer=summary_writer)
            scores.append(score)
        print(scores)

def full_train_and_pick(model):
    sess = tf.Session(config=config)
    if os.path.exists('./log'):
        os.system('rm -rf log')
    summary_writer = tf.summary.FileWriter('./log', sess.graph)
    feeder = feeder_mnist()
    train_data = feeder.train_data(n=None)
    test_data = feeder.test_data(onehot=False)
    score, losses = model.train_and_test(train_data, test_data, sess=sess,
                                 batch_size=100, nepoch=10000,
                                 summary_writer=summary_writer, return_loss= True)
    return score, losses





if __name__ == '__main__':
    # model = CNN()
    model = Official_CNN()
    # random_try(model)
    # balance_vs_random(model)
    full_train_and_pick(model)


