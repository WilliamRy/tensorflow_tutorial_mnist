import os
import tensorflow as tf

from model import Model
from feeder import feeder_mnist

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #仅使用第一块显卡
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7 #限定使用比例

#from keras.backend.tensorflow_backend import set_session
#config.gpu_options.allow_growth = True  #或者自适应分配显存
#set_session(tf.Session(config=config))
# #当使用keras时调用



def run():
    feeder = feeder_mnist() # load data


    sess = tf.Session(config = config)

    summary_writer = tf.summary.FileWriter('./log', sess.graph)
    model = Model(sess=sess, summary_writer=summary_writer)  # build model

    model.fit(X = feeder.train_data()['X'], y = feeder.train_data()['Y'])   # train model

    os.makedirs('./save', exist_ok = True)
    checkpoint_path = './save/model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_path, global_step = 0)    # save model


    tf.reset_default_graph()# reset default graph
    sess = tf.Session(config=config)

    model = Model(sess=sess)
    restore_vars = [v for v in tf.global_variables() if v.name[:5] == 'model']
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, checkpoint_path+'-0')     # restore model

    score = model.score(X = feeder.test_data()['X'], y = feeder.test_data(onehot=False)['Y'])
    # make predict on test data.
    print('>restore score: ', score)





if __name__ == '__main__':
    run()


