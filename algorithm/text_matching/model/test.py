import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from algorithm import Graph
import tensorflow as tf
from algorithm.text_matching.utils.load_data import load_char_word_static_data2
from algorithm.text_matching.utils.evaluate import cal_precision_recall_F1

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 字向量和word2vec QA数据集
# p_index, h_index, p_vec, h_vec, y = load_char_word_static_data('../input/test.csv', data_size=1000)

# 电路数据集
p_index, h_index, p_vec, h_vec, y = load_char_word_static_data2('../input/test1.csv') #

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    if 1 == 1:
        saver.restore(sess, '../output/model/model_45.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})
        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_46.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_47.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_48.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_49.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)
    else:
        saver.restore(sess, '../output/model/model_25.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_26.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_27.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_28.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)

        saver.restore(sess, '../output/model/model_29.ckpt')
        loss, acc, predict_list = sess.run([model.loss, model.acc, model.predict_list],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: y,
                                                      model.keep_prob: 1})

        try:
            cal_precision_recall_F1(len(y), y, predict_list[0])
        except Exception as e:
            print(e)


