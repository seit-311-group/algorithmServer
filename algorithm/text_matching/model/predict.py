import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from algorithm.text_matching.model.graph import Graph
import tensorflow as tf
from algorithm.text_matching.utils.load_data import load_char_word_static_data2, load_char_word_static_list_predict
import time
from tensorflow.keras.models import load_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == '__main__':
    model = Graph()
    saver = tf.train.Saver()

    p_index, h_index, p_vec, h_vec, label = load_char_word_static_data2('../input/ourPredict.csv')

    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '../output/model/model_49.ckpt')
        t1 = time.time()
        loss, acc, predict_list, similarity = sess.run([model.loss, model.acc, model.predict_list, model.similarity],
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: label,
                                                      model.keep_prob: 1})
        t2 = time.time()
        print(t2-t1)
        print("用时：" + str(int(round((t2 - t1) * 1000))) + "毫秒")
        print(similarity[0])

