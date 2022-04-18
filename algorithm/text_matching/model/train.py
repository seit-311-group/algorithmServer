import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from algorithm.text_matching.model.graph import Graph
import tensorflow as tf
from algorithm.text_matching.utils.load_data import load_char_word_static_data2
from algorithm.text_matching.model import args
from algorithm.text_matching.utils.evaluate import cal_precision_recall_F1

# 字向量和word2vec QA_Corpus
# p_index, h_index, p_vec, h_vec, y = load_char_word_static_data('../input/train.csv', data_size=None)
# p_index_dev, h_index_dev, p_vec_dev, h_vec_dev, y_dev = load_char_word_static_data('../input/dev.csv')
# 电路数据集 字向量+w2v
p_index, h_index, p_vec, h_vec, y = load_char_word_static_data2('../input/train1.csv', data_size=None)
p_index_dev, h_index_dev, p_vec_dev, h_vec_dev, y_dev = load_char_word_static_data2('../input/dev1.csv',
                                                                                       data_size=100)


p_index_holder = tf.placeholder(name='p_index', shape=(None, args.seq_length), dtype=tf.int32)
h_index_holder = tf.placeholder(name='h_index', shape=(None, args.seq_length), dtype=tf.int32)
p_vec_holder = tf.placeholder(name='p_vec', shape=(None, args.seq_length, args.char_embedding_size),
                              dtype=tf.float32)
h_vec_holder = tf.placeholder(name='h_vec', shape=(None, args.seq_length, args.char_embedding_size),
                              dtype=tf.float32)
y_holder = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)


dataset = tf.data.Dataset.from_tensor_slices((p_index_holder, h_index_holder, p_vec_holder, h_vec_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1

# epochs
epochs_loss = []

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_index_holder: p_index, h_index_holder: h_index, p_vec_holder: p_vec,
                                              h_vec_holder: h_vec,  y_holder: y})
    steps = int(len(y) / args.batch_size)
    for epoch in range(args.epochs):
        for step in range(steps):
            p_index_batch, h_index_batch, p_vec_batch, h_vec_batch,  y_batch = sess.run(next_element)
            _, loss, acc, predict_list = sess.run([model.train_op, model.loss, model.acc, model.predict_list],
                                    feed_dict={model.p: p_index_batch,
                                               model.h: h_index_batch,
                                               model.p_vec: p_vec_batch,
                                               model.h_vec: h_vec_batch,

                                               model.y: y_batch,
                                               model.keep_prob: args.keep_prob})
            if step % 50 == 0:
                print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)
                if step == 0:
                    epochs_loss.append(loss)
                try:
                    cal_precision_recall_F1(len(y_batch), y_batch, predict_list[0])
                except Exception as e:
                    print(e)
        print("eval：====================================================================")
        loss_eval, acc_eval, predict_list_eval = sess.run([model.loss, model.acc, model.predict_list],
                                      feed_dict={model.p: p_index_dev,
                                                 model.h: h_index_dev,
                                                 model.p_vec: p_vec_dev,
                                                 model.h_vec: h_vec_dev,
                                                 model.y: y_dev,
                                                 model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
        try:
            cal_precision_recall_F1(len(y_dev), y_dev, predict_list_eval[0])
        except Exception as e:
            print(e)
        print('\n')
        saver.save(sess, f'../output/model/model_{epoch}.ckpt')
