from flask import Flask, request, Response
import json
import time
from algorithm.function_matching.functionMatch import match_fun
from algorithm.text_matching.model.graph import Graph
import tensorflow as tf
from log import Logger
from algorithm.text_matching.utils.load_data import load_char_word_static_list_predict
import numpy as np

app = Flask("algorithm", static_folder="/static", template_folder="templates")

logging = Logger('all.log', level='debug')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

model = Graph()
saver = tf.train.Saver()
sess = tf.compat.v1.Session(config=config)

sess.run(tf.global_variables_initializer())
saver.restore(sess, './algorithm/text_matching/output/model/model_0.ckpt')

@app.route('/functionMatch', methods=['POST'])
def match_function():
    t1 = time.time()
    logging.logger.info('调用方程匹配功能')
    function1 = request.json['function1']
    function2 = request.json['function2']
    try:
        res = match_fun(function1=function1, function2=function2)
        res = {'similarity': res[0], 'function1': res[1], 'function2': res[2]}
    except Exception as e:
        logging.logger.error(e)
        return Response(json.dumps("错误"), mimetype='application/json')
    t2 = time.time()
    print("用时：" + str(round(t2 - t1, 3)))
    return Response(json.dumps(res), mimetype='application/json')

@app.route('/textMatch', methods=['POST'])
def match_text():
    logging.logger.info('调用文本匹配功能')
    question = request.json['question']     # 字符串
    candidate = request.json['candidate']   # list
    p_index, h_index, p_vec, h_vec, label = load_char_word_static_list_predict(question, candidate)
    similarity = sess.run(model.similarity,
                                           feed_dict={model.p: p_index,
                                                      model.h: h_index,
                                                      model.p_vec: p_vec,
                                                      model.h_vec: h_vec,
                                                      model.y: label,
                                                      model.keep_prob: 1})
    res = {}
    for i in range(len(candidate)):
        res[candidate[i]] = np.float(similarity[0][i][1])

    return Response(json.dumps(res), mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)