from flask import Flask, request, Response
import json
import time
from algorithm.function.functionMatch import match_fun
from log import Logger

app = Flask("algorithm", static_folder="/static", template_folder="templates")

logging = Logger('all.log',level='debug')


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)