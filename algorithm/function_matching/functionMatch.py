from sympy import *

def match_fun(function1, function2):
    function1 = pre_processing(function1)
    function2 = pre_processing(function2)
    function1_simplified = sympify(function1).expand().trigsimp()
    function2_simplified = sympify(function2).expand().trigsimp()
    function1_simplified_list = str(function1_simplified).split(" ")
    function2_simplified_list = str(function2_simplified).split(" ")
    min_len = min(len(function1_simplified_list), len(function2_simplified_list))
    sameNum = 0
    for i in range(min_len):
        if (function1_simplified_list[i] == function2_simplified_list[i]
                and function1_simplified_list[i] != " "):
            sameNum += 1
    return 2 * sameNum / (len(function1_simplified_list) + len(function2_simplified_list))\
        ,str(function1_simplified),str(function2_simplified)

def pre_processing(function):
    res = ''
    for ch in function:
        if ch != '=':
            res += ch
        else:
            break
    return res