import pandas as pd
import os
from algorithm.text_matching.utils.data_utils import shuffle, pad_sequences
import jieba
import re
from gensim.models import Word2Vec
import numpy as np
from algorithm.text_matching.model import args


# 加载字典
def load_char_vocab():
    path = os.path.join(os.path.dirname(__file__), '../input/vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 加载词典
def load_word_vocab():
    path = os.path.join(os.path.dirname(__file__), '../output/word2vec/word_vocab.tsv')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word

def w2v(words, model):
    try:
        return model.wv[words]
    except:
        return np.zeros(args.word_embedding_len)

# 静态w2v
def w2v2(words, model):
    vec = []
    for word in words:
        if word in model.wv.vocab:
            vec.append(model[word])
        else:
            vec.append(np.zeros(args.word_embedding_len))
    return vec


# 字->index 每个字对应vocab.txt中的字的索引
def char_index(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()      # 创建vocab.txt 词的索引

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


# 词->index
def word_index(p_sentences, h_sentences):
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


def w2v_process(vec):
    if len(vec) > args.max_word_len:
        vec = vec[0:args.max_word_len]
    elif len(vec) < args.max_word_len:
        zero = np.zeros(args.word_embedding_len)
        length = args.max_word_len - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


# 加载char_index训练数据 把数据集转换为矩阵 一行代表一个句子
def load_char_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)      # 把数据打乱
    # [1,2,3,4,5] [4,1,5,2,0]
    p_c_index, h_c_index = char_index(p, h)     # 每句话对应的句子向量

    return p_c_index, h_c_index, label


# 加载char_index与静态词向量的训练数据
def load_char_word_static_data(file, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec.model')

    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    return p_c_index, h_c_index, p_w_vec, h_w_vec, label

# 加载char_index与静态词向量的训练数据 自己的数据集
def load_char_word_static_data2(file, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec2.model')

    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v2(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v2(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    return p_c_index, h_c_index, p_w_vec, h_w_vec, label

# 加载char_index与静态词向量的训练数据 自己的数据集
def load_char_word_static_list_predict(question, candidate):
    model = Word2Vec.load('./algorithm/text_matching/output/word2vec/word2vec2.model')
    # model = Word2Vec.load('../output/word2vec/word2vec2.model')

    p, h, label = [], [], []

    for i in range(len(candidate)):
        p.append(question)
        h.append(candidate[i])
        label.append(0)  #label同一设置为1或者0 因为不用评价 所以label用不到

    label = np.array(label)
    p_c_index, h_c_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v2(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v2(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    return p_c_index, h_c_index, p_w_vec, h_w_vec, label

# 加载char_index与动态词向量的训练数据
def load_char_word_dynamic_data(path, data_size=None):
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_char_index, h_char_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p)
    h_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h)

    p_word_index, h_word_index = word_index(p_seg, h_seg)

    return p_char_index, h_char_index, p_word_index, h_word_index, label


# 加载char_index、静态词向量、动态词向量的训练数据
def load_all_data(path, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec.model')
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p))
    h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h))

    p_w_index, h_w_index = word_index(p_seg, h_seg)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    # 判断是否有相同的词
    same_word = []
    for p_i, h_i in zip(p_w_index, h_w_index):
        dic = {}
        for i in p_i:
            if i == 0:
                break
            dic[i] = dic.get(i, 0) + 1
        for index, i in enumerate(h_i):
            if i == 0:
                same_word.append(0)
                break
            dic[i] = dic.get(i, 0) - 1
            if dic[i] == 0:
                same_word.append(1)
                break
            if index == len(h_i) - 1:
                same_word.append(0)

    return p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label

# 自己的数据集
def load_all_data2(path, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec2.model')
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p))
    h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h))

    p_w_index, h_w_index = word_index(p_seg, h_seg)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    # 判断是否有相同的词
    same_word = []
    for p_i, h_i in zip(p_w_index, h_w_index):
        dic = {}
        for i in p_i:
            if i == 0:
                break
            dic[i] = dic.get(i, 0) + 1
        for index, i in enumerate(h_i):
            if i == 0:
                same_word.append(0)
                break
            dic[i] = dic.get(i, 0) - 1
            if dic[i] == 0:
                same_word.append(1)
                break
            if index == len(h_i) - 1:
                same_word.append(0)

    return p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label



def padding(text, max_len):
    pad_text = []
    for i in text:
        embedding_i = i[1]  #句子长度的list，每一个元素都是一个词向量。 求求了多debug一下你就知道了 我服了 浪费了一天
        if len(i[1]) > max_len:
            embedding_i = embedding_i[0:max_len]
        elif len(i[1]) < max_len:
            pad = [np.zeros(768)]
            embedding_i += pad * (max_len - len(i[0]))
        pad_text.append(np.array(embedding_i))
    return pad_text


def 统计label(path, data_size=None):
    df = pd.read_csv(path)
    label = df['label'].values[0:data_size]
    count = 0
    for l in label:
        if l == 1:
            count += 1
    print("正类：" ,count)
    print("负类：" ,len(label) - count)

def 统计lenght(path, data_size=None):
    df = pd.read_csv(path)
    s1 = df['sentence1'].values[0:data_size]
    s2 = df['sentence2'].values[0:data_size]
    sum = 0
    for s in s1:
        sum += len(s)

    for s in s2:
        sum += len(s)

    print("平均长度为：" ,sum / (2 * len(s1)))

def 最大lenght(path, data_size=None):
    df = pd.read_csv(path)
    s1 = df['sentence1'].values[0:data_size]
    s2 = df['sentence2'].values[0:data_size]
    max = 0
    for s in s1:
        if max < len(s):
            max = len(s)

    for s in s2:
        if max < len(s):
            max = len(s)

    print("最大长度为：", max)

if __name__ == '__main__':
    # load_all_data('../input/train.csv', data_size=100)
    统计lenght('../input/train.csv')
    最大lenght('../input/train.csv')
