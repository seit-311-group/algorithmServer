import tensorflow as tf
from algorithm.text_matching.model import args


class Graph:

    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.p_vec = tf.placeholder(name='p_word', shape=(None, args.seq_length, args.char_embedding_size),
                                    dtype=tf.float32)
        self.h_vec = tf.placeholder(name='h_word', shape=(None, args.seq_length, args.char_embedding_size),
                                    dtype=tf.float32)
        self.p_bert_vec = tf.placeholder(name='p_bert_word', shape=(None, args.seq_length, args.bert_emdedding_size),
                                    dtype=tf.float32)
        self.h_bert_vec = tf.placeholder(name='h_bert_word', shape=(None, args.seq_length, args.bert_emdedding_size),
                                    dtype=tf.float32)
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')
        self.predict_list = []
        self.similarity = []
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def bilstm(self, x, hidden_size):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def bigru(self, x, hidden_size):
        fw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        p_embedding = tf.concat((p_embedding, self.p_vec), axis=-1)
        h_embedding = tf.concat((h_embedding, self.h_vec), axis=-1)

        p = self.dropout(p_embedding)
        h = self.dropout(h_embedding)

        # attention层
        e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
        a_attention = tf.nn.softmax(e)
        b_attention = tf.transpose(tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1])), perm=[0, 2, 1])

        a = tf.matmul(a_attention, h)
        b = tf.matmul(b_attention, p)

        m_a = tf.concat((a, p, a - p, tf.multiply(a, p)), axis=2)
        m_b = tf.concat((b, h, b - h, tf.multiply(b, h)), axis=2)

        # 特征提取层
        with tf.variable_scope("lstm_a", reuse=tf.AUTO_REUSE):
            (a_f, a_b), _ = self.bilstm(m_a, args.context_hidden_size)
        with tf.variable_scope("lstm_b", reuse=tf.AUTO_REUSE):
            (b_f, b_b), _ = self.bilstm(m_b, args.context_hidden_size)


        a = tf.concat((a_f, a_b), axis=2)
        b = tf.concat((b_f, b_b), axis=2)

        with tf.variable_scope("bigru_a", reuse=tf.AUTO_REUSE):
            (a_f, a_b), _ = self.bigru(a, args.context_hidden_size)
        with tf.variable_scope("bigru_b", reuse=tf.AUTO_REUSE):
            (b_f, b_b), _ = self.bigru(b, args.context_hidden_size)

        a = tf.concat((a_f, a_b), axis=2)
        b = tf.concat((b_f, b_b), axis=2)

        a = self.dropout(a)
        b = self.dropout(b)

        a_avg = tf.reduce_mean(a, axis=2)
        b_avg = tf.reduce_mean(b, axis=2)

        a_max = tf.reduce_max(a, axis=2)
        b_max = tf.reduce_max(b, axis=2)

        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        v = tf.layers.dense(v, 512, activation='tanh')
        v = self.dropout(v)
        logits = tf.layers.dense(v, 2, activation='tanh')
        self.prob = tf.nn.softmax(logits)
        self.similarity.append(self.prob)
        self.prediction = tf.argmax(logits, axis=1)
        self.predict_list.append(self.prediction)
        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, label_smoothing=0.001)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
