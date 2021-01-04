# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 13:11
# @Author  : chenjunyu
# @FileName: utility
# @Software: PyCharm


import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import tensorflow as tf


def clean_str(string):
    string = string.strip().lower()    # 需要全部转成小写
    tokenized_string = word_tokenize(string)
    return tokenized_string


def build_glove_dic(glove_path):
    # 从文件中读取 pre-trained 的 glove 文件，对应每个词的词向量
    # 其中400000代表共有四十万个词，每个词300维，中间为一个空格或者tab键
    # glove_path = 'glove.6B.300d.txt'
    vocab = []  # 词汇表
    vectors = []  # 对应的向量
    lines = open(glove_path, 'r', encoding='utf-8').readlines()
    for line in lines:
        line = line.strip().split(' ')
        str = line[0]   # 单词
        vocab.append(str)
        vector = []   # 单词对应的向量
        for word in line[1:]:
            vector.append(float(word))
        vectors.append(vector)
    word_embedding = np.array(vectors)    # 二维矩阵[400000, 300]
    # for w in vocab:
    #    print(w)
    # print(word_embedding)
    print(np.shape(word_embedding))
    word_mean = np.mean(word_embedding, axis=0)   # 平均值？用词向量的平均值代表未见词？其实也行！
    # 这里的word_id是从2开始的，id为0代表是padding，id为1代表词不在词汇表中。因此embedding_word矩阵的真正的词向量也应该从下标1开始
    sr_word2id = pd.Series(range(2, len(vocab) + 2), index=vocab)  # 使用pandas的Series构建词典
    sr_word2id['<padding>'] = 0   # padding填充的词的embedding的下标是0
    sr_word2id['<unk>'] = 1   # 不在词汇表中的词，其对应的词汇表下标是1（oov）
    print(sr_word2id)
    zeros = np.zeros([300])
    word_embedding = np.vstack([zeros, word_mean, word_embedding])
    print(word_embedding)
    return sr_word2id, word_embedding   # 返回的是词典（词汇表的词——索引）


global sr_word2id, word_embedding
sr_word2id, word_embedding = build_glove_dic('./glove.6B.300d.txt')


def get_id(word):
    if word in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id['<unk>']


def seq2id(seq):
    seq = clean_str(seq)   # 先清洗数据
    # map返回的是迭代器
    seq_id = map(get_id, seq)   # 对seq中的每个词都得到对应的id
    # print(seq_id)
    return list(seq_id)


# 填充句子，统一输入格式
def padding_sentence(s1, s2):
    # 得到句子s1,s2以后，很直观地想法就是先找出数据集中的最大句子长度，
    # 然后用<padding>（即0）对句子进行填充
    # 输入的s1和s2均是经过seq2id处理之后的下标索引，首先先根据其找出最大句子长度，然后进行padding
    s1_length_max = max([len(s) for s in s1])
    s2_length_max = max([len(s) for s in s2])

    # sentence_length = max(s1_length_max, s2_length_max)  # 得到最大的句子长度
    # print('max_sentence_length:', sentence_length)

    sentence_length = 41    # MSRP最长句子41，SICK最长句子36
    sentence_num = s1.shape[0]  # 得到总共多少个句子
    s1_padding = np.zeros([sentence_num, sentence_length], dtype=int)
    s2_padding = np.zeros([sentence_num, sentence_length], dtype=int)

    for i, s in enumerate(s1):   # i表示第几个句子，s表示该句子原先的索引下标向量，
        s1_padding[i][:len(s)] = s
    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    print("句子填充完毕")
    return s1_padding, s2_padding  # 返回最终的s1向量索引矩阵和s2向量索引矩阵


def read_data_sets():    # 这个函数不太可靠
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度，在MSRP数据集中代表的是二分类0 or 1
    # sample_num代表数据总共有多少行
    #
    '''
    SICK_DIR = "SICK.txt"
    df_sick = pd.read_csv(SICK_DIR, sep="\t", usecols=[1, 2, 4], names=['s1', 's2', 'score'],
                          dtype={'s1': object, 's2': object, 'score': object})   # 读取1,2，4行
    df_sick = df_sick.drop([0])   # 将第0行丢弃（因为第0行是说明）
    s1 = df_sick.s1.values
    s2 = df_sick.s2.values
    score = df_sick.score.values  # 将结构数据转化为ndarray且数据类型是float类型
    score = score.tolist()
    score = list(map(float, score))
    score = np.array(score)
    '''
    MSRP_DIR = './msr_paraphrase_train_augmentation.txt'
    df_msrp = pd.read_csv(MSRP_DIR, sep='\t', usecols=[0, 3, 4], names=['score', 's1', 's2'],
                          dtype={'score': object, 's1': object, 's2': object}, encoding='utf-8-sig')
    # df_msrp = df_msrp.drop([0])   # 丢弃0行
    s1 = df_msrp.s1.values
    s2 = df_msrp.s2.values
    score = df_msrp.score.values   # 将结构数据转化为ndarray且数据类型是float类型
    print(score)
    score = score.tolist()
    score = list(map(float, score))
    score = np.array(score)

    # s1 = np.asarray(list(map(seq2id, s1)))
    # s2 = np.asarray(list(map(seq2id, s2)))
    # 返回的s1,s2是句子矩阵，但是是向量，且向量内容是索引

    # 填充句子
    # s1, s2 = padding_sentence(s1, s2)
    return s1, s2, score


# 实际上我们可以读两遍，很自然的就得到了
def read_msrp(path, is_Train=True):
    s1 = []
    s2 = []
    score = []
    MSRP_DIR = path
    lines = open(MSRP_DIR, 'r', encoding='utf-8-sig').readlines()
    for line in lines:
        line = line.strip().split('\t')
        score.append(int(line[0]))
        s1.append(line[3])
        s2.append(line[4])
        # 读取句子交换（augmentation）
        if(is_Train == True):   # 测试时不需要读两遍
            score.append(int(line[0]))
            s1.append(line[4])
            s2.append(line[3])
    score = np.array(score)
    print(score)
    num_labels = score.shape[0]
    print(num_labels)    # 句子个数
    nums_classes = 2
    index_offset = np.arange(num_labels) * nums_classes
    labels_one_hot = np.zeros((num_labels, nums_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + score.ravel()] = 1
    print(labels_one_hot)
    s1 = np.array(s1)
    s2 = np.array(s2)
    s1 = np.asarray(list(map(seq2id, s1)))
    s2 = np.asarray(list(map(seq2id, s2)))
    print(len(s1))
    print(s1)
    print(s2)
    s1, s2 = padding_sentence(s1, s2)
    return s1, s2, score, labels_one_hot


def read_sick(path, is_Train=True):    # 不用读两遍   训练时直接读取train和trial作为整体的训练集
    s1 = []
    s2 = []
    score = []
    SICK_DIR = path
    lines = open(SICK_DIR, 'r', encoding='utf-8').readlines()   # 读取训练数据
    for line in lines:
        line = line.strip().split('\t')
        score.append(float(line[3]))
        s1.append(line[1])
        s2.append(line[2])
        '''
        if (is_Train == True):
            score.append(float(line[3]))
            s1.append(line[2])
            s2.append(line[1])
        '''
    if(is_Train == True):
        lines = open('./SICK_trial.txt', 'r', encoding='utf-8').readlines()   # 将验证数据仍然也作为训练数据
        for line in lines:
            line = line.strip().split('\t')
            score.append(float(line[3]))
            s1.append(line[1])
            s2.append(line[2])
            '''
            score.append(float(line[3]))
            s1.append(line[2])
            s2.append(line[1])
            '''
    score = np.array(score)     # 这里的score只是一个列表，记录的是相似度分数，需要转换成分类类别1 ,2 ,3 ,4, 5
    print(score)
    lables = []   # 存储真正的多分类概率标签
    for s in score:
        lable = np.zeros(5)
        index = int(s) - 1
        lable[index] = int(s) + 1 - s    # 分类类别是向下取整
        if(index != 4):
            lable[index + 1] = s - int(s)    # 分类类别是向上取整
        lables.append(lable)
    lables = np.array(lables)
    print(lables)
    print(len(lables))
    s1 = np.array(s1)
    s2 = np.array(s2)
    print(len(s1))
    print(s1)
    print(s2)
    s1 = np.asarray(list(map(seq2id, s1)))
    s2 = np.asarray(list(map(seq2id, s2)))
    print(s1)
    print(s2)
    s1, s2 = padding_sentence(s1, s2)
    print(s1)
    print(s2)
    return s1, s2, score, lables


# 针对原始数据s1,s2,y进行batch划分，并以此产生对应的batch，依靠关键字yield
def batch_iter(s1, s2, y, lables, batch_size, num_epochs, shuffle=True):   # batch_size指的是每次训练的样本数
    data_size = len(s1)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if(shuffle):   # 如果需要打乱数据，在每轮迭代之前都预先打乱数据
            shuffle_indices = np.random.permutation(np.arange(data_size))   # 打乱下标0--data_size-1
            shuffled_s1 = s1[shuffle_indices]   # 打乱数据
            shuffled_s2 = s2[shuffle_indices]
            shuffled_y = y[shuffle_indices]
            shuffled_lables = lables[shuffle_indices]
        else:
            shuffled_s1 = s1
            shuffled_s2 = s2
            shuffled_y = y
            shuffled_lables = lables
        # 固定batch大小，没有达到batch的需要进行补充，一般都是最后一个没有达到batch大小、或者干脆不要最后一个batch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size    # 训练数据开始时的下标
            end_index = min((batch_num + 1) * batch_size, data_size)   # 训练数据结束时的下标
            if(end_index - start_index != batch_size):   # 表明最后一个没有batch大小
                continue   # 直接跳过最后一个batch
            yield shuffled_s1[start_index:end_index], shuffled_s2[start_index:end_index], shuffled_y[start_index:end_index], shuffled_lables[start_index:end_index],
            # 得到的都是batch大小的


# 传入的参数中inputs是[batch, r, l（最大长度）], seq_length是[batch]（即长度）
def softmax(inputs, seq_length):
    softmaxed_inputs = []
    batch_size = 10
    r = 1
    l = 41
    for i in range(batch_size):
        tensor = []
        for j in range(r):
            rj = tf.nn.softmax(inputs[i][j][:seq_length[i]], axis=0, name='softmax')
            # print(rj)
            padding = tf.zeros((l - seq_length[i]), dtype=tf.float32)   # 这里的最大长度为41
            new_row = tf.concat([rj, padding], axis=0)    # 一维
            # inputs[i][j].assign(new_row)
            # tf.scatter_update(inputs[i], j, new_row)
            # new_row = tf.reshape(new_row, [-1, tf.shape(new_row)[0]])   # 二维
            # print(new_row)
            tensor.append(new_row)
        tensor = tf.stack(tensor, axis=0)
        softmaxed_inputs.append(tensor)
    softmaxed_inputs = tf.stack(softmaxed_inputs, axis=0)
    # print(softmaxed_inputs)
    return softmaxed_inputs


if __name__ == '__main__':
    print('begining....')
    '''
    sr_word2id, word_embedding = build_glove_dic('./glove.6B.300d.txt')
    print(sr_word2id)
    print(word_embedding)
    seq_id = seq2id('Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier.')
    print(seq_id)
    '''
    '''
    sr_word2id, word_embedding = build_glove_dic('./glove.6B.300d.txt')
    s1, s2, score = read_msrp()
    print(score)
    print(np.shape(s1))
    print(s1)
    print(len(s1))
    print(s2)
    print(s1[:10, :])

    # 需要嵌入的shape是[batch, max_length]，None就是输入的batch大小
    sentence_one_word = tf.placeholder(tf.int32, [None, 41], name="sentence_one_word")
    word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_embedding, dtype=tf.float32))
    # [batch, max_length, embedding_size]
    word_embedded_sentence_one = tf.nn.embedding_lookup(word_mat, sentence_one_word)

    sentence_one_wordlevel_mask = tf.sign(sentence_one_word)   # mask
    sentence_one_len = tf.reduce_sum(sentence_one_wordlevel_mask, 1)   # 输入batch的句子长度
    print(sentence_one_len)

    s = tf.reduce_sum(word_embedded_sentence_one, axis=2)   # [batch, max_length(l)]
    s = tf.transpose(tf.reshape(s, [tf.shape(s)[0], tf.shape(s)[1], -1]), [0, 2, 1])   # [batch, 1, max_length(l)]
    # softmax函数的输入

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    word_embedded_sentence_one = sess.run(word_embedded_sentence_one, feed_dict={sentence_one_word: s1[:10, :]})
    print(word_embedded_sentence_one)

    output = softmax(s, sentence_one_len)   # [batch, 1, max_length]
    output = sess.run(output, feed_dict={sentence_one_word: s1[:10, :]})
    print(output)
    '''
    read_sick('./SICK_train.txt')
    read_sick('./SICK_test_annotated.txt', is_Train=False)

    # 验证成功，查表正确
    # 验证自己写的softmax函数：验证成功

