# -*- coding: utf-8 -*-
# @Time    : 2020/1/19 18:46
# @Author  : chenjunyu
# @FileName: config
# @Software: PyCharm


import tensorflow as tf


tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer('word_dim', 300, 'The dimension of the word embedding')
tf.flags.DEFINE_integer('rnn_hidden_size', 100, 'number of hidden units in the BiLSTM')
tf.flags.DEFINE_boolean('fine_tune_embedding', False, 'whether to fine tune the word embedding')
tf.flags.DEFINE_integer('sentence_max_length', 41, 'msrp or sick max sentence length')   # 最长的句子长度由样本确定
tf.flags.DEFINE_integer('nums_classes', 2, 'num of the labels')    # MSRP是二分类，SICK是五分类
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')      # 学习率
tf.flags.DEFINE_integer('rnn_layers', 3, 'the number of rnn layers')
tf.flags.DEFINE_integer('attention_hidden', 50, 'number of hidden units in attention layer')
tf.flags.DEFINE_integer('attention_num', 1, 'the number of different sentence semantics, r')
tf.flags.DEFINE_float('penalty_C', 0.001, 'Coefficient of penalty term')
tf.flags.DEFINE_integer('MLP_hidden_size', 50, 'the number of hidden units in MLP layer and output layer')
tf.flags.DEFINE_float('l2', 0.001, 'regularization parameter')
tf.flags.DEFINE_float("dropout_rate", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to be trained')
tf.flags.DEFINE_integer('batch_size', 25, 'size of mini batch')    # 每个batch大小是32

tf.flags.DEFINE_integer("display_step", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

config = tf.flags.FLAGS
# print(config.dropout_rate)
