# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 19:33
# @Author  : chenjunyu
# @FileName: model
# @Software: PyCharm


import tensorflow as tf


# 首先定义模型固定参数和其输入
class MSEM_WI(object):
    def __init__(self, config, word_embedding_matrix=None):
        self.word_embedding_dim = config.word_dim
        self.rnn_hidden_size = config.rnn_hidden_size
        self.fine_tune_embedding = config.fine_tune_embedding
        self.num_sentence_words = config.sentence_max_length
        self.learning_rate = config.learning_rate
        self.rnn_layers = config.rnn_layers
        self.attention_hidden = config.attention_hidden
        self.attention_num = config.attention_num   # r
        self.penalty_C = config.penalty_C
        self.MLP_hidden_size = config.MLP_hidden_size
        self.nums_classes = config.nums_classes
        self.l2 = config.l2
        self.batch_size_real_number = config.batch_size

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("input"):
            # 这里的句子占位符是根据查找词汇表vocab得到的下标序列，且最长为句子对的最大长度，超过长度的都设置为0
            # padding 的词都设置为0，embedding表格的第0行就是我们的padding向量？
            self.sentence_one_word = tf.placeholder(tf.int32, [None, self.num_sentence_words], name="sentence_one_word")
            self.sentence_two_word = tf.placeholder(tf.int32, [None, self.num_sentence_words], name="sentence_two_word")
            # y_true为对应的相似度标签，二分类是0或1；相似度分数转换为多分类问题，用专门的损失函数进行计算
            self.y_true = tf.placeholder(tf.float32, [None, self.nums_classes], name="true_labels")  # 0 or 1
            self.y = tf.placeholder(tf.int32, [None], name='real_similarity')   # 存储真正的相似度分数，未经过多分类标签化
            self.y_reshape = tf.reshape(self.y, [self.batch_size_real_number, 1])
            self.is_train = tf.placeholder(tf.bool, [], name="is_train")

        # tf.cond()类似于c语言中的if...else...
        self.dropout_rate = tf.cond(self.is_train, lambda: config.dropout_rate, lambda: 1.0)

        # 这里的word_embedding_matrix需要自己手动构建，因此需要写对应的函数
        self.word_mat = tf.get_variable("word_mat",
                                        initializer=tf.constant(word_embedding_matrix, dtype=tf.float32),
                                        trainable=self.fine_tune_embedding)

        # embedding之后的维度：[batch, num_sentence_words（句子最大的长度）, embedding_dim]
        with tf.variable_scope("embedding_layer"):
            # self.sentence_one_word应该是句子s1的索引下标，在句子输入之前就已经转换成下标的表示了
            # 需要在前面的文件中编写转换函数
            word_embedded_sentence_one = tf.nn.embedding_lookup(self.word_mat, self.sentence_one_word)
            word_embedded_sentence_two = tf.nn.embedding_lookup(self.word_mat, self.sentence_two_word)

        with tf.variable_scope("get_length_layer"):
            self.batch_size = tf.shape(self.sentence_one_word)[0]
            # tf.sign(x): -1 if x<0; 0 if x=0; 1 if x>0  （统计1的数量就是句子长度）
            # 0 是Padding，unknown需要怎么弄？
            # sentence_one_word是[batch, l]的大小
            self.sentence_one_wordlevel_mask = tf.sign(self.sentence_one_word, name="sentence_one_wordlevel_mask")
            self.sentence_two_wordlevel_mask = tf.sign(self.sentence_two_word, name="sentence_two_wordlevel_mask")
            # tf.reduce_sum会减少维度，用于降维，[batch]，一个一维数组
            self.sentence_one_len = tf.reduce_sum(self.sentence_one_wordlevel_mask, 1)
            self.sentence_two_len = tf.reduce_sum(self.sentence_two_wordlevel_mask, 1)

        with tf.variable_scope("sequence_encoder_layer"):
            one_fw_outputs, one_bw_outputs = self.my_bilstm(inputs=word_embedded_sentence_one,
                                                            dropout=self.dropout_rate,
                                                            n_layers=self.rnn_layers,
                                                            scope="bilstm",
                                                            sequence_length=self.sentence_one_len,
                                                            hidden_size=self.rnn_hidden_size,
                                                            initializer=initializer)
            two_fw_outputs, two_bw_outputs = self.my_bilstm(inputs=word_embedded_sentence_two,
                                                            dropout=self.dropout_rate,
                                                            n_layers=self.rnn_layers,
                                                            scope="bilstm",
                                                            sequence_length=self.sentence_two_len,
                                                            hidden_size=self.rnn_hidden_size,
                                                            initializer=initializer,
                                                            reuse=True)
            # 使用两者的结合作为我们的BiLSTM的输出
            # 当长度没有达到最大长度时
            one_lstm_output = tf.concat([one_fw_outputs, one_bw_outputs], axis=2)
            # [batch, l, 2*rnn_hidden_size]
            two_lstm_output = tf.concat([two_fw_outputs, two_bw_outputs], axis=2)

            # [batch * l, 2*rnn_hidden_size]]
            one_lstm_output_rsp = tf.reshape(one_lstm_output, [-1, 2 * self.rnn_hidden_size])
            two_lstm_output_rsp = tf.reshape(two_lstm_output, [-1, 2 * self.rnn_hidden_size])

        # 后续可以在上面的基础上实现self-attention
        # self-attention
        with tf.variable_scope("attention_layer"):
            W_s1 = tf.get_variable(name='W_s1', shape=[2 * self.rnn_hidden_size, self.attention_hidden], initializer=initializer)
            H_s1 = tf.nn.tanh(tf.matmul(one_lstm_output_rsp, W_s1))
            W_s2 = tf.get_variable(name='W_s2', shape=[self.attention_hidden, self.attention_num], initializer=initializer)
            H_s2 = tf.matmul(H_s1, W_s2)   # [batch*l,r]
            # 改变输出形状，得到[batch，num_steps, attention_num]
            # 转置，得到[batch, attention_num(r), num_stemps(l)]
            # 即batch个[r,l]矩阵
            H_s2_rsp = tf.transpose(tf.reshape(H_s2, [-1, self.num_sentence_words, self.attention_num]), [0, 2, 1])

            # 执行softmax进行归一化
            # 但是由于输入长度不同，有的已经padding的输出中会有很多输出为0，但是我们不能把0也算入到softmax中，因为会出错
            # H_s2_rsp,是[batch, r, l(最大长度)]
            # [batch, r, l]  已经经过归一化的，且0不会被算入到归一化中
            A1 = self.softmax(H_s2_rsp, self.sentence_one_len)
            self.heat_matrix1 = A1   # 句子1的热力图[batch, r, l]
            # [batch, r, l] * [batch, l, 2u]，得到[batch, r, 2u]
            M1 = tf.matmul(A1, one_lstm_output)    # r个单独的语义向量, [batch, r, 2u]
            A1_total = tf.reshape(tf.reduce_sum(A1, 1), [self.batch_size, -1, self.num_sentence_words])  # [batch, 1, l]
            A1_total = self.softmax(A1_total, self.sentence_one_len, is_total=True)   # 总体语义权重
            M1_total = tf.matmul(A1_total, one_lstm_output)   # 总体语义向量[batch, 1, 2u]

            H_s1 = tf.nn.tanh(tf.matmul(two_lstm_output_rsp, W_s1))
            H_s2 = tf.matmul(H_s1, W_s2)
            H_s2_rsp = tf.transpose(tf.reshape(H_s2, [-1, self.num_sentence_words, self.attention_num]), [0, 2, 1])
            A2 = self.softmax(H_s2_rsp, self.sentence_two_len)   # [batch, r, l]
            self.heat_matrix2 = A2
            M2 = tf.matmul(A2, two_lstm_output)   # [batch, r, 2u]，r个单独的语义向量
            A2_total = tf.reshape(tf.reduce_sum(A2, 1), [self.batch_size, -1, self.num_sentence_words])
            A2_total = self.softmax(A2_total, self.sentence_two_len, is_total=True)    # 总体语义权重 [batch, 1, l]
            M2_total = tf.matmul(A2_total, two_lstm_output)  # 总体语义向量[batch, 1, 2u]

        # 惩罚项，防止r相同
        with tf.variable_scope("penalization"):
            AA1_T = tf.matmul(A1, tf.transpose(A1, [0, 2, 1]))    # [batch, r, r]
            AA2_T = tf.matmul(A2, tf.transpose(A2, [0, 2, 1]))
            I = tf.eye(self.attention_num, batch_shape=[tf.shape(A1)[0]])   # [batch, r, r]的单位矩阵
            P1 = tf.square(tf.norm(AA1_T - I, axis=[-2, -1], ord="fro"))   # [batch, r, r]
            P2 = tf.square(tf.norm(AA2_T - I, axis=[-2, -1], ord="fro"))
            self.loss_P1 = tf.reduce_mean(self.penalty_C * P1)    # 惩罚系数 * 惩罚项
            self.loss_P2 = tf.reduce_mean(self.penalty_C * P2)

        with tf.variable_scope("importance_layer"):
            # M1_total_T = tf.transpose(M1_total, [0, 2, 1])   # [batch, 2u, 1]
            importance1 = self.compute_cosine_distance(M1, M1_total)   # [batch, r, 1]   # 一个batch中每个句子的每种语义 的重要性
            # importance1 = tf.matmul(M1, M1_total_T)
            self.importance1 = tf.nn.softmax(importance1, axis=1)  # [batch, r, 1]
            # M2_total_T = tf.transpose(M2_total, [0, 2, 1])
            importance2 = self.compute_cosine_distance(M2, M2_total)
            # importance2 = tf.matmul(M2, M2_total_T)
            self.importance2 = tf.nn.softmax(importance2, axis=1)   # [batch, r, 1]

        with tf.variable_scope("semantic_interaction_layer"):  # 余弦相似度
            # 2u * 2u的相似度矩阵
            # MM = tf.get_variable(name='MM', shape=[2 * self.rnn_hidden_size, 2 * self.rnn_hidden_size], initializer=initializer)
            # bias = tf.Variable(tf.constant(0.1, shape=[2 * self.rnn_hidden_size]), name='bias')
            # temp1 = tf.reshape(M1, [-1, 2 * self.rnn_hidden_size])   # [batch*r,2u]
            # temp2 = tf.matmul(temp1, MM)   # [batch*r, 2u]
            # temp3 = tf.reshape(temp2, [self.batch_size, -1, 2 * self.rnn_hidden_size])  # [batch, r, 2u]
            # interaction_matrix = tf.matmul(temp3, tf.transpose(M2, [0, 2, 1]))    # [batch, r, r]
            interaction_matrix = self.compute_cosine_distance(M1, M2)  # [batch, r, r]
            row_norm = tf.nn.softmax(interaction_matrix, axis=2)   # 行归一化 [batch, r, r]
            column_norm = tf.nn.softmax(interaction_matrix, axis=1)   # 列归一化 [batch, r, r]
            # 句子1在句子2中的对齐表示，[batch, r, 2u]
            alignment1_2 = tf.matmul(row_norm, M2)  # [batch,r, 2u]   r个对齐语义向量
            # 句子2在句子1中的对齐表示，[batch,r, 2u]
            alignment2_1 = tf.matmul(tf.transpose(column_norm, [0, 2, 1]), M1)  # [batch,r, 2u]

        with tf.variable_scope("Residual_decomposition_layer"):   # 残差分解
            # 首先分解原始语义，将其分解为平行和垂直于对齐语义的分量
            # 对于句子1的每个语义，其对齐表示是alignment1_2 [batch, r, 2u]
            # 分解句子1的原始语义，得到对应的平行和垂直分量
            # 或者分解句子1的对齐表示，原始语义不改变
            parallel1, vertical1 = self.residual_decompositon(M1, alignment1_2)   # 均是[batch, r, 2u]
            # 总的加权和平行向量
            weighted_parallel1 = tf.matmul(tf.transpose(self.importance1, [0, 2, 1]), parallel1)  # [batch, 1, 2u]
            # 总的加权和垂直向量
            weighted_vertical1 = tf.matmul(tf.transpose(self.importance1, [0, 2, 1]), vertical1)  # [batch, 1, 2u]

            # 分解句子2的原始语义，得到对应的平行和垂直分量
            parallel2, vertical2 = self.residual_decompositon(M2, alignment2_1)
            weighted_parallel2 = tf.matmul(tf.transpose(self.importance2, [0, 2, 1]), parallel2)
            weighted_vertical2 = tf.matmul(tf.transpose(self.importance2, [0, 2, 1]), vertical2)

            Mx = tf.multiply(M1_total, M2_total)   # 总体语义向量逐元素相乘作为额外特征向量, [batch, 1, 2u]
            M_ = M1_total - M2_total   # 总体语义向量逐元素相减作为额外特征向量, [batch, 1, 2u]
            # [batch, 1, 12u]
            features = tf.concat([Mx, M_, weighted_parallel1, weighted_parallel2, weighted_vertical1, weighted_vertical2], axis=2)

        with tf.variable_scope("Fully_connected_and_output_layer"):    # 全连接层
            # 12u * MLP_hidden_size
            W_f1 = tf.get_variable(name='W_f1', shape=[12 * self.rnn_hidden_size, self.MLP_hidden_size], initializer=initializer)
            W_f2 = tf.get_variable(name='W_f2', shape=[self.MLP_hidden_size, self.nums_classes], initializer=initializer)
            b1 = tf.Variable(tf.constant(0.1, shape=[self.MLP_hidden_size]), name="b1")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.nums_classes]), name="b2")
            self.features = tf.reshape(features, [-1, tf.shape(features)[1] * tf.shape(features)[2]])   # [batch, 12u]
            output1 = tf.nn.relu(tf.nn.xw_plus_b(self.features, W_f1, b1), name='output1')  # [batch, MLP_hidden_size]
            # [batch, nums_classes]，在行上进行归一化，得到每个类别的概率
            # output1 = tf.nn.dropout(output1, keep_prob=self.dropout_rate)
            final_output = tf.nn.xw_plus_b(output1, W_f2, b2, name='final_output')   # [batch, nums_classes]
            # final_output = tf.nn.dropout(final_output, keep_prob=self.dropout_rate)
            # final_output_plus_smooth = final_output + self.smooth   # 防止出现0的情况
            # self.final_output_reshape = tf.reshape(final_output, [self.batch_size_real_number])  # [batch]
            # tf.argmax会减少维度，在哪个axis进行arg，该维度就会减少
            # tf.argmax返回的是最大值所在的下标，在分类问题中，返回的其实就是对应的类别0 or 1
            # SICK的五分类任务中，返回的是0 1 2 3 4
            self.yhat = tf.argmax(final_output, axis=1, name="predictions")   # [batch]
            # yhat指的是最终的分类结果,其对应的是下标，因此真正的分类结果还需要下标+1
            # [batch]
            # self.real_yhat = tf.cast(self.yhat, dtype=tf.float32) + tf.ones(self.batch_size_real_number)   # +1 得到真正的分类结果
            # reduce_max也会减少维度，confidence是预测结果yhat对应的置信度，reduce_max返回最大值
            # [batch]  实际就是输入句子个数
            # confidence是置信度，也就是分成某一类的概率

            self.confidence = tf.reduce_max(tf.nn.softmax(final_output, axis=1), axis=1, name="confidence")

            # self.real_y = tf.argmax(self.y_true, 1)    # 将自己的多分类标签转换成真正的分类类别，但是是0 1 2 3 4
            # 依次检查self.yhat的值，将其与真正标签self.ral_y进行比较。这里真傻，预测时根本没有真实标签....
            '''
            self.y_predict = []
            for i in range(self.batch_size_real_number):   # 依次遍历
                if(self.yhat[i] == self.real_y[i]):     # 如果相等，则是i = y向下取整
                    y_temp = self.real_y[i] + 2 - self.confidence[i]
                elif(self.yhat[i] == self.real_y[i] + 1):   # i = y向下取整 + 1
                    y_temp = self.yhat[i] + self.confidence[i]
                else:
                    if(self.yhat[i] > self.real_y[i] + 1):
                        y_temp = self.yhat[i]
                    if(self.yhat[i] < self.real_y[i]):
                        y_temp = self.yhat[i] + 1 + self.confidence[i]
            '''

        with tf.variable_scope("loss"):   # 这里的Loss仍然使用交叉熵损失函数（即KL散度）
            # 使用均方差作为损失函数
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=self.y_true, name='Softmax')
            # 一个批次的均方误差   # [batch, 1]
            # losses = tf.reduce_mean(tf.square(tf.reshape(final_output - self.y_reshape, [self.batch_size_real_number])))
            # losses = tf.reduce_mean(tf.abs(tf.reshape(final_output - self.y_reshape, [self.batch_size_real_number])))
            # losses = self.kl_divergence(self.y_true, final_output)
            L2 = self.l2 * tf.add_n([tf.nn.l2_loss(param) for param in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses, name='losses') + L2
            # self.loss = tf.reduce_mean(losses, name='losses') + L2 + self.loss_P1 + self.loss_P2

        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.yhat, tf.argmax(self.y_true, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
            # 统计tf.cast(correct_predictions, 'float')中1的个数，就得到了本batch中有多少分类正确的样例
            correct_list = tf.cast(correct_predictions, 'float')
            self.correct_num = tf.reduce_sum(correct_list)   # 每个batch中正确分类的数目，测试时需要

        # with tf.variable_scope("mean_squared_error"):
            '''
            self.mse = tf.reduce_mean(tf.square(self.real_yhat - self.y))   # 一个批次的均方误差
            self.se = tf.reduce_sum(tf.square(self.real_yhat - self.y))   # 对于测试集，我们只需要知道一个批次的总的平方误差，最后再求均值
            '''
            '''
            self.mse = tf.reduce_mean(tf.square(self.real_yhat - self.y))
            # self.se = tf.reduce_sum(tf.square(tf.reshape(final_output - self.y_reshape, [self.batch_size_real_number])))
            self.se = tf.reduce_sum(tf.square(self.real_yhat - self.y))
            '''

        # SICK不需要f1指标
        with tf.variable_scope("f1"):
            self.accuracy_direct, _, _, self.F1 = self.tf_confusion_metrics(final_output, self.y_true)   # F1跟acc一样，是一个float标量

        # 在测试集上也只能分批读入 我们考虑每次在测试集上进行统计，但是测试集的分批读入需要统计每一批次的正确大小

    '''
    def kl_divergence(self, x, y):   # 传入[batch, 5]，x是真实分布，y是预测分布。x, y均是tensor
        kl = []
        # x_not_tensor = []
        # y_not_tensor = []
        batch_size = self.batch_size_real_number  # batch_size大小
        for i in range(batch_size):
            l1 = x[i][:]
            l2 = y[i][:]
            X = tf.distributions.Categorical(probs=l1)
            Y = tf.distributions.Categorical(probs=l2)
            kl.append(tf.distributions.kl_divergence(X, Y))
        kl_sum = tf.constant(0, dtype=tf.float32)
        for k in kl:
            kl_sum = kl_sum + k    # 一个batch总的KL散度
        return kl_sum / batch_size
    '''

    def my_bilstm(self, inputs, dropout, n_layers, scope, sequence_length, hidden_size, initializer, reuse=None):
        with tf.variable_scope("fw" + scope, reuse=reuse):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=tf.orthogonal_initializer())
                # fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=initializer)
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=1)   # LSTM不设置dropout
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.variable_scope("bw" + scope, reuse=reuse):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=tf.orthogonal_initializer())
                # bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0, initializer=initializer)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=1)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.variable_scope(scope, reuse=reuse):
            # tf.nn.bidirectional_dynamic_rnn返回的是outputs，last_states，其中outputs是[batch,num_steps,embedding_size]
            (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell_m,
                                                                          cell_bw=lstm_bw_cell_m,
                                                                          inputs=inputs,
                                                                          sequence_length=sequence_length,
                                                                          dtype=tf.float32)
        return fw_outputs, bw_outputs

    # 传入的参数中inputs是[batch, r, l（最大长度）], seq_length是[batch]（即长度）
    def softmax(self, inputs, seq_length, is_total=False):
        softmaxed_inputs = []
        batch_size = self.batch_size_real_number   # batch_size大小
        r = self.attention_num    # 语义个数
        if(is_total == True):
            r = 1
        max_len = self.num_sentence_words     # 最大长度
        for i in range(batch_size):
            tensor = []
            for j in range(r):
                rj = tf.nn.softmax(inputs[i][j][:seq_length[i]], axis=0, name='softmax')
                # print(rj)
                padding = tf.zeros((max_len - seq_length[i]), dtype=tf.float32)
                new_row = tf.concat([rj, padding], axis=0)
                # tf.assign(inputs[i], tf.scatter_update(inputs[i], j, new_row))
                # print(inputs[i])
                tensor.append(new_row)
            tensor = tf.stack(tensor, axis=0)
            softmaxed_inputs.append(tensor)
        return softmaxed_inputs

    # 余弦相似度计算
    def compute_cosine_distance(self, x, y):   # 传入的是[batch, r, 2u]
        with tf.name_scope('cosine_distance'):
            # cosine=x*y/(|x||y|)
            x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2))  # [batch, r]
            x_norm = tf.reshape(x_norm, [tf.shape(x)[0], tf.shape(x)[1], -1])   # [batch, r, 1]
            y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=2))  # [batch, r]
            y_norm = tf.reshape(y_norm, [tf.shape(y)[0], tf.shape(y)[1], -1])  # [batch, r, 1]
            norm_matrix = tf.matmul(x_norm, tf.transpose(y_norm, [0, 2, 1]))  # [batch, r, r]
            # 求x和y的内积
            x_y = tf.matmul(x, tf.transpose(y, [0, 2, 1]))   # [batch, r, r]
            # 内积除以模的乘积
            cosine = tf.divide(x_y, norm_matrix)   # [batch, r, r]
            return cosine

    # 残差分解
    def residual_decompositon(self, original_semantic, alignment_semantic):   # 输入都是[batch, r, 2u]的语义向量和语义对齐向量
        # 需要返回的是分解后的平行向量和垂直向量
        # 返回的大小是[batch, r, 2u]的平行和垂直向量
        '''
        parallel = []
        vertical = []
        batch_size = tf.shape(original_semantic)[0]
        r = tf.shape(original_semantic)[1]
        '''
        numerator = tf.matmul(original_semantic, tf.transpose(alignment_semantic, [0, 2, 1]))  # [batch, r, r]
        denominator = tf.matmul(alignment_semantic, tf.transpose(alignment_semantic, [0, 2, 1]))   # [batch, r, r]
        factor = tf.divide(numerator, denominator)   # [batch, r, r]，我们需要将非对角全部变为0
        factor_diag = tf.matrix_diag_part(factor)
        factor_diag_matrix = tf.matrix_diag(factor_diag)   # [batch, r, r]，仅保留对角元素，其余均是0
        parallel = tf.matmul(factor_diag_matrix, alignment_semantic)   # [batch, r, 2u]，original_semantic的平行分量，
        # 每一行都是一个平行分量 [batch, r, 2u]
        vertical = original_semantic - parallel   # [batch, r, 2u]，垂直分量
        return parallel, vertical

    # 混淆矩阵
    def tf_confusion_metrics(self, predict, real):    # 传入的是[batch, nums_classes]
        predictions = tf.argmax(predict, 1)
        actuals = tf.argmax(real, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )
        tn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )
        fp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                ),
                "float"
            )
        )
        fn = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                ),
                "float"
            )
        )
        tpr = tp / (tp + fn)     # recall
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        recall = tpr
        precision = tp / (tp + fp)
        f1_score = (2 * (precision * recall)) / (precision + recall)
        return accuracy, precision, recall, f1_score

