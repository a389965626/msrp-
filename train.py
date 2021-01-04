# -*- coding: utf-8 -*-
# @Time    : 2020/1/20 19:10
# @Author  : chenjunyu
# @FileName: train
# @Software: PyCharm


from utilities import *
from MSEM_WI_model import *
import os
import time
import datetime
from configs import config
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import datetime


# 预处理
def preprocess():
    # Load data
    print("Loading data...")

    s1, s2, y, lables = read_msrp('./msr_paraphrase_train.txt')    # 读取数据
    s1_test, s2_test, y_test, lables_test= read_msrp('./msr_paraphrase_test.txt', is_Train=False)

    '''
    s1, s2, y, lables= read_sick('./SICK_train.txt')  # 读取数据
    s1_test, s2_test, y_test, lables_test = read_sick('./SICK_test_annotated.txt', is_Train=False)
    '''
    # Split train/dev set
    # 最好是随机选择一部分用于dev验证（这里先不管验证，先跑起来再说）
    '''
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # 前半部分是训练集，后半部分是测试集
    s1_train, s1_dev = s1[:dev_sample_index], s1[dev_sample_index:]
    s2_train, s2_dev = s2[:dev_sample_index], s2[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    '''
    return s1, s2, y, lables, s1_test, s2_test, y_test, lables_test


def train(s1, s2, y, lables, s1_test, s2_test, y_test, lables_test):

    print("Training...")
    starttime = datetime.datetime.now()
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            my_model = MSEM_WI(config, word_embedding)

            # Define Training procedure（定义训练步骤，实际就是定义训练的optimizer）
            # 以下等价于 train_op = optimizer.minimize(loss, global_step=global_step)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)  # 优化算法
            grads_and_vars = optimizer.compute_gradients(my_model.loss)  # 梯度，方差
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries（将模型和总结写到该目录）
            timestamp = str(int(time.time()))  # 记录当前时间的时间戳
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy（保存loss和accuracytf.summary.scalar用来显示标量信息）
            loss_summary = tf.summary.scalar("loss", my_model.loss)
            acc_summary = tf.summary.scalar("accuracy", my_model.accuracy)
            f1_summary = tf.summary.scalar("f1", my_model.F1)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(s1_batch, s2_batch, y_batch, lables_batch):
                """
                A single training step
                """
                feed_dict = {
                    my_model.sentence_one_word: s1_batch,
                    my_model.sentence_two_word: s2_batch,
                    my_model.y_true: lables_batch,
                    my_model.y: y_batch,
                    my_model.is_train: True
                }
                # global_step用于记录全局的step数，就是当前运行到的step
                _, step, summaries, loss, acc, acc_direct, f1 = sess.run(
                    [train_op, global_step, train_summary_op, my_model.loss, my_model.accuracy, my_model.accuracy_direct, my_model.F1],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, acc_direct {:g}, f1 {:g}".format(time_str, step, loss, acc, acc_direct, f1))
                train_summary_writer.add_summary(summaries, step)
                endtime = datetime.datetime.now()
                print('训练时间(分钟):', (endtime - starttime).seconds/60)

            def test_step(s1_batch, s2_batch, y_batch, lables_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    my_model.sentence_one_word: s1_batch,
                    my_model.sentence_two_word: s2_batch,
                    my_model.y_true: lables_batch,
                    my_model.y: y_batch,
                    my_model.is_train: False
                }
                step, summaries, loss, acc, f1, correct_num, y, yhat= sess.run(
                    [global_step, dev_summary_op, my_model.loss, my_model.accuracy, my_model.F1, my_model.correct_num, my_model.y, my_model.yhat],
                    feed_dict)
                # print(y)
                # print(final_output)
                time_str = datetime.datetime.now().isoformat()
                # print("loss and acc in test_batch:")
                # print("{}: step {}, loss {:g}, mse {:g}, se {:g}".format(time_str, step, loss, mse, se))
                if writer:
                    writer.add_summary(summaries, step)
                return correct_num, y, yhat   # y, final_output 计算整体的均方误差

            # Generate batches
            batches = batch_iter(s1, s2, y, lables, config.batch_size, config.num_epochs)
            # Training loop. For each batch...
            for s1_batch, s2_batch, y_batch, lables_batch in batches:
                train_step(s1_batch, s2_batch, y_batch, lables_batch)
                current_step = tf.train.global_step(sess, global_step)    # 一个step就是一个batch

                # 每隔100个step，就对测试集进行一次测试。测试集也需要分batch读入，否则会内存不够
                # 测试集分批读入
                if current_step > 10:
                    if current_step % config.evaluate_every == 0:
                        print("\nTesting:")
                        '''
                        train_batches = batch_iter(s1, s2, y, lables, config.batch_size, 1, shuffle=True)
                        se_total_train = 0
                        batch_num_train = 0
                        for s1_batch, s2_batch, y_batch, lables_batch in train_batches:
                            se_train = test_step(s1_batch, s2_batch, y_batch, lables_batch)
                            batch_num_train = batch_num_train + 1
                            se_total_train = se_total_train + se_train
                        print('batch_train:', batch_num_train)
                        print("total_train:", batch_num_train * config.batch_size)
                        print('se_total_train:', se_total_train)
                        train_mse = se_total_train / (batch_num_train * config.batch_size)
                        print('train mse:', train_mse)
                        '''
                        test_batches = batch_iter(s1_test, s2_test, y_test, lables_test, config.batch_size, 1, shuffle=True)
                        correct_total_num = 0
                        # se_total = 0
                        batch_num = 0
                        y_total = []
                        y_total = np.array(y_total)
                        # final_output_total = []
                        # final_output_total = np.array(final_output_total)
                        yhat_total = []
                        yhat_total = np.array(yhat_total)

                        print('testing: ')
                        for s1_batch, s2_batch, y_batch, lables_batch in test_batches:
                            correct_num, y, yhat = test_step(s1_batch, s2_batch, y_batch, lables_batch)
                            y_total = np.append(y_total, y)
                            # final_output_total = np.append(final_output_total, final_output)
                            yhat_total = np.append(yhat_total, yhat)
                            # print(y_total)
                            # print('len_y_totoal:', len(y_total))
                            # print(final_output_total)
                            # print('len_final_output_totoal:', len(final_output_total))
                            batch_num = batch_num + 1   # batch数自增
                            correct_total_num = correct_total_num + correct_num   # 1个batch的分类正确数
                        print('batch:', batch_num)
                        print("total:", batch_num * config.batch_size)
                        print('correct_total_num:', correct_total_num)
                        test_acc = correct_total_num / (batch_num * config.batch_size)
                        print('test acc:', test_acc)
                        print('test acc direct:', acc_calculate(y_total, yhat_total))
                        # print('pearson:', pearsonr(y_total, real_yhat_total)[0])
                        # print('spearman:', spearmanr(y_total, real_yhat_total)[0])
                        print('test f1:', f1_score(y_total, yhat_total))
                    '''
                    if current_step % config.checkpoint_every == 0:  # 每隔100个step，就保存模型
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    '''


def acc_calculate(y, predict_y):
    return np.mean((y == predict_y).astype(float))


def mse_calculate(y, predict_y):
    return ((predict_y - y) ** 2).mean()


if __name__ == '__main__':
    # sr_word2id, word_embedding = build_glove_dic('./glove.6B.300d.txt')
    s1, s2, y, lables, s1_test, s2_test, y_test, lables_test = preprocess()  # 预处理
    train(s1, s2, y, lables, s1_test, s2_test, y_test, lables_test)

