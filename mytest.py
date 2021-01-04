# -*- coding: utf-8 -*-
# @Time    : 2020/1/30 16:06
# @Author  : chenjunyu
# @FileName: mytest
# @Software: PyCharm


from utilities import *
from MSEM_WI_model import *
from configs import config


def testset_process():
    print('test data loading...')
    s1, s2, y = read_msrp()
    return s1, s2, y


def test(s1, s2, y):
    # ==================================================
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./runs/1580369566/checkpoints/model-2500.meta')
        new_saver.restore(sess, './runs/1580369566/checkpoints/model-2500')
        graph = tf.get_default_graph()
        input_s1 = tf.placeholder(tf.int32, [None, self.num_sentence_words], name="sentence_one_word")
        input_s2 = graph.get_tensor_by_name('input/sentence_two_word')
        input_y = graph.get_tensor_by_name('input/true_labels')
        acc = graph.get_tensor_by_name('accuracy/accuracy')

        accuracy = sess.run(acc, feed_dict={input_s1:s1, input_s2:s2, input_y:y})
        return accuracy


if __name__ == '__main__':
    s1, s2, y = testset_process()
    acc = test(s1, s2, y)
    print('acc:', acc)

