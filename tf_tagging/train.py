#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 中文基于字序列标注
# 原文： https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html

import tensorflow as tf

hidden_size = 512
ntags = 9
lr = 0.001
embeddings = [] # array of existing embeddings of every characters

word_ids = tf.placeholder(tf.int32, shape=[None,None]) # batch size, max length of sentence in batch
sequence_lengths = tf.placeholder(tf.int32, shape=[None]) # shape = batch size

L = tf.Variable(embeddings, dtype=tf.float32, trainable=False) # 不要用 tfconstant 这里embedding层不做更新
word_embeddings = tf.nn.embedding_lookup(L, word_ids)  # pretrained word embeddings

lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size)

# 这里虽然用的是同一个lstm_cell
# 但是因为bidirectional_dynamic_rnn在不同的scope里面call它
# 所以参数是不一样的
(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell,
    lstm_cell, word_embeddings, sequence_length=sequence_lengths,
    dtype=tf.float32)
#加了sequence_lengths后，会把有效的句子长度加进去，pad的0忽略掉
#output_fw, output_bw分别表示的是前向和后向每一个rnn的output

context_rep = tf.concat([output_fw, output_bw], axis=-1)  #把前向的隐藏层和后向的隐藏层合并
W = tf.get_variable("W", shape=[2 * hidden_size, ntags],
                    dtype=tf.float32)

b = tf.get_variable("b", shape=[ntags], dtype=tf.float32,
                    initializer=tf.zeros_initializer())

ntime_steps = tf.shape(context_rep)[1]
context_rep_flat = tf.reshape(context_rep, [-1, 2 * hidden_size])
pred = tf.matmul(context_rep_flat, W) + b
scores = tf.reshape(pred, [-1, ntime_steps, ntags])  #算好分数后，再重新reshape成[batchsize, timesteps, num_class]

#接下来，有两种计算序列标注的Loss
# 一种是基于softmax，就是把每个输出的argmax的那个概率相乘，loss=-log(P(y_1)*P(y_2)*P(y_3)*..P(y_t_max))
# 二种是基于crf，会考虑周边的序列标注情况再来计算每条可能的序列的概率再normalize，最后loss=-log(P(sequence))
# shape = (batch, sentence)
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

#基于softmax的loss
'''
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
# shape = (batch, sentence, nclasses)
#tf.sequence_mask that transforms sequence lengths into boolean vectors (masks).
mask = tf.sequence_mask(sequence_lengths)
# apply mask
losses = tf.boolean_mask(losses, mask)
loss = tf.reduce_mean(losses)
'''


#基于crf的loss
# one-hot的label表示
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
scores, labels, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)

optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)

# evalueate
# softmax predict
#labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
    score, transition_params)

