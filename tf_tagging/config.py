#!/usr/bin/env python
# -*- coding: utf-8 -*-

class config():
    dim = 300

    glove_filename = "data/embedding/news_tensite_ch_clean.model".format(dim)
    trimmed_filename = "data/news_tensite_ch_clean_{}d.trimmed.npz".format(dim)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"

    dev_filename = "data/msra/msr_training.utf8.val"
    test_filename = "data/msra/msr_training.utf8.test"
    train_filename = "data/msra/msr_training.utf8.train"
    max_iter = None
    lowercase = True
    train_embeddings = False
    nepochs = 20
    dropout = 0.5
    batch_size = 20
    lr = 0.001
    lr_decay = 0.9
    nepoch_no_imprv = 3
    hidden_size = 300
    crf = True # if crf, training is 1.7x slower
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"