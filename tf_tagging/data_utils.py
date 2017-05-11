#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
#NONE = "O"
NONE = "s"
class CoNLLDataset(object):
    """
    Class that iterates over msra Dataset
    __iter__ method yields a tuple(words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is applied

    Example:
        data = MsraDataset(file)
        for sentence,tags in data:
            pass
    """
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None):
        """
            Args:
                filename: path to the file
                processing_words: (optional) function that takes a word as input
                processing_tags: (optional) function that takes a tag as input
                max_iter: (optional) max number of sentences to yield
            """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename,encoding="utf-8") as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    word, tag = line.split()
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print ("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print ("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags

def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print ("Building vocab...")
    vocab = set()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print ("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w", encoding="utf-8") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print ("- done. {} tokens".format(len(vocab)))

def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename,encoding="utf-8") as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename,encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            embedding = map(float, line[1:])
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(list(embedding))
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    print(filename)
    #with open(filename) as f:

    return np.load(filename)["embeddings"]



def get_processing_word(vocab_words=None,
                        lowercase=False):

    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3.  word id
        return word

    return f

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch



def _pad_sequences(sequences, pad_tok, max_length):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                                      pad_tok, max_length)
    return sequence_padded, sequence_length


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [0,0,1,2,3]
        tags = {"B": 1, "I": 2, "E": 3, "s":0}
        result = [(0, 1), (1, 2),(2,5)]
    """
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    start = 0
    for index,ch in enumerate(seq):
        if idx_to_tag[ch] == "B" or idx_to_tag[ch] =="s":
            if index-start!=0:
                chunks.append((start,index))
            start = index  # a new chunk

    chunks.append((start,index+1))
    return chunks
#tags = {"B": 1, "I": 2, "E": 3, "s":0}
#seq = [1,2,1,2,0]
#print(get_chunks(seq, tags))

'''
def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('-')[-1]


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok[0] == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

'''
