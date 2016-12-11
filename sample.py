# -*- coding:utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
import codecs

from preprocess import TextParser
from seq2seq_rnn import Model

def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=400,
                       help='number of words to sample')
    parser.add_argument('--start', default=u'如果',
                       help='prime text')
    parser.add_argument('--sample', type=str, default='weighted',
                       help='three choices:argmax,weighted,combined')

    args = parser.parse_args()
    sample(args)

def sample(args):
    # import configuration
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    # import the trained model
    model = Model(saved_args, True)
    with tf.Session() as sess:
	# initialize the model
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
	    # sample the new sequence word by word
            literature = model.sample(sess, words, vocab, args.n, args.start, args.sample)
    with codecs.open('result/sequence.txt','a','utf-8') as f:
        f.write(literature+'\n\n')
    print(literature)

if __name__ == '__main__':
    main()
