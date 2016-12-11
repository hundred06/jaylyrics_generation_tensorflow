# -*-  coding:utf-8 -*-
''' model for automatic speech recognition implemented in Tensorflow
author:

      iiiiiiiiiiii            iiiiiiiiiiii         !!!!!!!             !!!!!!    
      #        ###            #        ###           ###        I#        #:     
      #      ###              #      I##;             ##;       ##       ##      
            ###                     ###               !##      ####      #       
           ###                     ###                 ###    ## ###    #'       
         !##;                    `##%                   ##;  ##   ###  ##        
        ###                     ###                     $## `#     ##  #         
       ###        #            ###        #              ####      ####;         
     `###        -#           ###        `#               ###       ###          
     ##############          ##############               `#         #     
     
date:2016-12-01
'''

import numpy as np
import tensorflow as tf

from preprocess import TextParser
from seq2seq_rnn import Model as Model_rnn
from utils import count_params
from utils import logging

import argparse
import time
import os
from six.moves import cPickle

class Trainer():
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default='./data/luxun/',
                       help='set the data directory which contains input.txt')

        parser.add_argument('--save_dir', default='./save/',
                       help='set directory to store checkpointed models')

        parser.add_argument('--log_dir', default='./log/',
                       help='set directory to store checkpointed models')

        parser.add_argument('--rnn_size', type=int, default=128,
                       help='set size of RNN hidden state')

        parser.add_argument('--embedding_size', type=int, default=128,
                       help='set size of word embedding')

        parser.add_argument('--num_layers', type=int, default=2,
                       help='set number of layers in the RNN')

        parser.add_argument('--model', default='seq2seq_rnn',
                       help='set the model')

        parser.add_argument('--rnncell', default='lstm',
                       help='set the cell of rnn, eg. rnn, gru, or lstm')

        parser.add_argument('--attention', type=bool, default=False,
                       help='set attention mode or not')

        parser.add_argument('--batch_size', type=int, default=64,
                       help='set minibatch size')

        parser.add_argument('--seq_length', type=int, default=32,
                       help='set RNN sequence length')

        parser.add_argument('--num_epochs', type=int, default=10000,
                       help='set number of epochs')

        parser.add_argument('--save_every', type=int, default=1000,
                       help='set save frequency while training')

        parser.add_argument('--grad_clip', type=float, default=20.,
                       help='set clip gradients when back propagation')

        parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='set learning rate')

        parser.add_argument('--decay_rate', type=float, default=0.98,
                       help='set decay rate for rmsprop')                       

        parser.add_argument('--keep', type=bool, default=True,
		       help='init from trained model')

        args = parser.parse_args()
        self.train(args)

    def train(self,args):
	''' import data, train model, save model
	'''
        text_parser = TextParser(args.data_dir, args.batch_size, args.seq_length)
        args.vocab_size = text_parser.vocab_size
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
    
        if args.keep is True:
            # check if all necessary files exist 
	    if os.path.exists(os.path.join(args.save_dir,'config.pkl')) and \
		os.path.exists(os.path.join(args.save_dir,'words_vocab.pkl')) and \
		ckpt and ckpt.model_checkpoint_path:
                with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
                    saved_model_args = cPickle.load(f)
                with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
                    saved_words, saved_vocab = cPickle.load(f)
	    else:
		raise ValueError('configuration doesn"t exist!')

	if args.model == 'seq2seq_rnn':
            model = Model_rnn(args)
	else:
	    # TO ADD OTHER MODEL
	    pass
	trainable_num_params = count_params(model,mode='trainable')
	all_num_params = count_params(model,mode='all')
	args.num_trainable_params = trainable_num_params
	args.num_all_params = all_num_params
	print(args.num_trainable_params) 
	print(args.num_all_params) 
        with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
            cPickle.dump(args, f)
        with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
            cPickle.dump((text_parser.vocab_dict, text_parser.vocab_list), f)

        with tf.Session() as sess:
            if args.keep is True:
	        print('Restoring')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
	    else:
		print('Initializing')
    	        sess.run(model.initial_op)

            for e in range(args.num_epochs):
                start = time.time()
                #sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                sess.run(tf.assign(model.lr, args.learning_rate))
	        model.initial_state = tf.convert_to_tensor(model.initial_state) 
                state = model.initial_state.eval()
		total_loss = []
                for b in range(text_parser.num_batches):
                    x, y = text_parser.next_batch()
		    print('flag')
                    feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                    train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
		    total_loss.append(train_loss)
                    print("{}/{} (epoch {}), train_loss = {:.3f}" \
                                .format(e * text_parser.num_batches + b, \
                                args.num_epochs * text_parser.num_batches, \
                                e, train_loss))
                    if (e*text_parser.num_batches+b)%args.save_every==0 or (e==args.num_epochs-1 and b==text_parser.num_batches-1): 
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        model.saver.save(sess, checkpoint_path, global_step = e)
                        print("model has been saved in:"+str(checkpoint_path))
                end = time.time()
		delta_time = end - start
		ave_loss = np.array(total_loss).mean()
		logging(model,ave_loss,e,delta_time,mode='train')
		if ave_loss < 0.5:
		    break

if __name__ == '__main__':
    trainer = Trainer()
