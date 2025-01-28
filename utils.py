import os
import sys
import logging
import argparse
import math
import numpy as np
import torch

def parse_args():
    """Parse the command line configuration for a particular run.
    
    Returns:
        argparse.Namespace -- a set of parsed arguments.
    """
    p = argparse.ArgumentParser()

    p.add_argument('--data', help='The dataset to use.')
    
    # for debugging messages
    p.add_argument('--verbose', action='store_true',
                    help='whether to print the debugging logs.')

    '''
    p.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    p.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    p.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    p.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    p.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    '''



    p.add_argument('--mdate', help='Encoder model date to use.')
    #p.add_argument('--train', default=None, help='Train month. e.g., 2012-01')
    p.add_argument('--train_start', default=None, help='Train start month. e.g., 2012-01')
    p.add_argument('--train_end', default=None, help='Train end month. e.g., 2012-12')

    '''
    p.add_argument('--test_start', help='First test month.')
    p.add_argument('--test_end', help='Last test month.')

    p.add_argument('--ood', action='store_true', help='Use CAE OOD score to help sampling')
    p.add_argument('--local_pseudo_loss', action='store_true', help='Use local pseudo loss to select samples')
    p.add_argument('--reduce', type=str, choices=['none', 'max', 'mean'],
                    help='how to reduce the loss to compute the pseudo loss')
    
    p.add_argument('--unc', action='store_true', help='Uncertain sampling')

    p.add_argument('--result', type=str, help='file name to generate MLP performance csv result.')

    # encoder model
    p.add_argument('--encoder', default=None, \
                    choices=['cae', 'enc', 'mlp', \
                            'simple-enc-mlp'], \
                    help='The encoder model to get embeddings of the input.')
    p.add_argument('--encoder-retrain', action='store_true',
                   help='Whether to train the encoder again.')
    p.add_argument('--cold-start', action='store_true',
                   help='Whether to retrain the encoder from scratch.')
    
    # classifier
    p.add_argument('-c', '--classifier', default='svm',
                   choices=['mlp', 'svm', 'gbdt', \
                            'simple-enc-mlp'],
                   help='The target classifier to use.')
    # more arguments can be added here

    
    # arguments for the Encoder Classifier model.
    p.add_argument('--enc-hidden',
                help='The hidden layers of the encoder, example: "512-128-32"')
    p.add_argument('--bsize', default=None, type=int,
                   help='Training batch size.')
    p.add_argument('--plb', default=None, type=int,
                   help='Pseudo loss batch size.')
    p.add_argument('--sample-per-class', default=2, type=int,
                   help='Number of samples for each class in a batch.')
    
    p.add_argument('--learning_rate', default=0.01, type=float,
                   help='Overall learning rate.')
    p.add_argument('--warm_learning_rate', default=0.001, type=float,
                   help='Warm start learning rate.')
    
    p.add_argument('--lr_decay_rate', type=float, default=1,
                        help='decay rate for learning rate')
    p.add_argument('--lr_decay_epochs', type=str, default='30,1000,30',
                        help='where to decay lr. start epoch, end epoch, step size.')
    p.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'],
                        help='Choosing an optimzer')
    p.add_argument('--al_optimizer', default=None, type=str, choices=['adam', 'sgd'],
                        help='Choosing an optimzer')
    p.add_argument('--epochs', default=250, type=int,
                   help='Training epochs.')
    p.add_argument('--al_epochs', default=50, type=int,
                   help='Active learning training epochs.')
    p.add_argument('--xent-lambda', default=1, type=float,
                   help='lambda to scale the binary cross entropy loss.')
    
    p.add_argument('--retrain-first', action='store_true',
                   help='Whether to retrain the first model.')
    p.add_argument('--sampler', type=str, choices=['mperclass', 'proportional', 'half',
                    'triplet', 'random'],
                   help='The sampler to sample batches.')
    
    # arguments for the Contrastive Autoencoder and drift detection (build on the samples of top 7 families for example)
    p.add_argument('--cae-hidden',
                   help='The hidden layers of the giant autoencoder, example: "512-128-32", \
                         which in drebin_new_7 would make the architecture as "1340-512-128-32-7"')
    p.add_argument('--cae-batch-size', default=64, type=int,
                   help='Contrastive Autoencoder batch_size, use a bigger size for larger training set \
                        (when training, one batch only has 64/2=32 samples, another 32 samples are used for comparison).')
    p.add_argument('--cae-lr', default=0.001, type=float,
                   help='Contrastive Autoencoder Adam learning rate.')
    p.add_argument('--cae-epochs', default=250, type=int,
                   help='Contrastive Autoencoder epochs.')
    p.add_argument('--cae-lambda', default=1e-1, type=float,
                   help='lambda in the loss function of contrastive autoencoder.')
    p.add_argument('--margin', default=10.0, type=float,
                    help='Maximum margins of dissimilar samples when training contrastive autoencoder.')
    p.add_argument('--display-interval', default=10, type=int,
                    help='Show logs about loss and other information every xxx epochs when training the encoder.')
               
    '''
    p.add_argument('--log_path', type=str,
                   help='log file name.')

    args = p.parse_args()

    return args
    
