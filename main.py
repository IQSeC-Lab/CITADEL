from distutils import core
import os

import datetime as dt
import logging
import numpy as np
import time
import torch
import os
import sys
import argparse
import math
import xgboost as xgb

from collections import Counter, defaultdict
from dateutil.relativedelta import relativedelta
from pprint import pformat
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC

# local imports
import data
from joblib import dump
import utils


def main():
    """
    Step (0): Init log path and parse args.
    """
    args = utils.parse_args()


    
    log_file_path = args.log_path
    
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            )
    else:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG,
                            )
    
    logging.info('Running with configuration:\n' + pformat(vars(args)))


    """
    Step (1): Prepare the training dataset. Load the feature vectors and labels.
    """
    data_dir = args.data.split('/')[-1]
    logging.info(f'Dataset directory name {data_dir}')

    logging.info(f'Loading {args.data} training dataset')
    if args.data.startswith('tesseract') or \
        args.data.startswith('gen_tesseract') or \
        args.data.startswith('fam_tesseract') or \
        args.data.startswith('emberv2'):
        X_train, y_train, all_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
    else:
        logging.info(f'For API_GRAPH dataset start with the month 2012-01to2012-12_selected.')
        logging.info(f'For Androzoo dataset start with the month 2019-01to2019-12_selected.')
        
        X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
        # all_train_family has 'benign'
        ben_len = X_train.shape[0] - y_train_family.shape[0]
        y_ben_family = np.full(ben_len, 'benign')
        all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)
            
    train_families = set(all_train_family)
    
    # count label distribution
    counted_labels = Counter(y_train)
    logging.info(f'Loaded X_train: {X_train.shape}, {y_train.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_train: {Counter(y_train)}')

    # the index mapping for the first training set
    new_y_mapping = {}
    for _, label in enumerate(np.unique(y_train)):
        new_y_mapping[label] = label
    





if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f'time elapsed: {end - start} seconds')


