from distutils import core
import os
import random

import datetime as dt
import logging
import numpy as np
import time
import torch
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
from models import SimpleEncClassifier, MLPClassifier
from train import train_encoder, train_classifier
from similarity_score import Similarity
import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Step (0): Init log path and parse args.
    """
    args = utils.parse_args()

    log_file_path = args.log_path
    
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
    else:
        logging.basicConfig(filename=log_file_path, filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG)
    logging.info('Running with configuration:\n' + pformat(vars(args)))


    """
    Step (1): Prepare the training dataset. Load the feature vectors and labels.
    """
    data_dir = args.data.split('/')[-1]
    logging.info(f'Dataset directory name {data_dir}')

    logging.info(f'Loading {args.data} training dataset')
    logging.info(f'For API_GRAPH dataset start with the month 2012-01to2012-12_selected.')
    logging.info(f'For Androzoo dataset start with the month 2019-01to2019-12_selected.')
    
    X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args.data, args.train_start, args.train_end)
    # all_train_family has 'benign'
    ben_len = X_train.shape[0] - y_train_family.shape[0]
    y_ben_family = np.full(ben_len, 'benign')
    all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)
        
    train_families = set(all_train_family)
    logging.info(f"Number of y_train_family = {len(y_train_family)}")
    logging.info(f"Count of y_train_family = {Counter(y_train_family)}")
    logging.info(f'All train family = {all_train_family}')

    """
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
    """
    # count label distribution
    counted_labels = Counter(y_train)
    logging.info(f'Loaded X_train: {X_train.shape}, {y_train.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_train: {Counter(y_train)}')

    # the index mapping for the first training set
    new_y_mapping = {}
    for _, label in enumerate(np.unique(y_train)):
        new_y_mapping[label] = label
    
    """
    Step (2): Variable names and file names.
    """
    # some commonly used variables.
    if args.train_start != args.train_end:
        train_dataset_name = f'{args.train_start}to{args.train_end}'
    else:
        train_dataset_name = f'{args.train_start}'

    SAVED_MODEL_FOLDER = 'models/'
    # only based on malicious training samples
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    # convert y_train to y_train_binary
    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2


    # get the labeled and unlabeled indices
    labeled_indices, unlabeled_indices = utils.get_labeled_unlabeled_indices(X_train, y_train, percentage_labeled=50)
    logging.info(f'Number of labeled indices: {len(labeled_indices)}')
    logging.info(f'Number of unlabeled indices: {len(unlabeled_indices)}')


    """
    Step (3): Train the encoder model.
    `encoder` needs to have the same APIs.
    If they don't have the required API, we could use a wrapper.
    """
    cls_gpu = True
    if args.encoder == None:
        # We will not use an encoder. The input features are used directly.
        logging.info('Not using an encoder. Using the input features directly.')
    elif args.encoder == 'mlp':
        # assert args.encoder == args.classifier, "mlp encoder is from mlp classifier"
        if args.multi_class == True:
            output_dim = len(np.unique(y_train))
        else:
            output_dim = BIN_NUM_CLASSES
        mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, output_dim)
        enc_dims = mlp_dims[:-1]
        encoder = MLPClassifier(mlp_dims)
    elif args.encoder == 'cnn':
        raise Exception(f'The encoder {args.encoder} is not supported yet.')
        pass
    elif args.encoder == 'simple-enc-mlp':
        # Enc + MLP model 
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        encoder_name = 'simple_enc_classifier'
    else:
        raise Exception(f'The encoder {args.encoder} is not supported yet.')
    

    MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, data_dir)
    utils.create_folder(MODEL_DIR)

    # set optimizer for the model
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(encoder.parameters(), lr=args.learning_rate)
    else:
        raise Exception(f'The optimizer {args.optimizer} is not supported yet.')
    
    ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_lr{args.learning_rate}_{args.optimizer}_.pth')
    logging.info(f'Initial encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')

    if args.ssl == True and args.split_train == True:
        X_train_final = X_train[labeled_indices]
        y_train_final = y_train[labeled_indices]
        y_train_binary_final = y_train_binary[labeled_indices]

        X_unlabeled = X_train[unlabeled_indices]
        # y_unlabeled = y_train[unlabeled_indices]
        # y_unlabeled_binary = y_train_binary[unlabeled_indices]
        logging.info(f'Splitted Labeled X_train_final.shape: {X_train_final.shape}')
        logging.info(f'Splitted Labeled y_train_final.shape: {y_train_final.shape}')
        
    else:
        X_train_final = X_train
        y_train_final = y_train
        y_train_binary_final = y_train_binary

    upsample_values = None
    
    logging.info(f'upsample_values {upsample_values}')
    logging.info(f'X_train_final.shape: {X_train_final.shape}')
    logging.info(f'y_train_final.shape: {y_train_final.shape}')
    logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
    logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
    logging.info(f'y_train_final: {Counter(y_train_final)}')

    # removing all singleton families
    # make all singleton families the same as "unknown"
    if args.encoder != None and args.encoder.startswith('simple-enc-mlp') == True:
        counted_y_train = Counter(y_train)
        singleton_families = [family for family, count in counted_y_train.items() if count == 1]
        logging.info(f'Singleton families: {singleton_families}')
        logging.info(f'Number of singleton families: {len(singleton_families)}')

        X_train_final = np.array([X_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_final = np.array([y_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_binary_final = np.array([y_train_binary[i] for i, family in enumerate(y_train) if family not in singleton_families])
        
        all_train_family = np.array([all_train_family[i] for i, family in enumerate(y_train) if family not in singleton_families])
        logging.info(f'After removing singleton families: X_train_final.shape, {X_train_final.shape}, y_train_final.shape, {y_train_final.shape}')

    # train the encoder model if it does not already exist.
    # train mlp encoder in the classifier training step
    if args.encoder in ['enc', 'simple-enc-mlp']:
        if not os.path.exists(ENC_MODEL_PATH): # or args.retrain_first == True
            s1 = time.time()
            if args.ssl == True:
                train_encoder(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                                    optimizer, args.epochs, ENC_MODEL_PATH, adjust = True, save_best_loss = False, \
                                    save_snapshot = args.snapshot, X_unlabeled = X_unlabeled)
            else:
                train_encoder(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                                    optimizer, args.epochs, ENC_MODEL_PATH, adjust = True, save_best_loss = False, \
                                    save_snapshot = args.snapshot)
            e1 = time.time()
            logging.info(f'Training Encoder model time: {(e1 - s1):.3f} seconds')
            
            logging.info('Saving the model...')
            utils.save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
            logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')

        else:
            logging.info('Loading the model...')
            state_dict = torch.load(ENC_MODEL_PATH)
            encoder.load_state_dict(state_dict['model'])
    elif args.encoder == 'mlp':
        train_classifier(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                        optimizer, args.mlp_epochs, ENC_MODEL_PATH, \
                        save_best_loss = False, multi = args.multi_class)
        logging.info('Saving the model...')
        utils.save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
        logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')


    """
    Select the classifier model.
    """
    # prepare X_feat and X_feat_tensor if they are embeddings
    if args.cls_feat == 'encoded':
        X_train_tensor = torch.from_numpy(X_train).float()
        if torch.cuda.is_available():
            X_train_tensor = X_train_tensor.cuda()
            X_feat_tensor = encoder.cuda().encode(X_train_tensor)
            X_train_feat = X_feat_tensor.cpu().detach().numpy()
        else:
            X_train_feat = encoder.encode(X_train_tensor).numpy()
    else:
        # args.cls_feat == 'input'
        X_train_feat = X_train
    X_train_feat = X_train
    logging.info(f'X_train_feat.shape: {X_train_feat.shape}')

    
    if args.classifier in ['simple-enc-mlp'] or args.classifier == args.encoder:
        # we have already trained it as the sample selection model.
        classifier = encoder
        CLS_MODEL_PATH = ENC_MODEL_PATH
        cls_gpu = True
    elif args.classifier == 'mlp':
        if args.encoder == 'mlp':
            classifier = encoder
            CLS_MODEL_PATH = ENC_MODEL_PATH
        else:
            raise Exception(f'Different classifier and encoder is not implemented yet')
        cls_gpu = True
    
    

    """
    For semi supervised learning
    """
    # if args.ssl == True:
        # here, we can use our own pseudo labeling method
        # pseudo_indices, pseudo_labels = similarity.get_highly_similar_samples(X_train_feat, X_test_accum_feat, y_test_pred, \
        #                                                 feature_labels=y_test_binary, batch_size=args.bsize, n_sample=200)
        # this pseudo_labels is just considering confidence score
        # X_pseudo, y_pseudo = utils.generate_pseudo_labels(encoder, X_unlabeled, threshold=0.9)

        # add the pseudo samples to the training set
        # X_train_final = np.concatenate((X_train_final, X_pseudo), axis=0)
        # y_train_final = np.concatenate((y_train_final, y_pseudo), axis=0)
        # y_train_binary_final = np.array([1 if item != 0 else 0 for item in y_train_final])





    
    
    # saving the results

    fout = open(args.result, 'w')
    fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\n')
    fam_out = open(args.result.split('.csv')[0]+'_family.csv', 'w')
    fam_out.write('Month\tNew\tFamily\tFNR\tCnt\n')
    stat_out = open(args.result.split('.csv')[0]+'_stat.csv', 'w')
    stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')
    utils.eval_classifier(args, classifier, args.train_end, X_train_feat, y_train_binary, all_train_family, train_families, \
                    fout, fam_out, stat_out, gpu = cls_gpu, multi = args.eval_multi)
    sample_out = open(args.result.split('.csv')[0]+'_sample.csv', 'w')
    sample_out.write('date\tCount\tIndex\tTrue\tPred\tFamily\tScore\n')
    sample_out.flush()
    sample_explanation = open(args.result.split('.csv')[0]+'_sample_explanation.csv', 'w')
    sample_explanation.write('date\tCorrect\tWrong\tBenign\tMal\tNew_fam_cnt\tNew_fam\tUnique_fam\n')
    sample_explanation.flush()
    logging.info(f'1st step running complete.')


    
    """
    Now the following of the code is for active learning and sample selection if needed
    """

    start_date = dt.datetime.strptime(args.test_start, '%Y-%m')
    end_date = dt.datetime.strptime(args.test_end, '%Y-%m')
    cur_month = start_date
    # the index mapping for the first training set

    month_loop_cnt = 0
    prev_train_size = X_train.shape[0]
    cur_sample_indices = []
    pseudo_samples = []
    if args.encoder != None:
        NEW_ENC_MODEL_PATH = ENC_MODEL_PATH.split('.pth')[0] + f'_retrain_.pth'


    # active learning loop
    while cur_month <= end_date:
        """
        Step (4): Prepare the test dataset. Load the feature vectors and labels.
        """
        cur_month_str = cur_month.strftime('%Y-%m')
        logging.info(f'Loading {args.data} test dataset for the month {cur_month_str}')
        X_test, y_test, y_test_family = data.load_range_dataset_w_benign(args.data, cur_month_str, cur_month_str)
        # all_test_family has 'benign'
        ben_test_len = X_test.shape[0] - y_test_family.shape[0]
        y_ben_test_family = np.full(ben_test_len, 'benign')
        all_test_family = np.concatenate((y_test_family, y_ben_test_family), axis=0)

        test_families = set(all_test_family)
        logging.info(f"Number of y_test_family = {len(y_test_family)}")
        #logging.info(f"Count of y_test_family = {Counter(y_test_family)}")
        #logging.info(f'All test family = {all_test_family}')
        counted_test_labels = Counter(y_test)
        logging.info(f'Loaded X_test: {X_test.shape}, {y_test.shape}')
        logging.info(f'y_test labels: {np.unique(y_test)}')
        logging.info(f'y_test: {Counter(y_test)}')

        y_test_binary = np.array([1 if item != 0 else 0 for item in y_test])
        
        # compute the embedding once
        # this could be used to retrain the classifier
        X_test_tensor = torch.from_numpy(X_test).float()
        if args.encoder != None:
            if torch.cuda.is_available():
                X_test_feat_tensor = encoder.cuda().encode(X_test_tensor.cuda())
                X_test_encoded = X_test_feat_tensor.cpu().detach().numpy()
            else:
                X_test_encoded = encoder.encode(X_test_tensor).numpy()
        
        if args.cls_feat == 'encoded':
            X_test_feat = X_test_encoded
        else:
            X_test_feat = X_test
        
        X_test_feat = X_test ### check this line, later needed to remove
        
        # Only month_loop_cnt == 0 will we update the accum data with new month data
        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == args.test_start:
                X_test_accum = X_test
                y_test_accum = y_test
                all_test_family_accum = all_test_family
                X_test_accum_feat = X_test_feat # for the classifier
            else:
                X_test_accum = np.concatenate((X_test_accum, X_test), axis=0)
                y_test_accum = np.concatenate((y_test_accum, y_test), axis=0)
                all_test_family_accum = np.concatenate((all_test_family_accum, all_test_family), axis=0)
                X_test_accum_feat = np.concatenate((X_test_accum_feat, X_test_feat), axis=0) # for the classifier
        elif month_loop_cnt == 0:
            X_test_accum = X_test
            y_test_accum = y_test
            all_test_family_accum = all_test_family
            X_test_accum_feat = X_test_feat # for the classifier
        
        y_test_binary_accum = np.array([1 if item != 0 else 0 for item in y_test_accum])
        
        """
        Evaluate the test performance.
        """
        logging.info(f'Testing on {cur_month_str}')
        y_test_pred, neg_by_fam, family_to_idx = utils.eval_classifier(args, classifier, cur_month_str, X_test_feat, y_test_binary, all_test_family, train_families, \
                        fout, fam_out, stat_out, gpu = cls_gpu, multi = args.eval_multi)
        


        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == args.test_start: # for api_graph_dataset, it would be '2013-01'
                y_test_pred_accum = y_test_pred
            else:
                y_test_pred_accum = np.concatenate((y_test_pred_accum, y_test_pred), axis=0)
        elif month_loop_cnt == 0:
            y_test_pred_accum = y_test_pred
        


        # """
        # Step (5): Psuedo labeling and sample selection
        # # if args.pseudolabel == True:
        # """
        # if args.pseudolabel == True:
        #     # Train and Test Resprentation Similiarity Score Calculation
        #     similarity = Similarity()
        #     # Here, we are calculating the top K=5 similar indices
        #     # topk_sim_weight, topk_sim_indices = similarity.topk_similar(5, X_test_feat, X_train_feat)
        #     # # print("Top K similar Weights: ", topk_sim_weight)
        #     # print("Top K similar Indices: ", topk_sim_indices)

        #     # pseudo_labels, topk_info = similarity.get_majority_label(topk_sim_weight, topk_sim_indices, \
        #     #                                             feature_labels=y_train_binary, batch_size=args.bsize)

        #     # print("Pesudo Labels shape: ", pseudo_labels.shape)
        #     # print("Topk_info shape: ", topk_info.shape)

        #     # Sample Selection
        #     # Here, we will be selecting the samples based on the high confidence scores
        #     # similarity.get_high_confidence_samples are not implemented yet
        #     # pseudo_indices, pseudo_labels = similarity.get_highly_similar_samples(X_train_feat, X_test_accum_feat, y_test_pred, \
        #     #                                                 feature_labels=y_test_binary, batch_size=args.bsize, n_sample=200)
            
        #     pseudo_indices = random.sample(range(0, X_test_accum_feat.shape[0]), 200)
        #     pseudo_labels = y_test_pred[pseudo_indices]
        #     #pseudo_indices, pseudo_labels = 
            
        #     print("Pseudo Indices shape: ", pseudo_indices.shape)
        #     print("Pseudo Labels shape: ", pseudo_labels.shape)
        #     print("Pseudo Labels: ", pseudo_labels)
            
        # """
        # Step (6): Expand or Update the selected pseudo sample for SSL training.
        # """
        # if args.accumulate_data == True and month_loop_cnt == 0:
        #     if cur_month_str == args.test_start:
        #         X_pseudo_test_accum = X_test_accum[pseudo_indices]
        #         y_pseudo_test_accum = pseudo_labels  # here, we didn't add y_test[] which are actual labels annotated by the malware analyst
        #         all_pseudo_test_family_accum = all_test_family_accum[pseudo_indices]
        #         X_pseudo_test_accum_feat = X_test_accum_feat[pseudo_indices] # for the classifier
        #     else:
        #         X_pseudo_test_accum = np.concatenate((X_pseudo_test_accum, X_test_accum[pseudo_indices]), axis=0)
        #         y_pseudo_test_accum = np.concatenate((y_pseudo_test_accum, pseudo_labels), axis=0) 
        #         all_test_family_accum = np.concatenate((all_test_family_accum, all_test_family_accum[pseudo_indices]), axis=0)
        #         X_pseudo_test_accum_feat = np.concatenate((X_pseudo_test_accum_feat, X_test_accum_feat[pseudo_indices]), axis=0) # for the classifier
        # elif month_loop_cnt == 0:
        #     X_pseudo_test_accum = X_test_accum[pseudo_indices]
        #     y_test_accum = pseudo_labels  # here, we didn't add y_test[] which are actual labels annotated by the malware analyst
        #     all_test_family_accum = all_test_family_accum[pseudo_indices]
        #     X_pseudo_test_accum_feat = X_test_accum_feat[pseudo_indices] # for the classifier

        

        # #X_train = np.concatenate((X_train, X_test_accum[pseudo_indices]), axis=0)
        # #y_train_binary = np.concatenate((y_train_binary, pseudo_labels), axis=0)
        # X_pseudo_test_accum_tensor = torch.from_numpy(X_pseudo_test_accum).float()
        # y_pseudo_test_accum_tensor = torch.from_numpy(y_pseudo_test_accum).type(torch.int64)
        # y_pseudo_binary_cat_tensor = torch.from_numpy(to_categorical(y_pseudo_test_accum)).float()
        # labeled_families = ['benign' if label == 0 else 'malware' for i, label in enumerate(pseudo_labels)]
        # original_y = np.array(labeled_families)

        # """
        # Now, we will deleted the pseudo samples from the accummulated test set
        # """
        #  # Remove selected samples from test data
        # X_test_accum = np.delete(X_test_accum, pseudo_indices, axis=0)
        # X_test_accum_feat = np.delete(X_test_accum_feat, pseudo_indices, axis=0)
        # y_test_accum = np.delete(y_test_accum, pseudo_indices, axis=0)
        # all_test_family_accum = np.delete(all_test_family_accum, pseudo_indices, axis=0)
        # y_test_pred_accum = np.delete(y_test_pred_accum, pseudo_indices, axis=0)


        # X_train_final = X_train
        # y_train_final = y_train
        # y_train_binary_final = y_train_binary
        # upsample_values = None

        # logging.info(f'upsample_values {upsample_values}')
        # logging.info(f'X_train_final.shape: {X_train_final.shape}')
        # logging.info(f'y_train_final.shape: {y_train_final.shape}')
        # logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
        # logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
        # logging.info(f'y_train_final: {Counter(y_train_final)}')

        # pseudo_data = TensorDataset(X_pseudo_test_accum_tensor, y_pseudo_test_accum_tensor, y_pseudo_binary_cat_tensor)
        # pseudo_loader = DataLoader(pseudo_data, batch_size=args.bsize, shuffle=True)
        # """
        # Step (7): Train the classifier model.
        # """
        # if args.encoder_retrain == True:
        #     if args.al_optimizer == None:
        #         # use the same optimizer as the first model
        #         logging.info(f'Active learning using optimizer {args.optimizer}')
        #         pass
        #     elif args.al_optimizer == 'adam':
        #         # Adam optimizer
        #         optimizer_func = torch.optim.Adam
        #         logging.info(f'Active learning using optimizer {args.al_optimizer}')
        #     elif args.al_optimizer == 'sgd':
        #         # SGD optimizer
        #         optimizer_func = torch.optim.SGD
        #         logging.info(f'Active learning using optimizer {args.al_optimizer}')

        #     # warm start learning rate, e.g., 0.1 * args.learning_rate
        #     optimizer = optimizer_func(encoder.parameters(), lr=args.warm_learning_rate)
        #     al_total_epochs = args.al_epochs


        #     s2 = time.time()
        #     logging.info('Training Encoder model...')
        #     train_encoder_func(args, encoder, X_train_final, y_train_final, y_train_binary_final,
        #                     optimizer, al_total_epochs, NEW_ENC_MODEL_PATH,
        #                     weight = None,
        #                     adjust = True, warm = not args.cold_start, save_best_loss = False, pseudo_loader = pseudo_loader)
        #     e2 = time.time()
        #     logging.info(f'Training Encoder model time: {(e2 - s2):.3f} seconds')

        # if args.cls_feat == 'encoded':
        #     X_train_tensor = torch.from_numpy(X_train).float()
        #     if torch.cuda.is_available():
        #         X_train_tensor = X_train_tensor.cuda()
        #         X_feat_tensor = encoder.cuda().encode(X_train_tensor)
        #         X_train_feat = X_feat_tensor.cpu().detach().numpy()
        #     else:
        #         X_train_feat = encoder.encode(X_train_tensor).numpy()
        # else:
        #     # args.cls_feat == 'input'
        #     X_train_feat = X_train

        # if args.classifier in ['simple-enc-mlp'] or args.classifier == args.encoder:
        #     logging.info('Classifier model is the same as the encoder...')
        #     NEW_CLS_MODEL_PATH = NEW_ENC_MODEL_PATH

        # """
        
        # """






        prev_train_size = X_train.shape[0]
        # increment to next month
        cur_month += relativedelta(months=1)
        """
            uncertainty sampling
            
            implementation of coreset replay

            expanding the training set

        """
    
    # finish writing the result file
    fout.close()
    fam_out.close()
    sample_out.close()
    stat_out.close()
    # sample_score_out.close()
    sample_explanation.close()
    return







if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f'time elapsed: {end - start} seconds')


