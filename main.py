from distutils import core
import os

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

from similarity_score import pesudo_labeling

def load_data(data_dir, start_month, end_month):
    """
    Load the training and validation data.
    """
    logging.info(f'Loading {data_dir} dataset from {start_month} to {end_month}')
    X, y, y_family = data.load_range_dataset_w_benign(data_dir, start_month, end_month)
    ben_len = X.shape[0] - y_family.shape[0]
    y_ben_family = np.full(ben_len, 'benign')
    all_family = np.concatenate((y_family, y_ben_family), axis=0)

    return X, y, all_family

def prepare_encoder_and_classifier(
        args, X_train, y_train, y_train_binary, all_train_family, NUM_FEATURES, BIN_NUM_CLASSES, \
            NUM_CLASSES, SAVED_MODEL_FOLDER, data_dir):

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
    
    ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_lr{args.learning_rate}_{args.optimizer}_Final.pth')
    logging.info(f'Initial encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')

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
        #if args.retrain_first == True or not os.path.exists(ENC_MODEL_PATH):
        s1 = time.time()
        train_encoder(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                            optimizer, args.epochs, ENC_MODEL_PATH, adjust = True, save_best_loss = False, \
                            save_snapshot = True)
        e1 = time.time()
        logging.info(f'Training Encoder model time: {(e1 - s1):.3f} seconds')
        
        logging.info('Saving the model...')
        utils.save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
        logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')
        # else:
        #     logging.info('Loading the model...')
        #     state_dict = torch.load(ENC_MODEL_PATH)
        #     encoder.load_state_dict(state_dict['model'])
    elif args.encoder == 'mlp':
        train_classifier(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                        optimizer, args.mlp_epochs, ENC_MODEL_PATH, \
                        save_best_loss = False, multi = args.multi_class)
        logging.info('Saving the model...')
        utils.save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
        logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')

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

    return cls_gpu, encoder, classifier, X_train_final, \
         y_train_binary_final, ENC_MODEL_PATH 


def evaluate_train_data(args, classifier, end_date, X_data, y_data_binary, \
                        all_data_family, data_families, cls_gpu, eval_multi, data_type):
    """
    Evaluate the classifier on the given dataset and write the results to files.
    """
    fout = open(args.result.split('.csv')[0] + f'_{data_type}.csv', 'w')
    fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\n')
    fam_out = open(args.result.split('.csv')[0] + f'_family_{data_type}.csv', 'w')
    fam_out.write('Month\tNew\tFamily\tFNR\tCnt\n')
    stat_out = open(args.result.split('.csv')[0] + f'_stat_{data_type}.csv', 'w')
    stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')

    utils.eval_classifier(args, classifier, end_date, X_data, y_data_binary, all_data_family, data_families, fout, fam_out, stat_out, gpu=cls_gpu, multi=eval_multi)

    fout.close()
    fam_out.close()
    stat_out.close()

def evaluate_valid_test_data(args, encoder, end_date, X_data, y_data, all_data_family, \
                              cls_gpu, eval_multi, data_type, fout, fam_out, stat_out):
    data_families = set(all_data_family)
    # logging.info(f"Number of y_valid_family = {y_valid_family}")
    logging.info(f'All {data_type} family = {all_data_family}')

    # count label distribution
    # counted_labels_valid = Counter(y_data)
    logging.info(f'Loaded X_{data_type}: {X_data.shape}, {y_data.shape}')
    logging.info(f'y_{data_type} labels: {np.unique(y_data)}')
    logging.info(f'y_{data_type}: {Counter(y_data)}')

    # convert y_valid to y_valid_binary
    y_data_binary = np.array([1 if item != 0 else 0 for item in y_data])

    # prepare X_feat and X_feat_tensor if they are embeddings
    if args.cls_feat == 'encoded':
        X_data_tensor = torch.from_numpy(X_data).float()
        if torch.cuda.is_available():
            X_data_tensor = X_data_tensor.cuda()
            X_feat_tensor_valid = encoder.cuda().encode(X_data_tensor)
            X_data_feat = X_feat_tensor_valid.cpu().detach().numpy()
        else:
            X_data_feat = encoder.encode(X_data_tensor).numpy()
    else:
        # args.cls_feat == 'input'
        X_data_feat = X_data

    utils.eval_classifier(args, encoder, end_date, X_data_feat, y_data_binary, all_data_family, data_families, \
                    fout, fam_out, stat_out, gpu = cls_gpu, multi = eval_multi)

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
    logging.info(f"Number of y_train_family = {y_train_family}")
    logging.info(f'All train family = {all_train_family}')

    # count label distribution
    # counted_labels = Counter(y_train)
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
    # # some commonly used variables.
    # if args.train_start != args.train_end:
    #     train_dataset_name = f'{args.train_start}to{args.train_end}'
    # else:
    #     train_dataset_name = f'{args.train_start}'

    SAVED_MODEL_FOLDER = "/home/ihossain/ISMAIL/SSL-malware/models/"
    # only based on malicious training samples
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    # convert y_train to y_train_binary
    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2


    """
    Step (3): Train the encoder model.
    `encoder` needs to have the same APIs.
    If they don't have the required API, we could use a wrapper.
    """
    logging.info(f'Pretrained model: {args.pretrined_model}')
    if args.pretrined_model == 'False':
        cls_gpu, encoder, classifier, X_train_final, \
            y_train_binary_final, ENC_MODEL_PATH = prepare_encoder_and_classifier(
            args, X_train, y_train, y_train_binary, all_train_family, NUM_FEATURES, BIN_NUM_CLASSES, \
                NUM_CLASSES, SAVED_MODEL_FOLDER, data_dir)

        # Train data evaluation
        evaluate_train_data(args, classifier, args.train_end, X_train_final, y_train_binary_final,\
                       all_train_family, train_families, cls_gpu, args.eval_multi, 'train')

        # sample_out = open(args.result.split('.csv')[0]+'_sample_train.csv', 'w')
        # sample_out.write('date\tCount\tIndex\tTrue\tPred\tFamily\tScore\n')
        # sample_out.flush()
        # sample_explanation = open(args.result.split('.csv')[0]+'_sample_explanation_train.csv', 'w')
        # sample_explanation.write('date\tCorrect\tWrong\tBenign\tMal\tNew_fam_cnt\tNew_fam\tUnique_fam\n')
        # sample_explanation.flush()

        #--------------------------------------------------------------------------------------------------
        """
        Step (4): Prepare the validation dataset. Load the feature vectors and labels.
        """
        # Validation data evaluation
        fout = open(args.result.split('.csv')[0]+'_valid.csv', 'w')
        fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\n')
        fam_out = open(args.result.split('.csv')[0]+'_family_valid.csv', 'w')
        fam_out.write('Month\tNew\tFamily\tFNR\tCnt\n')
        stat_out = open(args.result.split('.csv')[0]+'_stat_valid.csv', 'w')
        stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')

        for i in range(1, 7):
            valid_start = (dt.datetime.strptime(args.valid_start, '%Y-%m') + relativedelta(months=i)).strftime('%Y-%m')
            valid_end = (dt.datetime.strptime(args.valid_end, '%Y-%m') + relativedelta(months=i)).strftime('%Y-%m')
        
            # logging.info(f'Loading {args.data} validation dataset form {args.valid_start} to {args.valid_end}')
            
            X_valid, y_valid, all_valid_family = load_data(args.data, valid_start, valid_end)
                
            # Valid data evaluation
            evaluate_valid_test_data(args, classifier, valid_end, X_valid, y_valid,\
                       all_valid_family, cls_gpu, args.eval_multi, 'valid', fout, fam_out, stat_out)
        
        # sample_out = open(args.result.split('.csv')[0]+'_sample_valid.csv', 'w')
        # sample_out.write('date\tCount\tIndex\tTrue\tPred\tFamily\tScore\n')
        # sample_out.flush()
        # sample_explanation = open(args.result.split('.csv')[0]+'_sample_explanation_valid.csv', 'w')
        # sample_explanation.write('date\tCorrect\tWrong\tBenign\tMal\tNew_fam_cnt\tNew_fam\tUnique_fam\n')
        # sample_explanation.flush()
        #--------------------------------------------------------------------------------------------------

        logging.info(f'Run complete for Train and Validation.')

    else:
        MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, data_dir)
        encoder_name = 'simple_enc_classifier'
        ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_lr{args.learning_rate}_{args.optimizer}_Final.pth')
        logging.info(f'Initial encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')

        if args.encoder == 'simple-enc-mlp':
            enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES, args.enc_hidden, NUM_CLASSES)
            mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
            encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        # load the pre-trained encoder model
        logging.info("Model loading started...")
        state_dict = torch.load(ENC_MODEL_PATH, weights_only=False)
        
        encoder.load_state_dict(state_dict['model'])
        classifier = encoder
        cls_gpu = True
        logging.info("Model loading ended!")

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
    
    # Test data evaluation
    fout = open(args.result.split('.csv')[0]+'_test.csv', 'w')
    fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\n')
    fam_out = open(args.result.split('.csv')[0]+'_family_test.csv', 'w')
    fam_out.write('Month\tNew\tFamily\tFNR\tCnt\n')
    stat_out = open(args.result.split('.csv')[0]+'_stat_test.csv', 'w')
    stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')

    # for i in range(1, 13):
    # args.test_start = args.test_end
    # test_start = (dt.datetime.strptime(args.test_start, '%Y-%m') + relativedelta(months=i-1)).strftime('%Y-%m')
    # test_end = (dt.datetime.strptime(args.test_end, '%Y-%m') + relativedelta(months=i-1)).strftime('%Y-%m')
    
    logging.info(f'Loading {args.data} Test dataset form {args.test_start} to {args.test_end}')
        
    X_test, y_test, all_test_family = load_data(args.data, args.test_start, args.test_end)
        
    # Test data evaluation
    evaluate_valid_test_data(args, classifier, args.test_end, X_test, y_test,\
                    all_test_family, cls_gpu, args.eval_multi, 'test', fout, fam_out, stat_out)

    logging.info(f'Run complete for Test.')
    #--------------------------------------------------------------------------------------------------

    """
    Now the following of the code is for pseudo-labeling and sample selection if needed
    """
    logging.info(f'Pseudo labeling for Test data Started.....')

    y_test_labels = y_test.reshape(-1, 1)
    df = pesudo_labeling(args=args, is_single_k_sm_fn=True, K=5, sm_fn='euclidean', ckpt_index=None, \
                    X_train_feat=X_train_feat, y_train_binary=y_train_binary, \
                        X_test_feat=X_test, y_test_binary=y_test_labels)

    logging.info(f'Test data pseudo label info: {df.head(10)}')
    logging.info(f'Pseudo labeling for Test data Ended.')
    logging.info(f'Sample selection Started.....')

    # Sort the dataframe based on 'Min_similarity' column and select the first 200 rows
    df_sorted['original_index'] = df.index
    df_sorted = df.sort_values(by='Min_value').head(200)
    logging.info(f'Selected top 200 samples based on minimum similarity:\n{df_sorted.head(10)}')
    selected_indices = df_sorted['original_index'].values
    selected_X_test_feat = X_test[selected_indices]
    selected_y_test_binary = y_test[selected_indices]
    logging.info(f'Selected X_test_feat shape:\n{selected_X_test_feat.shape}')

    # Merge the selected test samples with the training data
    X_train_feat = np.concatenate((X_train_feat, selected_X_test_feat), axis=0)
    y_train_binary = np.concatenate((y_train_binary, selected_y_test_binary), axis=0)
    logging.info(f'Updated X_train_feat shape: {X_train_feat.shape}')
    logging.info(f'Updated y_train_binary shape: {y_train_binary.shape}')

    #--------------------------------------------------------------------------------------------------
    
    """
    Now the following of the code is for active learning and sample selection if needed
    """


    """
    active learning loop

        uncertainty sampling
        
        implementation of coreset replay if we find it useful


        expanding the training set


    
    """
    
    # finish writing the result file
    fout.close()
    fam_out.close()
    # sample_out.close()
    stat_out.close()
    # sample_score_out.close()
    # sample_explanation.close()
    return


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f'time elapsed: {end - start} seconds')