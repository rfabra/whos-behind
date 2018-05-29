import numpy as np
from datasets import uci
import utils.queries as Queries
from utils import r_classifiers as R
from utils.evaluation import csv_confusion_matrix, csv_report

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

import json
import os


def launch(clf_names,
           compute_features=False,
           metadatasets_path='files_test/metadatasets/',
           oracles_path='files_test/oracles/',
           predictions_path='files_test/predictions/',
           results_path='files_test/evaluation/',
           r_path='files_test/R_in/',
           kappa=True,
           max_measure=False,
           metamodel=False):
    """Launch the experiments.

    Args:
        clf_names (list): the names of classifiers (both oracle and surrogate are the same).
        metadatasets_path (str): path to store metadatasets file.
        oracles_path (str): path to store oracles files (surrogate and instances).
        predictions_path (str) path to store predictions of surrogate models over surrogate datasets.
        results_path (str): path to store the evaluation results.
        r_path (str): path to store R input files for predictions.
        kappa (bool): indicates wether to use kappa as feature.
        max_measure (bool): indicates if must evaluate the model with the maximum kappa approach.
        metamodel (bool): indicates if must evaluate the model with the meta-model (SVM) approach."""

    # Load UCI datasets
    dss = uci.load_raw_datasets('datasets/UCI/')

    vect = DictVectorizer()
    imputer = Imputer()
    scaler = StandardScaler()


    # initialize an empty list of features (for meta-model) and oracle labels
    feats = []
    oracle_labels = [None for _ in range(len(dss) * len(clf_names))]
    if kappa:
        feats = [[None for _ in range(len(clf_names))] for _ in range(len(dss) * len(clf_names))]

    i_oracle = 0
    if compute_features:
        # iterate each dataset
        for ids, ds in enumerate(dss):
            print('----------------------------------------------')
            print('Processing {} ({}): {} features, {} classes...'.format(ids + 1, ds[3], len(ds[0][0]), len(set(ds[1]))))
            print('----------------------------------------------\n')


            # Training set, training labels, feature meta-information and dataset name
            o_train, o_train_l, o_meta, o_name = ds[0], ds[1], ds[2], ds[3]
            # Remove '.arff'
            o_name = o_name[:-5]

            # Surrogate dataset generation
            s_train = Queries.uniform(o_train, 100*len(o_meta), o_meta)

            o_train = [dict(zip([att for att, _ in o_meta], d)) for d in o_train]
            o_train = vect.fit_transform(o_train).toarray()
            o_train = imputer.fit_transform(o_train)
            o_train = scaler.fit_transform(o_train)

            s_train = vect.transform(s_train).toarray()
            s_train = imputer.transform(s_train)
            s_train = scaler.transform(s_train)

            # iterate each oracle
            for i_clf, o_clf in enumerate(clf_names):
                # Prepare training, training labels, test and predictions file names
                # for the surrogate dataset
                tr_file   =  r_path + R.prepare_file(o_name, clf_names[i_clf], None, 'otrain')
                tr_l_file =  r_path + R.prepare_file(o_name, clf_names[i_clf], None, 'otrain_l')
                ts_file   =  r_path + R.prepare_file(o_name, clf_names[i_clf], None, 'strain')
                pred_file = oracles_path + '{}_{}_SD.json'.format(o_name, clf_names[i_clf])

                # if not computed previously, do it
                if not os.path.isfile(pred_file):
                    s_train_l = R.model(clf_names[i_clf], np.array(o_train), o_train_l,
                                        np.array(s_train), tr_file, tr_l_file, ts_file, pred_file)
                # otherwise, load previous computations
                else:
                    f = open(pred_file, 'r')
                    s_train_l = f.readlines()
                    s_train_l = list(map(lambda t: t.strip('\n'), s_train_l))
                    f.close()

                # iterate each surrogate model
                for j_clf, s_clf in enumerate(clf_names):
                    # Train and evaluate surrogate classifiers
                    # with 5-fold stratified cross-validation
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

                    # True and predicted labels
                    true_l = []
                    pred_l = []
                    n_fold = 1
                    try:
                        # Start cross-validation
                        for tr, ts in kf.split(s_train, s_train_l):
                            print('ORIGINAL: {} --- SURROGATE: {}'.format(clf_names[i_clf], clf_names[j_clf]))
                            print('FOLD: {}/5'.format(n_fold))
                            # Training and validation (test) sets
                            train = [s_train[i] for i in tr]
                            train_l = [s_train_l[i] for i in tr]

                            test = [s_train[i] for i in ts]
                            test_l = [s_train_l[i] for i in ts]

                            # Prepare files for predictions
                            tr_file   = r_path + R.prepare_file(o_name, clf_names[i_clf], clf_names[j_clf], 'train')
                            tr_l_file = r_path + R.prepare_file(o_name, clf_names[i_clf], clf_names[j_clf], 'train_l')
                            ts_file   = r_path + R.prepare_file(o_name, clf_names[i_clf], clf_names[j_clf], 'test')
                            pred_file = predictions_path + R.prepare_file(o_name, clf_names[i_clf], clf_names[j_clf], 'pred_{}'.format(n_fold))

                            # If prediction was not made previosly, do it
                            if not os.path.isfile(pred_file):
                                pred = R.model(clf_names[j_clf], np.array(train), train_l,
                                               np.array(test), tr_file, tr_l_file, ts_file, pred_file)
                            # Otherwise, load existing predictions
                            else:
                                f = open(pred_file, 'r')
                                pred = f.readlines()
                                pred = list(map(lambda t: t.strip('\n'), pred))
                                f.close()

                            # Add new predictions to the list
                            pred_l += list(pred)

                            # Add true labels to the list
                            true_l += test_l
                            n_fold += 1
                    except Exception as e:
                        print(str(e))
                        continue

                    # Save the results of the cross-validation procedure
                    # If they already existed, load them
                    pred_file = predictions_path + '{}_{}_{}_CV.csv'.format(o_name, clf_names[i_clf], clf_names[j_clf])
                    if not os.path.isfile(pred_file):
                        # Save predicitions of the surrogate model
                        f = open(pred_file, 'w')
                        f.write(','.join(pred_l))
                        f.close()
                    else:
                        f = open(pred_file, 'r')
                        pred_l = f.read().split(',')
                        f.close()


                    # Compute kappa score between oracle's output (labels of surrogate dataset)
                    # and predicted labels for the current surrogate model
                    if kappa:
                        kap_score = cohen_kappa_score(true_l, pred_l)
                        feats[i_oracle][j_clf] = kap_score

                # Save oracle instance as kappa features
                oracle_labels[i_oracle] = clf_names[i_clf]
                path = oracles_path + '{}_{}_instance.json'.format(o_name, clf_names[i_clf])
                if not os.path.isfile(path):
                    f = open(path, 'w')
                    json.dump((feats[i_oracle], oracle_labels[i_oracle]), f)
                    f.close()
                # Load oracle instances, if they were previously computed
                else:
                    f = open(path, 'r')
                    feats[i_oracle], oracle_labels[i_oracle] = json.load(f)
                    f.close()
                i_oracle += 1


            # Save metadataset: all computed instances (oracles) up to this point
            path = metadatasets_path + '{}_metadataset.json'.format(ids+1)
            if not os.path.isfile(path):
                f = open(path, 'w')
                json.dump((feats, oracle_labels), f)
                f.close()
            else:
                f = open(path, 'r')
                feats, oracle_labels = json.load(f)
                f.close()

    #########################
    # Family Identification #
    #########################

    # Load the metadaset (oracles as kappa) and labels
    path = metadatasets_path + '25_metadataset.json'
    f = open(path, 'r')
    feats, oracle_labels = json.load(f)
    f.close()

    print("Number of instances    : " + str(len(feats)))
    print("Feats per instance: " + str(len(feats[0])))
    print("Number of oracle labels: " + str(len(oracle_labels)))


    true_l = []
    pred_l = []

    # Meta-model
    algorithm_selector = SVC(kernel='rbf', C=100, gamma=0.01)

    # Number of surrogate surrogate models
    n_surr = len(clf_names)
    # Number of oracle instances to take per surrogate model
    n_ts_samples = 1

    # Cross-validation evaluation
    for n in range(0, len(feats), n_ts_samples*n_surr):
        # Train and test labels
        pred = None
        test =  feats[n:n+n_ts_samples*n_surr]
        test_l = oracle_labels[n:n+n_ts_samples*n_surr]

        train = feats[:n] + feats[n + n_ts_samples*n_surr:]
        train_l = oracle_labels[:n] + oracle_labels[n + n_ts_samples*n_surr:]

        # Remove instances with NaN (those for which the kappa could not be estimated)
        f_o_tr = [(f, l) for (f, l) in zip(train, train_l) if None not in f and not np.isnan(f).any()]
        train = [f for (f, l) in f_o_tr]
        train_l = [l for (f, l) in f_o_tr]

        # Same for test sets
        f_o_ts = [(f, l) for (f, l) in zip(test, test_l) if None not in f and not np.isnan(f).any()]
        test = [f for (f, l) in f_o_ts]
        test_l = [l for (f, l) in f_o_ts]

        true_l += test_l

        # Meta-model approach
        if metamodel:
            sc = StandardScaler()
            train = sc.fit_transform(train)
            test = sc.transform(test)

            algorithm_selector.fit(train, train_l)
            pred = list(algorithm_selector.predict(test))

        # Maximum kappa approach
        if max_measure:
            pred = [sorted(list(zip(clf_names, ts)), key=lambda t: t[1], reverse=True)[0][0] for ts in test]

        pred_l += pred

    # Write evaluation results (confusion matrix and accuracy)
    out_conf = ''
    #out_report = ''

    if max_measure:
        out_conf = results_path + '/MAX_confmat.csv'
        #out_report = results_path + '/MAX_report.csv'

    if metamodel:
        out_conf = results_path + '/SVM_confmat.csv'
        #out_report = results_path + '/SVM_report.csv'

    f_res = open(out_conf, 'w')
    f_res.write(csv_confusion_matrix(confusion_matrix(true_l, pred_l, clf_names), clf_names) + '\n')
    f_res.close()

    #f_res = open(out_report, 'w')
    #f_res.write(csv_report(classification_report(true_l, pred_l, clf_names)))
    #f_res.write('Accuracy: ' + str(round(accuracy_score(true_l, pred_l),2)) + '\n')
    #f_res.close()

    # Print evaluation results (confusion matrix and accuracy)
    print(classification_report(true_l, pred_l, labels=clf_names))
    print(str(confusion_matrix(true_l, pred_l, labels=clf_names)))
    print('Accuracy, ' + str(accuracy_score(true_l, pred_l)))


# Oracles/surrogate model names
clf_names = [
    "RDA",          # OK
    "RF",           # OK
    "C5.0",         # OK
    "SVM_GAUSSIAN", # OK
    "MLP",          # OK
    "NAIVE BAYES",  # OK
    "KNN",          # OK
    "GLMNET",       # OK
    "SIMPLS",       # OK
    "PMR",          # OK
    "MARS"          # OK
]

# Launch the experiments
launch(clf_names,
       compute_features=True,
       metadatasets_path='files_test/metadatasets/',
       oracles_path='files_test/oracles/',
       predictions_path='files_test/predictions/',
       results_path='files_test/evaluation/',
       r_path='files_test/R_in/',
       kappa=True,
       max_measure=False,
       metamodel=True)
