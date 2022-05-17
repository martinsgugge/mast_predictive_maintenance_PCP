import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from array_manipulation import one_hot_decode
from metadata import MetadataSVM
from plots import plot_confusion_matrix


def train_svm(x, y, meta: MetadataSVM):
    print(type(x))
    print(x.shape)
    print(type(y))
    print(y.shape)
    if meta.method == 'linear':
        clf = LinearSVC(loss='hinge', C=meta.c, verbose=True, class_weight='balanced', max_iter=10000)

    elif meta.method == 'poly':
        clf = SVC(kernel=meta.method, degree=meta.degree, gamma=meta.gamma, coef0=meta.coef0, C=meta.c, verbose=True,
                  cache_size=1000, class_weight='balanced', tol=1e-2, max_iter=10000)


    elif meta.method == 'rbf':

        clf = SVC(kernel="rbf", gamma='scale', C=meta.c, verbose=True, cache_size=500, class_weight='balanced',
                  tol=1e-2, max_iter=10000)




    elif meta.method == 'GridSearchCV rbf':

        # Set the parameters by cross-validation

        tuned_parameters = [

            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1e-1, 1]}

        ]

        score = 'balanced_accuracy'

        clf = GridSearchCV(SVC(cache_size=1000, verbose=True, class_weight='balanced', tol=1e-2, max_iter=10000),

                           tuned_parameters, scoring=score)

    elif meta.method == 'HalvingGridSearchCV rbf':

        # Set the parameters by cross-validation

        tuned_parameters = [

            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1e-1, 1]}
        ]

        score = 'balanced_accuracy'

        clf = HalvingGridSearchCV(SVC(cache_size=1000, verbose=True, class_weight='balanced', tol=1e-2, max_iter=10000),
                           tuned_parameters, scoring=score)


    elif meta.method == 'GridSearchCV sigmoid':

        # Set the parameters by cross-validation

        tuned_parameters = [

            {"kernel": ["sigmoid"], "gamma": [1e-3, 1e-4], "C": [1e-1, 1]}

        ]

        score = 'balanced_accuracy'

        clf = GridSearchCV(SVC(cache_size=1000, verbose=True, class_weight='balanced', tol=1e-2, max_iter=10000),

                           tuned_parameters, scoring=score)

    elif meta.method == 'sigmoid':

        # Set the parameters by cross-validation


        score = 'balanced_accuracy'

        clf = SVC(kernel='sigmoid', cache_size=1000, verbose=True, C=1, gamma=1, class_weight='balanced', tol=1e-2, max_iter=10000)

    elif meta.method == 'GridSearchCV poly':

        # Set the parameters by cross-validation

        tuned_parameters = [
            {"kernel": ["poly"], "degree": [1, 2], "gamma": [1e-3, 1e-4],
             "C": [1e-1, 1]}

        ]

        score = 'balanced_accuracy'

        clf = GridSearchCV(SVC(cache_size=1000, verbose=True, class_weight='balanced', tol=1e-2, max_iter=10000),
                           tuned_parameters, scoring=score)
    elif meta.method == 'GridSearchCV linear':
        # Set the parameters by cross-validation
        tuned_parameters = [
            {"kernel": ["linear"], "C": [1e-1, 1]},
        ]
        score = 'balanced_accuracy'
        clf = GridSearchCV(SVC(cache_size=1000, verbose=True, class_weight='balanced', tol=1e-2, max_iter=10000),
                           tuned_parameters, scoring=score)


    elif meta.method == 'HalvingGridSearchCV linear':

        # Set the parameters by cross-validation

        tuned_parameters = [

            {"kernel": ["linear"], "C": [1e-1, 1]}
        ]

        score = 'balanced_accuracy'
        #clf = LinearSVC(loss='hinge', C=meta.c, verbose=True, class_weight='balanced', max_iter=10000)
        clf = HalvingGridSearchCV(
            LinearSVC(loss='hinge', verbose=True, class_weight='balanced', max_iter=10000),
            tuned_parameters, scoring=score
                                  )
    elif meta.method == 'rbf':
        clf = SVC(kernel="rbf", gamma=meta.gamma, C=meta.c, verbose=True, cache_size=1000, class_weight='balanced',
                  tol=1e-2, max_iter=10000)

    clf.fit(x, y)
    if meta.method[:12] == 'GridSearchCV' or meta.method[:19] == 'HalvingGridSearchCV':
        meta.c = clf.best_params_.get('C')
        meta.gamma = clf.best_params_.get('gamma')
        meta.method = clf.best_params_.get('kernel')
        meta.coef0 = clf.best_params_.get('coef0')

    return clf


def test_svm(model, x_test, y_test, meta):
    y_pred = model.predict(x_test)

    # y_pred_dec = one_hot_decode(y_pred)
    # y_test_dec = one_hot_decode(y_test)
    tot_accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    meta.accuracy = tot_accuracy
    print(f'total accuracy: {tot_accuracy}')
    print(f'confsion matrix: {confusion_matrix_}')

    lstm_cm = confusion_matrix(y_test, y_pred)
    lstm_cm_norm = lstm_cm.astype('float') / lstm_cm.sum(axis=1)[:, np.newaxis]
    lstm_cm_acc = lstm_cm_norm.diagonal()
    # REDO
    print(lstm_cm)
    print(lstm_cm.sum(axis=1))
    try:
        meta.test_accuracy = f'total accuracy={tot_accuracy}; Normal={lstm_cm_acc[0]}; ' \
                             f'One week from fail={lstm_cm_acc[1]};' \
                             f'one day from fail={lstm_cm_acc[2]}'
        print("Accuracy Label 0 (Normal run): ", lstm_cm_acc[0])
        print("Accuracy Label 1 (< 1 week from fail: ", lstm_cm_acc[1])
        meta.accuracy = f'total accuracy={tot_accuracy}; Normal={lstm_cm_acc[0]}; ' \
                        f'One week from fail={lstm_cm_acc[1]};' \
                        f'one day from fail={lstm_cm_acc[2]}'
    except IndexError:
        pass

    try:
        print("Accuracy Label 2 (< 1 day from fail): ", lstm_cm_acc[2])
    except IndexError as e:
        print(e)
        print('Did not find any datapoints where <1 day from failure')
    # print("Accuracy Label 3 (< 24h from fail): ", lstm_cm_acc[3])
    # print("Accuracy Label 4 (< 1h from fail): ", lstm_cm_acc[4])
    failmatrix1 = np.zeros((3, 3))
    try:
        for i in range(len(lstm_cm[0])):
            failmatrix1[0, i] = lstm_cm[0, i]
            failmatrix1[1, i] = lstm_cm[1, i]
            failmatrix1[2, i] = lstm_cm[2, i]
        failmatrix2 = np.zeros((3, 3))
        for i in range(3):
            failmatrix2[i, 0] = failmatrix1[i, 0]
            failmatrix2[i, 1] = failmatrix1[i, 1]
            failmatrix2[i, 2] = failmatrix1[i, 2]
        lstm_fail_norm = failmatrix2.astype('float') / failmatrix2.sum(axis=1)[:, np.newaxis]
        lstm_fail_acc = lstm_fail_norm.diagonal()
        print("Fail mode accuracy: ", lstm_fail_acc[2])
        print("Fail norm: ", lstm_fail_norm[2])
        lstm_disp = ConfusionMatrixDisplay(confusion_matrix=lstm_cm_norm,
                                           display_labels=['Normal', 'One week from fail', 'One day from fail'])

        lstm_disp.plot()
        lstm_cm_norm = lstm_cm_norm.round(decimals=3)

        # plt.show(block=True)
        if os.path.exists(f'./cfsmatplots/SVM/{meta.model_name}'):
            plot_confusion_matrix(lstm_cm_norm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/SVM/{meta.model_name}/in percent {meta.checkpoint_path[len("./checkpoints/"):]}')
        else:
            os.makedirs(f'./cfsmatplots/SVM/{meta.model_name}')
            plot_confusion_matrix(lstm_cm_norm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/SVM/{meta.model_name}/in percent {meta.checkpoint_path[len("./checkpoints/"):]}')

        if os.path.exists(f'./cfsmatplots/SVM/{meta.model_name}'):
            plot_confusion_matrix(lstm_cm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/SVM/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')
        else:
            os.makedirs(f'./cfsmatplots/SVM/{meta.model_name}')
            plot_confusion_matrix(lstm_cm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/SVM/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')

    except IndexError:
        pass

    return lstm_cm
