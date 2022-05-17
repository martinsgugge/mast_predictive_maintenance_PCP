import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from array_manipulation import one_hot_decode, one_hot_encode
from metadata import Metadata
from plots import plot_confusion_matrix

def create_model(meta: Metadata):

    # print(f'X Shape[0,0]: {y_train[0].shape[0]}')
    # print(f'X Shape[0]: {y_train[0].shape[1]}')
    # print(f'Layer 1 Shape[0]: {meta.neurons_layer_one}')
    # print(f'X Shape[1]: {y_train.shape[1]}')
    # print(f'Y Shape: {y_train.shape}')
    model = keras.Sequential()
    # TODO for å få til bruk av GPU:
    # Rett og slett installere tensorflow-gpu?
    # https://www.tensorflow.org/guide/gpu
    # https://www.tensorflow.org/guide/keras/masking_and_padding
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
    model.add(keras.layers.GRU(meta.neurons_layer_one,
                                batch_input_shape=(meta.batch_size, meta.sequence_steps, meta.no_of_features),
                                return_sequences=True, stateful=meta.stateful))
    if meta.layers == 2:
        model.add(keras.layers.GRU(meta.neurons_layer_two, return_sequences=True, stateful=False))
    elif meta.layers == 3:
        model.add(keras.layers.GRU(meta.neurons_layer_two, return_sequences=True, stateful=False))
        model.add(keras.layers.GRU(meta.neurons_layer_three, return_sequences=True, stateful=False))

    model.add(keras.layers.Dense(meta.no_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)
    print(model.summary())

    return model


def train_model(model, X_train, X_test, y_train, y_test, meta: Metadata):
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    if os.path.exists(f'./GRU_models/{meta.model_name}{meta.checkpoint_path[1:]}'):
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'./GRU_models/{meta.model_name}{meta.checkpoint_path[1:]}/cp.ckpt',
            save_weights_only=True, verbose=1)
    else:
        os.makedirs(f'./GRU_models/{meta.model_name}{meta.checkpoint_path[1:]}')
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'./GRU_models/{meta.model_name}{meta.checkpoint_path[1:]}/cp.ckpt',
            save_weights_only=True, verbose=1)

    print(meta.checkpoint_path)
    history = model.fit(
        X_train, y_train,
        epochs=meta.number_of_epochs,
        steps_per_epoch=len(X_train)//meta.batch_size,
        batch_size=meta.batch_size,
        callbacks=[callback, cp_callback],
        verbose=1,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test)//meta.batch_size
    )
    if os.path.exists(f'./GRU_models/{meta.model_name}'):
        model.save(f'./GRU_models/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')
    else:
        os.makedirs(f'./GRU_models/{meta.model_name}')
        model.save(f'./GRU_models/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')

    plt.plot(history.history['loss'], label='Train')
    try:
        plt.plot(history.history['val_loss'], label='Validation')
    except KeyError as e:
        print(e)

    plt.legend()
    plt.savefig(f'./lossplots/{meta.checkpoint_path[len("./checkpoints/"):]}.png')
    meta.accuracy = history.history['accuracy']
    #plt.show()

    return model

def define_and_train_model(X_train, X_test, y_train, y_test, meta: Metadata):
    #https://github.com/christianversloot/machine-learning-articles/blob/main/build-an-lstm-model-with-tensorflow-and-keras.md
    print(f'meta is ', meta.train_run)
    if meta.train_run is None:
        model = create_model(meta)
    else:
        model = create_model(meta)

        try:
            model.load_weights(f'./GRU_models/{meta.model_name}{meta.previous_checkpoint_path[1:]}/cp.ckpt')
        except FileNotFoundError:
            model.load_weights(meta.previous_checkpoint_path + '/cp.ckpt')

    model = train_model(model, X_train, X_test, y_train, y_test, meta)

    return model

def test_model(model, X_test, y_test, meta: Metadata):

    #model.load_weights(meta.checkpoint_path+'/cp.ckpt')
    X_test = X_test[0:((len(X_test)//meta.batch_size)*meta.batch_size)]
    y_test = y_test[0:(round(len(y_test) // meta.batch_size) * meta.batch_size)]
    print(X_test.shape)
    #y_pred = model.predict(X_test)
    y_pred = model.predict(X_test, batch_size=meta.batch_size)
    print(f'y_pred shape = {y_pred.shape}')
    print(f'y_test shape = {y_test.shape}')
    shape_y_pred = y_pred.shape
    shape_y_test = y_test.shape

    y_pred = y_pred.reshape(shape_y_pred[0] * shape_y_pred[1], shape_y_pred[2])
    y_test = y_test.reshape(shape_y_test[0] * shape_y_test[1], shape_y_test[2])
    #print(f'y_pred reshaped = {len(y_pred)}')
    print(f'y_pred shape = {y_pred.shape}')
    print(f'y_test shape = {y_test.shape}')


    y_pred_dec = one_hot_decode(y_pred)
    y_test_dec = one_hot_decode(y_test)
    # print(f'y_pred shape = {y_pred_dec.shape}')
    # print(f'y_test shape = {y_test_dec.shape}')

    tot_accuracy = accuracy_score(y_test_dec, y_pred_dec)
    print("Total Accuracy: " + str(tot_accuracy))

    #meta.accuracy =
    lstm_cm = confusion_matrix(y_test_dec, y_pred_dec)
    lstm_cm_norm = lstm_cm.astype('float') / lstm_cm.sum(axis=1)[:, np.newaxis]
    lstm_cm_acc = lstm_cm_norm.diagonal()
    #REDO
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

        #plt.show(block=True)
        if os.path.exists(f'./cfsmatplots/GRU/{meta.model_name}'):
            plot_confusion_matrix(lstm_cm_norm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/GRU/{meta.model_name}/in percent {meta.checkpoint_path[len("./checkpoints/"):]}')
        else:
            os.makedirs(f'./cfsmatplots/GRU/{meta.model_name}')
            plot_confusion_matrix(lstm_cm_norm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/GRU/{meta.model_name}/in percent {meta.checkpoint_path[len("./checkpoints/"):]}')
        if os.path.exists(f'./cfsmatplots/GRU/{meta.model_name}'):
            plot_confusion_matrix(lstm_cm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/GRU/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')
        else:
            os.makedirs(f'./cfsmatplots/GRU/{meta.model_name}')
            plot_confusion_matrix(lstm_cm, ['Normal', 'One week from fail', 'One day from fail'],
                                  f'./cfsmatplots/GRU/{meta.model_name}/{meta.checkpoint_path[len("./checkpoints/"):]}')
    except IndexError:
        pass

    return lstm_cm_acc
