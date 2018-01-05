#!/usr/bin/python2

import cv2
import numpy as np

import os
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

def aspect_fit(where, what):
    aspect = float(what[0]) / what[1]
    if aspect >= 1.0:
        rect = [0.0, 0.0, float(where[0]), float(where[0]) / aspect]
        if rect[3] <= where[1]:
            rect[1] = float(where[1] - rect[3]) / 2.0
        else:
            rect[3] = float(where[1])
            rect[2] = float(where[1]) * aspect
            rect[0] = float(where[0] - rect[2]) / 2.0
    else:
        rect = [0.0, 0.0, float(where[1]) * aspect, float(where[1])]
        if rect[2] <= where[0]:
            rect[0] = float(where[0] - rect[2]) / 2.0
        else:
            rect[2] = float(where[0])
            rect[3] = float(where[0]) / aspect
            rect[1] = float(where[1] - rect[3]) / 2.0

    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

def resize_fit(frame, desired_size):
    r = aspect_fit(desired_size, (frame.shape[1], frame.shape[0]))

    frame = cv2.resize(frame, (r[2], r[3]), interpolation = cv2.INTER_AREA)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(frame, r[1], desired_size[1] - r[1] - r[3], r[0], desired_size[0] - r[0] - r[2], cv2.BORDER_CONSTANT, value=color)

def process(cap, fname):
    scale_height = 128
    scale_width = 160
    target_fps = 8

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25

    nt = 10  # number of timesteps used for sequences in training

    print(fps)

    X = []
    Xcur = []

    i = 0
    j = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        j += 1
        if (j < fps / target_fps):
            continue
        j = 0
        i += 1

        frame = resize_fit(frame, (scale_width, scale_height))

        cv2.imshow('frame', frame)

        Xcur.append(frame)

        if len(Xcur) >= nt:
            X.append(np.array(Xcur).astype(np.float32) / 255)
            Xcur = []

        k = cv2.waitKey(1) & 0xff
        if k == 32:
            k = cv2.waitKey() & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    X = np.array(X)
    print(X.shape)
    Y = np.zeros(X.shape[0], np.float32)
    print(Y.shape)

    assert(K.image_data_format() != 'channels_first')

    save_model = True  # if weights will be saved
    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_' + fname + '_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_' + fname + '_model.json')

    # Training parameters
    nb_epoch = 2
    batch_size = 4

    # Model parameters
    n_channels, im_height, im_width = (3, 128, 160)
    input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
    time_loss_weights[0] = 0


    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)
    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
    errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
        callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

    model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, validation_split=0.2, verbose=1, callbacks=callbacks)

    if save_model:
        json_string = model.to_json()
        with open(json_file, "w") as f:
            f.write(json_string)


if __name__ == "__main__":
    fn = 'ped_train.avi'
    cap = cv2.VideoCapture(fn)
    process(cap, fn.split(".")[0])
