#!/usr/bin/python2

import cv2
import numpy as np

import os
from six.moves import cPickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

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

def process(cap):
    scale_height = 128
    scale_width = 160
    target_fps = 8
    n_plot = 40

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25

    nt = 10  # number of timesteps used for sequences in training
    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_ped_train_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_ped_train_model.json')

    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(weights_file)

    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)

    print(fps)

    Xcur = []

    mses = []

    #plt.ylim(0, 0.01)
    plt.ion()

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
            X_test = np.array([np.array(Xcur).astype(np.float32) / 255])
            X_hat = test_model.predict(X_test)

            mses += list(np.mean(((X_test[:, 1:] - X_hat[:, 1:])**2)[0], axis=(1,2,3)))

            plt.plot(mses)
            plt.pause(0.05)

            #aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
            #plt.figure(figsize = (nt, 2*aspect_ratio))
            #gs = gridspec.GridSpec(2, nt)
            #gs.update(wspace=0., hspace=0.)
            #for t in range(nt):
            #    plt.subplot(gs[t])
            #    plt.imshow(X_test[0,t], interpolation='none')
            #    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            #    if t==0: plt.ylabel('Actual', fontsize=10)

            #    plt.subplot(gs[t + nt])
            #    plt.imshow(X_hat[0,t], interpolation='none')
            #    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            #    if t==0: plt.ylabel('Predicted', fontsize=10)

            #plt.show()

            Xcur = []

        k = cv2.waitKey(1) & 0xff
        if k == 32:
            k = cv2.waitKey() & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Done!")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    cap = cv2.VideoCapture('ped_train.avi')
    process(cap)
