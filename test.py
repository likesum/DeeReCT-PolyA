#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import sys, os
import argparse
from model import Net

############ Parameters ############
PATCH_SIZE = 10
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
############ **************** ############

def pad_dataset(dataset, labels=[]):
    ''' Pad sequences to (length + 2*DEPTH - 2) wtih 0.25 '''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    if labels != []:
        labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels

def get_accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def read_file(filename):
    data = []
    alphabet = np.array(['A', 'G', 'T', 'C'])
    with open(filename, 'r') as f:
        for line in f:
            line = list(line.strip('\n'))
            seq = np.array(line, dtype = '|U1').reshape(-1, 1)
            seq_data = (seq == alphabet).astype(np.float32)
            data.append(seq_data)
        data = np.stack(data).reshape([-1, 206, 1, 4])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data file, can be .txt file containing sequeces or .npz file containing one-hot encoded sequences')
    parser.add_argument('wts', help='Trained model (.npz file)')
    parser.add_argument('--out', default=None, help='Save predictions to (.txt file)')
    opts = parser.parse_args()
    
    # Build model
    sess = tf.Session()
    model = Net()

    # Load trained model
    model.load_weights(opts.wts, sess)
    print('\n########################')
    print('Model loaded from %s.'%opts.wts)

    # Load and pad data
    if opts.data.endswith('.npz'):
        data = np.load(opts.data)
        dataset = data['test_dataset']
        labels = data['test_labels'] if 'test_labels' in data else []
    elif opts.data.endswith('.txt'):
        dataset = read_file(opts.data)
        labels = []
    dataset, labels = pad_dataset(dataset, labels)
    print("Read %d sequences and %d labels from %s."%(len(dataset), len(labels), opts.data))

    predictions = model.get_prediction(sess, dataset, istrain=False)
    if labels != []:
        accuracy = get_accuracy(predictions, labels)
        print('\nTest accuracy: %.1f%%'%accuracy)

    if opts.out is not None:
        predictions = np.argmax(predictions, 1)
        predictions = np.where(predictions, 'F', 'T')
        with open(opts.out, 'w') as f:
            for pred in predictions:
                f.write(pred+'\n')
        print('\nPredictions wrote to %s.'%opts.out)









