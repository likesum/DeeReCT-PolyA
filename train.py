#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import sys, os
import argparse
from model import Net

############ Parameters ############
BATCH_SIZE = 64
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 64
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NUM_EPOCHS = 200
############ **************** ############

def pad_dataset(dataset, labels):
    ''' Pad sequences to (length + 2*DEPTH - 2) wtih 0.25 '''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels

def get_accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data npz files')
    parser.add_argument('--out', default=None, help='Save model weights to (.npz file)')
    parser.add_argument('--hparam', default=None, help='Hyper-params (.npz file), default using random hyper-params')
    parser.add_argument('--pretrained', default=None, help='Fine-tuning a pretrained model')

    opts = parser.parse_args()

    # Load and pad data
    data = np.load(opts.data)
    train_data, train_labels = pad_dataset(data['train_dataset'], data['train_labels'])
    valid_data, valid_labels = pad_dataset(data['valid_dataset'], data['valid_labels'])

    # Build model and trainning graph
    hyper_dict = dict(np.load(opts.hparam)) if opts.hparam is not None else None
    model = Net(hyper_dict)
    model.build_training_graph()

    # Kick off training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if opts.pretrained is not None:
        model.load_weights(opts.pretrained, sess)
        print('Fine-tuning the pre-trained model %s'%opts.pretrained)
    else:
        print('Initialized')
    pred = model.get_prediction(sess, valid_data, False)
    print('Validation accuracy at the beginning: %.1f%%' % get_accuracy(pred, valid_labels))
    
    train_resuts, valid_results = [], []
    save_weights = []
    for epoch in range(NUM_EPOCHS):
        permutation = np.random.permutation(train_labels.shape[0])
        shuffled_dataset = train_data[permutation, :, :]
        shuffled_labels = train_labels[permutation, :]

        accuracy = 0.
        for step in range(shuffled_labels.shape[0] // BATCH_SIZE):
            offset = step * BATCH_SIZE
            batch_data = train_data[offset:(offset+BATCH_SIZE), :, :, :]
            batch_labels = train_labels[offset:(offset+BATCH_SIZE), :]
            fd = {
                model.dataset: batch_data, 
                model.labels: batch_labels,
                model.istrain: True
            }
            _, pred = sess.run([model.optimizeOp, model.prediction], feed_dict=fd)
            accuracy += get_accuracy(pred, batch_labels)
            sess.run(model.stepOp)
        
        accuracy = accuracy / (shuffled_labels.shape[0] // BATCH_SIZE)
        train_resuts.append(accuracy)
        pred = model.get_prediction(sess, valid_data, False)
        valid_results.append(get_accuracy(pred, valid_labels))

        print('Training accuracy at epoch %d: %.1f%%' % (epoch, train_resuts[-1]))
        print('Validation accuracy: %.1f%%' % valid_results[-1])

        # Early stopping based on validation results
        if epoch > 10 and valid_results[-11] > max(valid_results[-10:]):
            train_resuts = train_resuts[:-10]
            valid_results = valid_results[:-10]
            print('\n\n########################\nFinal result:')
            print("Best valid epoch: %d"%(len(train_resuts)-1))
            print("Training accuracy: %.2f%%"%train_resuts[-1])
            print("Validation accuracy: %.2f%%"%valid_results[-1])
            if opts.out is not None:
                np.savez(opts.out, **save_weights[0])
            break

        # Model saving
        sw = sess.run(model.weights)
        save_weights.append(sw)
        if epoch > 10:
            save_weights.pop(0)

        if epoch == NUM_EPOCHS-1:
            print('\n\n########################\nFinal result:')
            print("Best valid epoch: %d"%(len(train_resuts)-1))
            print("Training accuracy: %.2f%%"%train_resuts[-1])
            print("Validation accuracy: %.2f%%"%valid_results[-1])
            if opts.out is not None:
                np.savez(opts.out, **save_weights[-1])









