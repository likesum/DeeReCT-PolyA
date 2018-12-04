#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import sys, os
from decimal import Decimal
from hm_prep import *

############ Model Selection ############
POS_PATH = 'data/human/dragon_polyA_data/positive5fold/'
NEG_PATH = 'data/human/dragon_polyA_data/negatives5fold/'
# POS_PATH = 'human_data/omni_polyA_data/positive/'
# NEG_PATH = 'human_data/omni_polyA_data/negative/'
BATCH_SIZE = 64
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 64
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NUM_EPOCHS = 200
NUM_FOLDS = 5
HYPER_DICT = None
############ **************** ############

tf.app.flags.DEFINE_string(
    'train_dir', None,
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer(
    'training_job_index', 0,
    'index of training result for logging')

tf.app.flags.DEFINE_string(
    'training_result_dir', None,
    'The file which the training result is written to')

FLAGS = tf.app.flags.FLAGS


def pad_dataset(dataset, labels):
    ''' Change dataset height to height + 2*DEPTH - 2'''
    new_dataset = np.ones([dataset.shape[0], dataset.shape[1]+2*PATCH_SIZE-2, dataset.shape[2], dataset.shape[3]], dtype = np.float32) * 0.25
    new_dataset[:, PATCH_SIZE-1:-(PATCH_SIZE-1), :, :] = dataset
    labels = (np.arange(NUM_LABELS) == labels[:,None]).astype(np.float32)
    return new_dataset, labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def gen_hyper_dict(hyper_dict=None):
    def rand_log(a, b):
        x = np.random.sample()
        return 10.0 ** ((np.log10(b) - np.log10(a)) * x + np.log10(a))

    def rand_sqrt(a, b):
        x = np.random.sample()
        return (b - a) * np.sqrt(x) + a

    if hyper_dict is None:
        hyper_dict = {
            'tf_learning_rate': rand_log(.0005, .05),
            'tf_momentum': rand_sqrt(.95, .99),
            'tf_motif_init_weight': rand_log(1e-2, 10),
            'tf_fc_init_weight': rand_log(1e-2, 10),
            'tf_motif_weight_decay': rand_log(1e-5, 1e-3),
            'tf_fc_weight_decay': rand_log(1e-5, 1e-3),
            'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
            'tf_ngroups': np.random.choice([2,4,8])
        }
    # for k, v in hyper_dict.items():
    #     print("%s: %.2e"%(k, Decimal(v)))
    # print()
    return hyper_dict


# Disable print
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore print
def enable_print():
    sys.stdout = sys.__stdout__


def train(dataset, hyper_dict):
    graph = tf.Graph()

    with graph.as_default():

        # Load hyper-params
        tf_learning_rate = hyper_dict['tf_learning_rate']
        tf_momentum = hyper_dict['tf_momentum']
        tf_motif_init_weight = hyper_dict['tf_motif_init_weight']
        tf_fc_init_weight = hyper_dict['tf_fc_init_weight']
        tf_motif_weight_decay = hyper_dict['tf_motif_weight_decay']
        tf_fc_weight_decay = hyper_dict['tf_fc_weight_decay']
        tf_keep_prob = hyper_dict['tf_keep_prob']
        tf_ngroups = hyper_dict['tf_ngroups']

        # Input data.
        tf_train_dataset = tf.placeholder(
          tf.float32, shape=(BATCH_SIZE, SEQ_LEN, 1, NUM_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_train_valid_dataset = tf.constant(dataset['train_dataset'])
        tf_valid_dataset = tf.constant(dataset['valid_dataset'])
        tf_test_dataset = tf.constant(dataset['test_dataset'])
        tf_motif_test_dataset = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            tf_motif_test_dataset[motif] = tf.constant(dataset['motif_dataset'][motif]['test_dataset'])

        # Variables.
        conv_weights = tf.Variable(tf.truncated_normal(
          [PATCH_SIZE, 1, NUM_CHANNELS, DEPTH], stddev=tf_motif_init_weight))
        conv_biases = tf.Variable(tf.zeros([DEPTH]))
        layer1_weights = tf.Variable(tf.truncated_normal(
          [21*DEPTH, NUM_HIDDEN], stddev=tf_fc_init_weight))
        layer1_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [NUM_HIDDEN, NUM_LABELS], stddev=tf_fc_init_weight))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

        # Store Variables
        weights = {}
        weights['conv_weights'] = conv_weights
        weights['conv_biases'] = conv_biases
        weights['layer1_weights'] = layer1_weights
        weights['layer1_biases'] = layer1_biases
        weights['layer2_weights'] = layer2_weights
        weights['layer2_biases'] = layer2_biases

        # Model.
        def model(data, drop=True):
            conv = tf.nn.conv2d(data, conv_weights, [1, 1, 1, 1], padding='VALID')
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH//tf_ngroups, tf_ngroups])
            mu, var = tf.nn.moments(conv, [1, 2, 3], keep_dims=True)
            conv = (conv - mu) / tf.sqrt(var + 1e-12)
            conv = tf.reshape(conv, [-1, 215, 1, DEPTH])
            hidden = tf.nn.relu(conv + conv_biases)
            hidden = tf.nn.max_pool(hidden, [1, 10, 1, 1], [1, 10, 1, 1], padding = 'VALID')
            shape = hidden.get_shape().as_list()
            motif_score = tf.reshape(hidden, [shape[0], shape[1]*DEPTH])
            if drop:
                hidden_nodes = tf.nn.dropout(tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases), tf_keep_prob)
            else:
                hidden_nodes = tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases)
            return tf.matmul(hidden_nodes, layer2_weights) + layer2_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + tf_fc_weight_decay*(tf.nn.l2_loss(layer1_weights)+tf.nn.l2_loss(layer2_weights)) + tf_motif_weight_decay*tf.nn.l2_loss(conv_weights)

        # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(tf_learning_rate, global_step, 3000, 0.96)
        optimizer = tf.train.MomentumOptimizer(learning_rate, tf_momentum).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(model(tf_train_valid_dataset, drop=False))
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, drop=False))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, drop=False))
        motif_test_prediction = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            motif_test_prediction[motif] = tf.nn.softmax(model(tf_motif_test_dataset[motif], drop=False))


    # Kick off training
    train_resuts = []
    valid_results = []
    test_results = []
    motif_test_results = {motif: [] for motif in HUMAN_MOTIF_VARIANTS}
    save_weights = []
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        train_dataset = dataset['train_dataset']
        train_labels = dataset['train_labels']
        np.random.seed()
        print('Initialized')
        print('Training accuracy at the beginning: %.1f%%' % accuracy(train_prediction.eval(), train_labels))
        print('Validation accuracy at the beginning: %.1f%%' % accuracy(valid_prediction.eval(), dataset['valid_labels']))
        for epoch in range(NUM_EPOCHS):
            permutation = np.random.permutation(train_labels.shape[0])
            shuffled_dataset = train_dataset[permutation, :, :]
            shuffled_labels = train_labels[permutation, :]
            for step in range(shuffled_labels.shape[0] // BATCH_SIZE):
                offset = step * BATCH_SIZE
                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
                session.run(stepOp)

            train_resuts.append(accuracy(train_prediction.eval(), train_labels))
            valid_pred = valid_prediction.eval()
            valid_results.append(accuracy(valid_pred, dataset['valid_labels']))
            test_pred = test_prediction.eval()
            test_results.append(accuracy(test_pred, dataset['test_labels']))
            for motif in HUMAN_MOTIF_VARIANTS:
                motif_test_pred = motif_test_prediction[motif].eval()
                motif_test_results[motif].append(accuracy(motif_test_pred, dataset['motif_dataset'][motif]['test_labels']))
            print('Training accuracy at epoch %d: %.1f%%' % (epoch, train_resuts[-1]))
            print('Validation accuracy: %.1f%%' % valid_results[-1])

            # Early stopping based on validation results
            if epoch > 10 and valid_results[-11] > max(valid_results[-10:]):
                train_resuts = train_resuts[:-10]
                valid_results = valid_results[:-10]
                test_results = test_results[:-10]
                motif_test_results = {motif: motif_test_results[motif][:-10] for motif in HUMAN_MOTIF_VARIANTS}
                return train_resuts, valid_results, test_results, motif_test_results, save_weights[0]

            # Model saving
            sw = {}
            for k in weights:
                sw[k] = weights[k].eval()
            if epoch < 10:
                save_weights.append(sw)
            else:
                save_weights.append(sw)
                save_weights.pop(0)

    return train_resuts, valid_results, test_results, motif_test_results, save_weights[-1]


def main(_):

    # block_print()

    hyper_dict = gen_hyper_dict(HYPER_DICT)
    pos_data, pos_labels, neg_data, neg_labels = produce_motif_dataset(NUM_FOLDS, POS_PATH, NEG_PATH)

    # Cross validate
    train_accuracy_split = []
    valid_accuracy_split = []
    test_accuracy_split = []
    motif_test_accuracy_split = {motif: [] for motif in HUMAN_MOTIF_VARIANTS}

    for i in range(NUM_FOLDS):
        split =  {
            'train': [(i + j) % NUM_FOLDS for j in range(NUM_FOLDS-2)], 
            'valid': [(i + NUM_FOLDS-2) % NUM_FOLDS], 
            'test': [(i + NUM_FOLDS-1) % NUM_FOLDS]
            }
        save = motif_data_split(pos_data, pos_labels, neg_data, neg_labels, NUM_FOLDS, split)
        dataset = {}
        dataset['train_dataset'], dataset['train_labels'] = pad_dataset(save['train_dataset'], save['train_labels'])
        dataset['valid_dataset'], dataset['valid_labels'] = pad_dataset(save['valid_dataset'], save['valid_labels'])
        dataset['test_dataset'], dataset['test_labels'] = pad_dataset(save['test_dataset'], save['test_labels'])
        dataset['motif_dataset'] = {}
        for motif in HUMAN_MOTIF_VARIANTS:
            dataset['motif_dataset'][motif] = {}
            dataset['motif_dataset'][motif]['test_dataset'], dataset['motif_dataset'][motif]['test_labels'] = pad_dataset(save['motif_dataset'][motif]['test_dataset'], save['motif_dataset'][motif]['test_labels'])

        train_resuts, valid_results, test_results, motif_test_results, save_weights = train(dataset, hyper_dict)
        print("\nbest valid epoch: %d"%(len(train_resuts)-1))
        print("Training accuracy: %.2f%%"%train_resuts[-1])
        print("Test accuracy: %.2f%%"%test_results[-1])
        print("Validation accuracy: %.2f%%"%valid_results[-1])
        for motif in HUMAN_MOTIF_VARIANTS:
            print('*%s* accuracy: %.1f%%' % (motif, motif_test_results[motif][-1]))

        # Dump model
        if FLAGS.train_dir is not None:
            with open(os.path.join(FLAGS.train_dir, 'cv%d_model.pkl'%i), 'wb') as f:
                pickle.dump(save_weights, f, 2)

        train_accuracy_split.append(train_resuts[-1])
        valid_accuracy_split.append(valid_results[-1])
        test_accuracy_split.append(test_results[-1])
        for motif in HUMAN_MOTIF_VARIANTS:
            motif_test_accuracy_split[motif].append(motif_test_results[motif][-1])

    train_accuracy = np.mean(train_accuracy_split)
    valid_accuracy = np.mean(valid_accuracy_split)
    test_accuracy = np.mean(test_accuracy_split)
    motif_test_accuracy = {}
    for motif in HUMAN_MOTIF_VARIANTS:
        motif_test_accuracy[motif] = np.mean(motif_test_accuracy_split[motif])
    print('\n\n########################\nFinal result:')
    print('Training accuracy: %.1f%%' % (train_accuracy))
    print('Validation accuracy: %.1f%%' % (valid_accuracy))
    print('Test accuracy: %.1f%%' % (test_accuracy ))
    for motif in HUMAN_MOTIF_VARIANTS:
        print('*%s* accuracy: %.1f%%' % (motif, motif_test_accuracy[motif]))

    if FLAGS.training_result_dir is not None:
        with open(os.path.join(FLAGS.training_result_dir, 'result.pkl'), 'wb') as f:
            hyper_dict['train_accuracy'] = train_accuracy
            hyper_dict['valid_accuracy'] = valid_accuracy
            hyper_dict['test_accuracy'] = test_accuracy
            hyper_dict['motif_test_accuracy'] = motif_test_accuracy
            pickle.dump(hyper_dict, f, 2)


    # enable_print()


if __name__ == '__main__':
    tf.app.run()









