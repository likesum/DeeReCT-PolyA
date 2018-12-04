from __future__ import print_function
import numpy as np
import os
import sys
from six.moves import cPickle as pickle

HUMAN_MOTIF_VARIANTS = [
    'AATAAA',
    'ATTAAA',
    'AAAAAG',
    'AAGAAA',
    'TATAAA',
    'AATACA',
    'AGTAAA',
    'ACTAAA',
    'GATAAA',
    'CATAAA',
    'AATATA',
    'AATAGA'
]

def get_data(data_root, label):
    data = []
    for data_file in os.listdir(data_root):
        data_path = os.path.join(data_root, data_file)
        with open(data_path, 'r') as f:
            alphabet = np.array(['A', 'G', 'T', 'C'])
            for line in f:
                line = list(line.strip('\n'))
                seq = np.array(line, dtype = '|U1').reshape(-1, 1)
                seq_data = (seq == alphabet).astype(np.float32)
                data.append(seq_data)
    data = np.stack(data).reshape([-1, 206, 1, 4])
    if label:
        labels = np.zeros(data.shape[0])
    else:
        labels = np.ones(data.shape[0])
    return data, labels


def get_motif_data(data_root, label):
    data = {}
    labels = {}
    for motif in HUMAN_MOTIF_VARIANTS:
        data[motif] = []
        for data_file in os.listdir(data_root):
            if motif in data_file:
                data_path = os.path.join(data_root, data_file)
                with open(data_path, 'r') as f:
                    alphabet = np.array(['A', 'G', 'T', 'C'])
                    for line in f:
                        line = list(line.strip('\n'))
                        seq = np.array(line, dtype = '|U1').reshape(-1, 1)
                        seq_data = (seq == alphabet).astype(np.float32)
                        data[motif].append(seq_data)
        data[motif] = np.stack(data[motif]).reshape([-1, 206, 1, 4])
        if label:
            labels[motif] = np.zeros(data[motif].shape[0])
        else:
            labels[motif] = np.ones(data[motif].shape[0])
    return data, labels


def motif_data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split):
    motif_data = {}
    for motif in HUMAN_MOTIF_VARIANTS:
        motif_data[motif] = data_split(pos_data[motif], pos_labels[motif], neg_data[motif], neg_labels[motif], num_folds, split)

    train_data = np.concatenate([motif_data[motif]['train_dataset'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)
    valid_data = np.concatenate([motif_data[motif]['valid_dataset'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)
    test_data = np.concatenate([motif_data[motif]['test_dataset'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)
    train_labels = np.concatenate([motif_data[motif]['train_labels'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)
    valid_labels = np.concatenate([motif_data[motif]['valid_labels'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)
    test_labels = np.concatenate([motif_data[motif]['test_labels'] for motif in HUMAN_MOTIF_VARIANTS], axis=0)

    data = {}
    data['train_dataset'], data['train_labels'] = shuffle(train_data, train_labels)
    data['valid_dataset'], data['valid_labels'] = shuffle(valid_data, valid_labels)
    data['test_dataset'], data['test_labels'] = shuffle(test_data, test_labels)

    data['motif_dataset'] = {motif: {} for motif in HUMAN_MOTIF_VARIANTS}
    for motif in HUMAN_MOTIF_VARIANTS:
        data['motif_dataset'][motif]['test_dataset'] = motif_data[motif]['test_dataset']
        data['motif_dataset'][motif]['test_labels'] = motif_data[motif]['test_labels']

    return data


def shuffle(dataset, labels, randomState=None):
    if randomState is None:
        permutation = np.random.permutation(labels.shape[0])
    else:
        permutation = randomState.permutation(labels.shape[0])
    shuffled_data = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels


def data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split):
    pos_data_folds = np.array_split(pos_data, num_folds)
    neg_data_folds = np.array_split(neg_data, num_folds)
    pos_label_folds = np.array_split(pos_labels, num_folds)
    neg_label_folds = np.array_split(neg_labels, num_folds)

    train_pos_data = np.concatenate([pos_data_folds[i] for i in split['train']], axis=0)
    train_pos_labels = np.concatenate([pos_label_folds[i] for i in split['train']], axis=0)
    valid_pos_data = np.concatenate([pos_data_folds[i] for i in split['valid']], axis=0)
    valid_pos_labels = np.concatenate([pos_label_folds[i] for i in split['valid']], axis=0)

    train_neg_data = np.concatenate([neg_data_folds[i] for i in split['train']], axis=0)
    train_neg_labels = np.concatenate([neg_label_folds[i] for i in split['train']], axis=0)
    valid_neg_data = np.concatenate([neg_data_folds[i] for i in split['valid']], axis=0)
    valid_neg_labels = np.concatenate([neg_label_folds[i] for i in split['valid']], axis=0)

    train_data = np.concatenate((train_pos_data, train_neg_data), axis=0)
    valid_data = np.concatenate((valid_pos_data, valid_neg_data), axis=0)
    train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
    valid_labels = np.concatenate((valid_pos_labels, valid_neg_labels), axis=0)

    data = {}
    data['train_dataset'], data['train_labels'] = shuffle(train_data, train_labels)
    data['valid_dataset'], data['valid_labels'] = shuffle(valid_data, valid_labels)

    if 'test' in split:
        test_pos_data = np.concatenate([pos_data_folds[i] for i in split['test']], axis=0)
        test_pos_labels = np.concatenate([pos_label_folds[i] for i in split['test']], axis=0)
        test_neg_data = np.concatenate([neg_data_folds[i] for i in split['test']], axis=0)
        test_neg_labels = np.concatenate([neg_label_folds[i] for i in split['test']], axis=0)
        test_data = np.concatenate((test_pos_data, test_neg_data), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)
        data['test_dataset'], data['test_labels'] = shuffle(test_data, test_labels)

    return data


def produce_dataset(num_folds, pos_path, neg_path, seed=0):
    pos_data, pos_labels = get_data(pos_path, True)
    neg_data, neg_labels = get_data(neg_path, False)
    randomState = np.random.RandomState(seed)
    pos_data, pos_labels = shuffle(pos_data, pos_labels, randomState)
    neg_data, neg_labels = shuffle(neg_data, neg_labels, randomState)
    print('Positive:', pos_data.shape, pos_labels.shape)
    print('Negative:', neg_data.shape, neg_labels.shape)
    return pos_data, pos_labels, neg_data, neg_labels


def produce_motif_dataset(num_folds, pos_path, neg_path, seed=0):
    pos_data, pos_labels = get_motif_data(pos_path, True)
    neg_data, neg_labels = get_motif_data(neg_path, False)
    randomState = np.random.RandomState(seed)
    for motif in HUMAN_MOTIF_VARIANTS:
        pos_data[motif], pos_labels[motif] = shuffle(pos_data[motif], pos_labels[motif], randomState)
        neg_data[motif], neg_labels[motif] = shuffle(neg_data[motif], neg_labels[motif], randomState)
        print('Positive %s:'%motif, pos_data[motif].shape, pos_labels[motif].shape)
        print('Negative %s:'%motif, neg_data[motif].shape, neg_labels[motif].shape)
    return pos_data, pos_labels, neg_data, neg_labels


def motif_exclude_data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split, excluded_motifs=[]):
    motif_data = {}
    for motif in HUMAN_MOTIF_VARIANTS:
        if motif not in excluded_motifs:
            motif_data[motif] = data_split(pos_data[motif], pos_labels[motif], neg_data[motif], neg_labels[motif], num_folds, split)
    
    # We don't need test data for motifs that are not in excluded_motifs
    train_data = np.concatenate([motif_data[motif]['train_dataset'] for motif in HUMAN_MOTIF_VARIANTS if motif not in excluded_motifs], axis=0)
    valid_data = np.concatenate([motif_data[motif]['valid_dataset'] for motif in HUMAN_MOTIF_VARIANTS if motif not in excluded_motifs], axis=0)
    train_labels = np.concatenate([motif_data[motif]['train_labels'] for motif in HUMAN_MOTIF_VARIANTS if motif not in excluded_motifs], axis=0)
    valid_labels = np.concatenate([motif_data[motif]['valid_labels'] for motif in HUMAN_MOTIF_VARIANTS if motif not in excluded_motifs], axis=0)

    data = {}
    data['train_dataset'], data['train_labels'] = shuffle(train_data, train_labels)
    data['valid_dataset'], data['valid_labels'] = shuffle(valid_data, valid_labels)

    return data


def motif_exclude_get_test_data(pos_data, pos_labels, neg_data, neg_labels, excluded_motifs=[]):
    # Use all data of motifs in excluded_motifs for testing
    pos_test_data = np.concatenate([pos_data[motif] for motif in excluded_motifs], axis=0)
    neg_test_data = np.concatenate([neg_data[motif] for motif in excluded_motifs], axis=0)
    test_data = np.concatenate([pos_test_data, neg_test_data])
    pos_test_labels = np.concatenate([pos_labels[motif] for motif in excluded_motifs], axis=0)
    neg_test_labels = np.concatenate([neg_labels[motif] for motif in excluded_motifs], axis=0)
    test_labels = np.concatenate([pos_test_labels, neg_test_labels])
    test_data, test_labels = shuffle(test_data, test_labels)
    return test_data, test_labels

