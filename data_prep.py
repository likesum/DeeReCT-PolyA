import numpy as np
import os
import sys
import argparse

def get_data(data_root, label):
    """
    Process all files in the input directory to read sequences.
    Sequences are encoded with one-hot.
    data_root: input directory
    label: the label (True or False) for the sequences in the input directory
    """
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


def shuffle(dataset, labels, randomState=None):
    """
    Shuffle sequences and labels jointly
    """
    if randomState is None:
        permutation = np.random.permutation(labels.shape[0])
    else:
        permutation = randomState.permutation(labels.shape[0])
    shuffled_data = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels


def data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split):
    """
    Split the dataset into num_folds folds.
    Then split train, valid, and test sets according to the input dict split.
    """
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


def produce_dataset(pos_path, neg_path, seed=0):
    pos_data, pos_labels = get_data(pos_path, True)
    neg_data, neg_labels = get_data(neg_path, False)
    randomState = np.random.RandomState(seed)
    pos_data, pos_labels = shuffle(pos_data, pos_labels, randomState)
    neg_data, neg_labels = shuffle(neg_data, neg_labels, randomState)
    print('Positive:', pos_data.shape, pos_labels.shape)
    print('Negative:', neg_data.shape, neg_labels.shape)
    return pos_data, pos_labels, neg_data, neg_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pos_root', help='Directory of files containing positive data')
    parser.add_argument('neg_root', help='Directory of files containing negative data')
    parser.add_argument('outfile', help='Save the processed dataset to')
    parser.add_argument('--nfolds', default=5, type=int, help='Seperate the data into how many folds')
    opts = parser.parse_args()

    pos_data, pos_labels = get_data(opts.pos_root, True)
    neg_data, neg_labels = get_data(opts.neg_root, False)
    randomState = np.random.RandomState(0)
    pos_data, pos_labels = shuffle(pos_data, pos_labels, randomState)
    neg_data, neg_labels = shuffle(neg_data, neg_labels, randomState)

    print('Read %d positive sequences from %s'%(pos_labels.shape[0], opts.pos_root))
    print('Read %d negative sequences from %s\n'%(neg_labels.shape[0], opts.neg_root))

    num_folds = opts.nfolds
    split_dict = {
        'train': [i for i in range(num_folds-2)],
        'valid': [num_folds-2],
        'test': [num_folds-1]
    }

    dataset = data_split(pos_data, pos_labels, neg_data, neg_labels, num_folds, split_dict)
    print('Size of training dataset: %d'%dataset['train_labels'].shape[0])
    print('Size of validation dataset: %d'%dataset['valid_labels'].shape[0])
    print('Size of test dataset: %d\n'%dataset['test_labels'].shape[0])

    np.savez(opts.outfile, **dataset)
    print('Finish writing dataset to %s'%opts.outfile)





