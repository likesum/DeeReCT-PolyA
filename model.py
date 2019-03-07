import numpy as np
import tensorflow as tf

BATCH_SIZE = 64
PATCH_SIZE = 10
DEPTH = 16
NUM_HIDDEN = 64
SEQ_LEN = 206 + 2*PATCH_SIZE-2
NUM_CHANNELS = 4
NUM_LABELS = 2
NGROUPS = 4

def gen_hyper_dict(hyper_dict=None):
    def rand_log(a, b):
        x = np.random.sample()
        return 10.0 ** ((np.log10(b) - np.log10(a)) * x + np.log10(a))

    def rand_sqrt(a, b):
        x = np.random.sample()
        return (b - a) * np.sqrt(x) + a

    hyper_dict = {
        'tf_learning_rate': rand_log(.0005, .05),
        'tf_momentum': rand_sqrt(.95, .99),
        'tf_motif_init_weight': rand_log(1e-2, 10),
        'tf_fc_init_weight': rand_log(1e-2, 10),
        'tf_motif_weight_decay': rand_log(1e-5, 1e-3),
        'tf_fc_weight_decay': rand_log(1e-5, 1e-3),
        'tf_keep_prob': np.random.choice([.5, .75, 1.0]),
    }
    return hyper_dict


class Net:
    def __init__(self, hyper_dict=None):
        self.hyper_dict = gen_hyper_dict() if hyper_dict is None else hyper_dict
        self.build_model()

    def build_model(self):
        # Load hyper-params
        hyper_dict = self.hyper_dict
        tf_motif_init_weight = hyper_dict['tf_motif_init_weight']
        tf_fc_init_weight = hyper_dict['tf_fc_init_weight']
        tf_motif_weight_decay = hyper_dict['tf_motif_weight_decay']
        tf_fc_weight_decay = hyper_dict['tf_fc_weight_decay']
        tf_keep_prob = hyper_dict['tf_keep_prob']

        self.dataset = tf.placeholder(
              tf.float32, shape=(None, SEQ_LEN, 1, NUM_CHANNELS))
        self.labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
        self.istrain = tf.placeholder(tf.bool, shape=[])

        keep_prob = tf.where(self.istrain, tf_keep_prob, 1.0)

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
        self.weights = weights

        conv = tf.nn.conv2d(self.dataset, conv_weights, [1, 1, 1, 1], padding='VALID')
        conv = tf.reshape(conv, [-1, 215, 1, DEPTH//NGROUPS, NGROUPS])
        mu, var = tf.nn.moments(conv, [1, 2, 3], keep_dims=True)
        conv = (conv - mu) / tf.sqrt(var + 1e-12)
        conv = tf.reshape(conv, [-1, 215, 1, DEPTH])
        hidden = tf.nn.relu(conv + conv_biases)
        hidden = tf.nn.max_pool(hidden, [1, 10, 1, 1], [1, 10, 1, 1], padding = 'VALID')
        shape = hidden.get_shape().as_list()
        motif_score = tf.reshape(hidden, [-1, shape[1]*DEPTH])
        hidden_nodes = tf.nn.dropout(tf.nn.relu(tf.matmul(motif_score, layer1_weights) + layer1_biases), tf_keep_prob)
        logits = tf.matmul(hidden_nodes, layer2_weights) + layer2_biases
        self.logits = logits
        self.prediction = tf.nn.softmax(logits)

    def build_training_graph(self):
        hd = self.hyper_dict
        wts = self.weights

        # Loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels, 
            logits=self.logits))
        self.loss += hd['tf_fc_weight_decay']*(tf.nn.l2_loss(wts['layer1_weights'])+tf.nn.l2_loss(wts['layer2_weights'])) 
        self.loss += hd['tf_motif_weight_decay']*tf.nn.l2_loss(wts['conv_weights'])

        # Optimizer.
        global_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
        self.stepOp = tf.assign_add(global_step, 1).op
        learning_rate = tf.train.exponential_decay(hd['tf_learning_rate'], global_step, 3000, 0.96)
        self.optimizeOp = tf.train.MomentumOptimizer(learning_rate, hd['tf_momentum']).minimize(self.loss)


    def get_prediction(self, sess, data, istrain=True):
        fd = {self.dataset: data, self.istrain: istrain}
        return sess.run(self.prediction, feed_dict=fd)


    def load_weights(self, wts, sess):
        wts = np.load(wts)
        ph = tf.placeholder(tf.float32)
        for k in self.weights:
            sess.run(tf.assign(self.weights[k], ph).op, feed_dict={ph: wts[k]})

