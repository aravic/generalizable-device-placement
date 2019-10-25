import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from rl_agent import *
from nn_primitives import *
from progressive_nn import *


class SimpleNN(ProgressiveNN):

  def __init__(self,
               emb_size,
               n_nodes,
               n_devs,
               _,
               seed=42,
               config={},
               debug=False,
               normalize_aggs=False,
               bn_pre_classifier=False):
    ProgressiveNN.__init__(self, seed, config['dont_repeat_ff'])

    self.n_devs = n_devs
    E = self.E = tf.placeholder(tf.float32)

    # inp_size = (1+n_devs)* n_nodes
    inp_size = emb_size * n_nodes
    self.fnn = FNN(inp_size, [2 * inp_size, 2 * inp_size], n_devs, 'simple_nn')
    self.no_noise_logits = self.fnn.build(E)
    self.build_train_ops(self.no_noise_logits)

  def get_feed_dict(self, pg, node, start_times, n_peers):
    '''
    p = pg.get_placement()
    pl = [[0]* self.n_devs for _ in range(pg.n_nodes())]
    for i, n in enumerate(pg.nodes()):
      pl[i][p[n]] = 1.

    pl = flatten(pl)
    idx = pg.get_idx(node)
    o = [0]* pg.n_nodes()
    o[idx] = 1
    pl += o
    pl = np.expand_dims(pl, axis=0)
    '''
    E = pg.get_embeddings().flatten()
    E = np.expand_dims(E, axis=0)
    d = {self.E: E}
    return d
