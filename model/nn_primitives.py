import tensorflow as tf
import numpy as np

INIT_SCALE = 1

def glorot(shape, scope='default', dtype=tf.float32):
  # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
  with tf.variable_scope(scope):
    init_range = np.sqrt(6.0* INIT_SCALE/ (shape[0] + shape[1]))
    init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
    return tf.Variable(init, dtype=dtype)

def zero_init(shape, scope='default', dtype=tf.float32):
  with tf.variable_scope(scope):
    init = np.zeros(shape)
    return tf.Variable(init, dtype=dtype)

class FNN(object):
  # hidden_layer_sizes: list of hidden layer sizes
  # out_size: Size of the last softmax layer
  def __init__(self, inp_size, hidden_layer_sizes, out_size, name, dtype=tf.float32):

    layers = []
    sizes = [inp_size] + hidden_layer_sizes + [out_size]
    for i in range(len(sizes) - 1):
      w = glorot((sizes[i], sizes[i+1]), name, dtype=dtype)
      b = zero_init((1, sizes[i+1]), name, dtype=dtype)
      layers.append([w, b])

    self.layers = layers

  # *Don't* add softmax or relu at the end
  def build(self, inp_tensor):
    out = inp_tensor
    for idx, [w, b] in enumerate(self.layers):
      out = tf.matmul(out, w) + b
      if idx != len(self.layers) - 1:
        out = tf.nn.relu(out)

    return out


'''
  Combines a bunch of embeddings together at a specific node
  g(sum_i(f_i)) = relu_g(Sum_i(relu_f(e_i* M_f + b_f))* M_g + b_g)
  g(sum_i(f_i)) = relu_g((Mask* (relu_f(E* M_f + b_f)))* M_g + b_g)
  To be more specific when the number of embeddings to combine is variable,
  we use a mask Ma
  Dimensions:
    E: N x d        placeholder
    M_f: d x d1     Variable
    b_f: 1 x d1     Variable
    Ma: 1 x N       placeholder
    M_g: d1 x d2    Variable
    b_g: 1 x d2     Variable
'''

class Aggregator(object):
  # N is the max number of children to be aggregated
  # d is the degree of embeddings
  # d1 is the degree of embedding transformation
  # d2 is degree of aggregation
  def __init__(self, d, d1=None, d2=None, use_mask=True, normalize_aggs=False,
                                            small_nn=False, dtype=tf.float32):
    self.d = d
    self.d1 = d1
    self.d2 = d2
    self.normalize_aggs = normalize_aggs
    self.dtype = dtype

    if d1 is None:
      d1 = self.d1 = d
    if d2 is None:
      d2 = self.d2 = d

    self.use_mask = use_mask
    if use_mask:
      self.Ma = tf.placeholder(dtype, shape=(None, None))
      
    if small_nn:
      hidden_layer_f, hidden_layer_g = [], []
    else:
      hidden_layer_f = [self.d]
      hidden_layer_g = [self.d1]

    self.f = FNN(self.d, hidden_layer_f, self.d1, 'f', dtype=dtype)
    self.g = FNN(self.d1, hidden_layer_g, self.d2, 'g', dtype=dtype)

  def build(self, E, debug=False, mask=None):
    summ = 100

    f = tf.nn.relu(self.f.build(E))
    self.f_out = f

    if debug:
      f = tf.Print(f, [f], message='output of f: ', summarize=summ)

    if self.use_mask or mask is not None:
      if mask is None:
        mask = self.Ma

      g = tf.matmul(mask, f)
      if self.normalize_aggs:
        d = tf.cond(
              tf.reduce_sum(mask) > 0,
                  lambda: tf.reduce_sum(mask), 
                  lambda: 1.)

        g /= d

      if debug:
        print(f, g, self.Ma)
    else:
      g = tf.reduce_sum(f, 0, keepdims=True)

    if debug:
      g = tf.Print(g, [g], message='after mask: ',summarize=summ)

    g = tf.nn.relu(self.g.build(g))

    if debug:
      g = tf.Print(g, [g], message='output of g: ', summarize=summ)

    return g

  def get_ph(self):
    return self.Ma

class Classifier(object):
  def __init__(self, inp_size, hidden_layer_sizes, out_size, dtype=tf.float32):
    self.nn = FNN(inp_size, hidden_layer_sizes,
                   out_size, 'classifier', dtype=dtype)

  def build(self, inp_tensor):
    return self.nn.build(inp_tensor)

''' 
  change this to be generic across different graphs to be placed
'''
class Messenger(object):

  def __init__(self, d, d1, small_nn=False, dtype=tf.float32):
    # forward pass
    with tf.name_scope('FPA'):
      # self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
      self.fpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
    with tf.name_scope('BPA'):
      self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
      # self.bpa = Aggregator(d, d1, d1, False, small_nn=small_nn, dtype=dtype)
    with tf.name_scope('node_transform'):
      if small_nn:
        self.node_transform = FNN(d, [d], d1, 'fnn', dtype=dtype)
      else:
        self.node_transform = FNN(d, [d, d], d1, 'fnn', dtype=dtype)

  def build(self, G, node_order, E, bs=1):
    try:
      self_trans = self.node_transform.build(E)
      def message_pass(nodes, messages_from, agg):
        node2emb = {}

        for n in nodes:
          msgs = [node2emb[pred] for pred in messages_from(n)]

          node2emb[n] = tf.expand_dims(self_trans[G.get_idx(n), :],
                                        axis=0)

          if len(msgs) > 0:
            t = tf.concat(msgs, axis=0)
            inp = agg.build(t)
            node2emb[n] += inp

        return tf.concat([node2emb[n] for n in G.nodes()], 
                            axis=0)

      out_fpa = message_pass(node_order, G.predecessors, self.fpa)
      out_bpa = message_pass(reversed(node_order), G.neighbors, self.bpa)

      out = tf.concat([out_fpa, out_bpa], axis=-1)
      
      return out
    except Exception:
      import my_utils; my_utils.PrintException()
      import pdb; pdb.set_trace()

class RadialMessenger(Messenger):

  def __init__(self, k, d, d1, small_nn=False, dtype=tf.float32):
    Messenger.__init__(self, d, d1, small_nn, dtype)
    self.dtype = dtype
    self.k = k

  def build(self, G, f_adj, b_adj, E, bs=1):
    assert np.trace(f_adj) == 0
    assert np.trace(b_adj) == 0

    E = tf.cast(E, dtype=self.dtype)

    E = tf.reshape(E, [-1, tf.shape(E)[-1]])
    self_trans = self.node_transform.build(E)
    # self_trans = tf.Print(self_trans, [self_trans], message='self_trans: ', summarize=100000000)

    def message_pass(adj, agg):
        sink_mask = (np.sum(adj, axis=-1) > 0)
        # sink_mask = np.float32(sink_mask)
        # sink_mask = np.float16(sink_mask)
        # adj = np.float16(adj)
        sink_mask = tf.cast(sink_mask, self.dtype)
        adj = tf.cast(adj, self.dtype)

        x = self_trans
        for i in range(self.k):
          # x = tf.Print(x, [x], message='pre agg: x', summarize=1000)
          x = agg.build(x, mask=adj)
          # x = tf.Print(x, [x], message='x', summarize=1000)
          x = sink_mask * tf.transpose(x)
          x = tf.transpose(x)
          x += self_trans

        return x

    def f(adj):
      n = adj.shape[0]
      t = np.zeros([bs*n]* 2, dtype=np.float32)
      for i in range(bs):
        t[i*n: (i+1)* n, i*n: (i+1)*n] = adj

      return t

    f_adj = f(f_adj)
    b_adj = f(b_adj)

    with tf.variable_scope('Forward_pass'):
        out_fpa = message_pass(f_adj, self.fpa)
    with tf.variable_scope('Backward_pass'):
        out_bpa = message_pass(b_adj, self.bpa)

    out = tf.concat([out_fpa, out_bpa], axis=-1)
    out = tf.cast(out, tf.float32)

    return out

  def mess_build(self, G, node_order, E):
    return Messenger.build(self, G, node_order ,E)


