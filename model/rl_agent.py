import rl_params as params
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import adam, rmsprop, gradient_descent
from tensorflow.python.ops import clip_ops

class ReinforceAgent(object):

  def __init__(self, seed, bs):
    self.seed = seed
    self.bs = bs
    self.rew = tf.placeholder(tf.float32)
    self.baseline = tf.placeholder(tf.float32)
    self.adv = self.rew - self.baseline
    self.logits_noise_train_ph = tf.placeholder(tf.float32)
    self.sample_ph = tf.placeholder(tf.float32)
    self.is_eval_ph = tf.placeholder_with_default(0., None)

    # self.episode = tf.placeholder(tf.float32)
    self.global_step = tf.train.get_or_create_global_step()
    self.init_global_step = tf.assign(self.global_step, 0)

    self.lr_init = params.lr_init
    self.lr_dec = params.lr_dec
    self.lr_start_decay_step = params.lr_start_decay_step
    self.lr_decay_steps = params.lr_decay_steps
    self.lr_min = params.lr_min
    self.lr_dec_approach = params.lr_dec_approach

    self.ent_dec_init = params.ent_dec_init
    self.ent_dec = params.ent_dec
    self.ent_start_dec_step = params.ent_start_dec_step
    self.ent_dec_steps = params.ent_dec_steps
    self.ent_dec_min = params.ent_dec_min
    self.ent_dec_lin_steps = params.ent_dec_lin_steps
    self.ent_dec_approach = params.ent_dec_approach

    self.optimizer_type = params.optimizer_type

    self.eps_init = params.eps_init
    self.eps_dec_steps = params.eps_dec_steps
    self.start_eps_dec_step = params.start_eps_dec_step
    self.stop_eps_dec_step = params.stop_eps_dec_step
    self.eps_dec_rate = params.eps_dec_rate

    self.tanhc_init = params.tanhc_init
    self.tanhc_dec_steps = params.tanhc_dec_steps
    self.tanhc_max = params.tanhc_max
    self.tanhc_start_dec_step = params.tanhc_start_dec_step

    self.tanhc_decay_func = None
    if self.tanhc_init:
      tanhc_decay_func = tf.train.polynomial_decay(self.tanhc_init,
            self.global_step - self.tanhc_start_dec_step, self.tanhc_dec_steps, self.tanhc_max)

      self.tanhc_decay_func = tf.minimum(tanhc_decay_func, self.tanhc_max)

    self.no_grad_clip = params.no_grad_clip
    self.set_vars_ops = None
    ent_gstep = self.global_step - self.ent_start_dec_step
    if self.ent_dec_approach == 'exponential':
        self.ent_dec_func = tf.train.exponential_decay(self.ent_dec_init, 
                        ent_gstep, self.ent_dec_steps, self.ent_dec, False),
    elif self.ent_dec_approach == 'linear':
        self.ent_dec_func = tf.train.polynomial_decay(self.ent_dec_init, 
                        ent_gstep, self.ent_dec_lin_steps, self.ent_dec_min)
    elif self.ent_dec_approach == 'step':
        self.ent_dec_func = tf.constant(self.ent_dec_min)

    ent_dec = tf.cond(
          tf.less(self.global_step, self.ent_start_dec_step),
          lambda: tf.constant(self.ent_dec_init),
          lambda: self.ent_dec_func,
          name = 'ent_decay')

    self.ent_dec = tf.maximum(ent_dec, self.ent_dec_min)

  def setup_lr(self):

    lr_gstep = self.global_step - self.lr_start_decay_step
    lr_init = self.lr_init

    def f1():
      return tf.constant(lr_init)

    def exp_f():
      return tf.train.exponential_decay(lr_init, lr_gstep,
                               self.lr_decay_steps, self.lr_dec, True)

    def poly_f():
      return tf.train.polynomial_decay(lr_init, lr_gstep,
                              self.lr_decay_steps, self.lr_min)

    f2 = exp_f
    if self.lr_dec_approach == 'linear':
      f2 = poly_f

    learning_rate = tf.cond(
        tf.less(self.global_step, self.lr_start_decay_step),
        f1,
        f2,
        name="learning_rate")
    self.lr = tf.maximum(learning_rate, self.lr_min)
    return self.lr 

  def _get_optimizer(self):

    lr = self.lr = self.setup_lr()

    # tf.summary.scalar('lr', self.lr)
    optimizer_type = self.optimizer_type
    if optimizer_type == "adam":
      opt = adam.AdamOptimizer(lr)
    elif optimizer_type == "sgd":
      opt = gradient_descent.GradientDescentOptimizer(lr)
    elif optimizer_type == "rmsprop":
      opt = rmsprop.RMSPropOptimizer(lr)
    return opt

  '''
    build train ops such that the advantage can be scaled to the gradients
    without re-evaluating the feed-forward phase (cache gradients during feed forward
    stage)
  '''
  def _build_train_ops(self,
                       grad_bound=1.25,
                       dont_repeat_ff=False):

    tf_variables = tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES),

    opt = self._get_optimizer()

    pl_ent_loss = self.pl_ent_loss

    # print some ent, adv stats
    all_grads = []
    b_grads = []
    for i in range(self.bs):
      with tf.variable_scope('log_prob_grads'):
        grads_and_vars = opt.compute_gradients(self.log_prob_loss[i], tf_variables)
      b_grads.append(grads_and_vars)
      for x in grads_and_vars:
        all_grads.append(x)

    grad_norm = clip_ops.global_norm([tf.cast(g, tf.float32) for g, _ in all_grads if g is not None])
    self.logprob_grad_outs = [[g for g, _ in b_grads[i] if g is not None] for i in range(self.bs)]

    # print some ent, adv stats
    all_grads2 = []
    b_grads2 = []
    for i in range(self.bs):
      with tf.variable_scope('placement_ent_grads'):
        grads_and_vars2 = opt.compute_gradients(pl_ent_loss[i], tf_variables)
      b_grads2.append(grads_and_vars2)
      for x in grads_and_vars2:
        all_grads2.append(x)

    grad_norm2 = clip_ops.global_norm([tf.cast(g, tf.float32) for g, _ in all_grads2 if g is not None])
    self.ent_grad_outs = [[g for g, _ in b_grads2[i] if g is not None] for i in range(self.bs)]

    self.reinforce_grad_norm = tf.reduce_mean(grad_norm)
    self.entropy_grad_norm = tf.reduce_mean(grad_norm2)
    self.grad_phs = []
    self.grad_outs = []
    gradphs_and_vars = []

    # if not dont_repeat_ff:
    # grads_and_vars = opt.compute_gradients(loss, tf_variables)
    self.grad_outs = None

    for i, [g, v] in enumerate(grads_and_vars):
      if g is not None:
        # if not dont_repeat_ff: 
        # self.grad_outs.append(g)
        grad_vtype = tf.float32
        if v.dtype == tf.as_dtype('float16_ref'):
          grad_vtype = tf.float16
        p = tf.placeholder(grad_vtype, name='grad_phs_%d' % i)
        self.grad_phs.append(p)
        gradphs_and_vars.append((p, v))

    self.grad_norm = tf.global_norm([tf.cast(g, tf.float32) for g in self.grad_phs])

    clipped_grads = gradphs_and_vars
    self.gradphs_and_vars = gradphs_and_vars
    
    if not self.no_grad_clip:
      clipped_grads = self._clip_grads_and_vars(gradphs_and_vars, 
                                                  self.grad_norm, grad_bound)

    train_op = opt.apply_gradients(clipped_grads, self.global_step)

    return train_op, self.grad_outs, self.logprob_grad_outs, self.ent_grad_outs

  def get_vars(self, sess, share_classifier=True):
    d = {}
    l = []
    for _, v in self.gradphs_and_vars:
      if not share_classifier:
        if 'classifier' in v.name:
          continue
      l.append(v)

    vs = sess.run(l)

    for i, (_, v) in enumerate(self.gradphs_and_vars):
      if not share_classifier:
        if 'classifier' in v.name:
          continue
      d[v.name] = np.float64(vs[i])

    return d

  def set_vars(self, sess, var_vals, share_classifier=True):
    if self.set_vars_ops is None:
      self.set_vars_ops = []
      self.set_vars_ph = {}

      for _, v in self.gradphs_and_vars:
        if (not share_classifier) and 'classifier' in v.name:
          continue
        ph = tf.placeholder(tf.float32)
        self.set_vars_ph[v.name] = ph
        set_op = v.assign(ph)
        self.set_vars_ops.append(set_op)

    d = {}
    for k, v in self.set_vars_ph.items():
      d[v] = np.float32(var_vals[k])

    sess.run(self.set_vars_ops, feed_dict=d)

  def _clip_grads_and_vars(self, grads_and_vars, grad_norm, grad_bound):

    all_grad_norms = {}
    clipped_grads = []
    clipped_rate = tf.maximum(grad_norm / grad_bound, 1.0)

    for g, v in grads_and_vars:
      if g is not None:
        if isinstance(g, tf_ops.IndexedSlices):
          raise Exception('IndexedSlices not allowed here')
        else:
          clipped = g / tf.cast(clipped_rate, g.dtype)
          norm_square = tf.reduce_sum(clipped * clipped, axis=-1)

        all_grad_norms[v.name] = tf.sqrt(norm_square)
        clipped_grads.append((clipped, v))

    return clipped_grads

  def get_apply_grad_feed(self, grads):
    f = {}
    for i, p in enumerate(self.grad_phs):
      f[p] = grads[i]

    return f

  def _sample(self, logits, sample_ph=False):

    sample_argmax = tf.argmax(logits, axis=-1)

    if sample_ph:
      sample = self.sample_ph

    else:
      sample = tf.multinomial(logits, 1, seed=self.seed)

      if self.eps_init > 0:
        # add epsilon greedy to the above sampled action
        eps = tf.random_uniform((1,1))
        decayed_eps = tf.train.exponential_decay(
                        self.eps_init, 
                        self.global_step, 
                        self.eps_dec_steps,
                        self.eps_dec_rate)

        eps_thres = tf.cond(self.global_step > self.start_eps_dec_step, 
                            lambda: decayed_eps,
                            lambda: tf.constant(self.eps_init))

        eps_thres = tf.cond(self.global_step < self.stop_eps_dec_step,
                              lambda: eps_thres,
                              lambda: tf.constant(1.))

        sample = tf.cond(tf.reduce_sum(eps) < eps_thres,
                          lambda seed=self.seed: tf.multinomial(tf.ones_like(logits), 1, seed=seed),
                          lambda: sample)

      sample = tf.cond(tf.reduce_sum(self.is_eval_ph) > .5,
                    lambda: sample_argmax,
                    lambda: sample)


    sample = tf.reshape(tf.to_int32(sample), [-1])
    sample_argmax = tf.reshape(tf.to_int32(sample_argmax), [-1])
    # use during eval phase
    expl_act = tf.logical_not(tf.equal(sample, sample_argmax))
    log_probs = -1.* tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=logits, labels=sample)

    return sample, log_probs, expl_act

  def _add_entropy(self, loss, logits):
    # e = tf.reduce_sum(self._get_entropy(logits), axis=-1)
    e = self._get_entropy(logits)
    loss -= self.ent_dec* e
    return loss, e, self.ent_dec

  def _get_entropy(self, logits):
    # with tf.name_scope('Entropy_logits'):
    p = tf.nn.softmax(logits)
    lp = tf.log(p + 1e-3)
    e = -p* lp
    e = tf.reduce_sum(e, axis=-1)
    return e

  # def update_baseline(self, new_rew):
  #   self.bl = self.bl_dec* self.bl + (1 - self.bl_dec)* new_rew

  # def get_baseline(self):
  #   return self.bl
