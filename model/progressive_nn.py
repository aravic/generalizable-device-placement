import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from rl_agent import *
from nn_primitives import *


class ProgressiveNN(ReinforceAgent):
  is_supervised = False

  def __init__(self, seed=42, dont_repeat_ff=False, bs=1):
    ReinforceAgent.__init__(self, seed, bs)
    self.dont_repeat_ff = dont_repeat_ff

  def get_eval_ops(self):
    return [self.sample, self.logits, self.log_probs]

  def build_train_ops(self, logits):

    self.logits = logits
    self.sample, self.log_probs, self.expl_act = self._sample(logits)

    self.entropy = self._get_entropy(logits)

    # note that loss resamples instead of reading from ph
    self.pl_ent_loss = - self.entropy* self.ent_dec
    self.log_prob_loss = - self.log_probs

    # high overhead
    # self.logits_logprob_grad = tf.gradients(self.log_prob_loss, self.logits_eval)
    # self.logits_ent_grad = tf.gradients(self.pl_ent_loss, self.logits_eval)
    # self.logits_train_grad = tf.gradients(self.loss, self.logits_train)
    
    self.train_op, self.grad_outs, self.logprob_grad_outs, self.ent_grad_outs =\
                        self._build_train_ops(dont_repeat_ff=self.dont_repeat_ff)

  def get_train_specific_fd(self, rew, baseline, ln, sample):
    d = {self.rew: rew,
         self.baseline: baseline,
         self.logits_noise_train_ph: ln,
         self.sample_ph: sample}
    return d
