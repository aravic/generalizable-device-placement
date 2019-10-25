import multiprocessing as mp
from multiprocessing import Queue
import threading
import numpy as np
from numpy.linalg import norm as np_norm
import time, os, json, random
import copy
import sys
import tensorflow as tf
from cust_tb_summ import CustomTBSummary

class Coordinator(object):
  def __init__(self):
    self.set_seeds()

  def start(self, config, agent_fn):
    verbose = self.verbose = config['debug_verbose']
    n_agents = config['n_workers']
    scale_norm = config['scale_norm']
    self.config = config
    self.n_agents = n_agents
    self.baseline_mask = config['baseline_mask']
    self.graph_names = config['mul_graphs']
    assert self.graph_names is not None

    if self.baseline_mask is None:
      self.baseline_mask = [0]* n_agents
    print("Number of agents: %d"% n_agents)
    gpus = config['use_gpus']
    if config['shuffle_gpu_order']:
      random.seed(config['seed'])
      random.shuffle(gpus)

    assert n_agents > 0

    params_send_qs = [Queue(1) for _ in range(n_agents)]
    params_recv_qs = [Queue(1) for _ in range(n_agents)]
    grad_send_qs = [Queue(1) for _ in range(n_agents)]
    grad_recv_qs = [Queue(1) for _ in range(n_agents)]
    summ_send_qs = [Queue(1) for _ in range(n_agents)]
    summ_recv_qs = [Queue(1) for _ in range(n_agents)]
    send_baseline_qs = [Queue(1) for _ in range(n_agents)]
    recv_baseline_qs = [Queue(1) for _ in range(n_agents)]

    configs = []
    for i in range(n_agents):
      c = copy.deepcopy(config)
      c['id'] = i
      c['name'] += '/%s-%d'% (c['name'], i)
      c['params_send_q'] = params_recv_qs[i]
      c['params_recv_q'] = params_send_qs[i]
      c['grads_send_q'] = grad_recv_qs[i]
      c['grads_recv_q'] = grad_send_qs[i]
      c['summ_send_q'] = summ_recv_qs[i]
      c['summ_recv_q'] = summ_send_qs[i]
      c['send_baseline_q'] = recv_baseline_qs[i]
      c['recv_baseline_q'] = send_baseline_qs[i]
      if len(c['pickled_inp_file']) > 0:
          c['pickled_inp_file'] = \
                    [c['pickled_inp_file'][i % len(c['pickled_inp_file'])]]

      if c['remote_async_start_ports'] is not None:

          for j, p in enumerate(c['remote_async_start_ports']):
            c['remote_async_start_ports'][j] = p + c['remote_async_n_sims'][j]* i

      configs.append(c)

    self.summ_recv_qs, self.summ_send_qs = summ_recv_qs, summ_send_qs
    threading.Thread(target=self.handle_summaries).start()
    self.send_baseline_qs, self.recv_baseline_qs = send_baseline_qs, recv_baseline_qs
    threading.Thread(target=self.handle_baselines).start()

    agents = []
    for i in range(n_agents):
      runnable = mp.Process
      if config['use_threads']:
        runnable = threading.Thread
      agents.append(runnable(target=agent_fn, args=(configs[i],)))
 
    for i in range(n_agents):
      if gpus is not None:
        g = gpus[i % len(gpus)]
        os.environ['CUDA_VISIBLE_DEVICES'] = g
      agents[i].start()

    if self.config['restore_from'] is None or self.config['dont_restore_softmax']:
        for i in range(n_agents):
          init_ws = params_recv_qs[i].get()

        for i in range(n_agents):
          params_send_qs[i].put(init_ws)

    print("Cordinator initialization sequence finished")

    while True:
      a_ws, a_gs, a_norm = [], [], []

      for i in range(n_agents):
        if verbose:
          print('Coordinator: Getting gradients from %d'% i) 
          sys.stdout.flush()
        gs = grad_recv_qs[i].get()
        
        for k, v in gs.items():
          gs[k] = np.float64(v)
          if config['dont_share_classifier']:
            assert 'classifier' not in k

        if scale_norm:
          norms = {}
          for k, v in gs.items():
            norm = norms[k] = np.float64(np.linalg.norm(v))
            if norm > 0:
              gs[k] /= norm
          a_norm.append(norms)

        a_gs.append(gs)

      grad_out = a_gs[0]
      for gs in a_gs[1:]:
        assert gs.keys() == grad_out.keys()
        for k in grad_out:
          grad_out[k] += gs[k]

      if scale_norm:
        for k, v in grad_out.items():
          grad_out[k] = v* np.sum([norm[k] for norm in a_norm],
                                              dtype=np.float64)

      for i in range(n_agents):
        if verbose:
          print('Coordinator: Putting gradients into %d'% i) 
          sys.stdout.flush()
        grad_send_qs[i].put(grad_out)

      for i in range(n_agents):
        if verbose:
          print('Coordinator: Getting params from %d'% i) 
          sys.stdout.flush()
        a_ws.append(params_recv_qs[i].get())

      if verbose:
        print('Coordinator: Episode sequence finished')
        sys.stdout.flush()

      any_agent_violation = False

      for i, w in enumerate(a_ws[1:]):

        assert w.keys() == a_ws[0].keys()
        violation = False

        for k, v in w.items():
          if not (v == a_ws[0][k]).all():
            violation = True
            any_agent_violation = True
            if np_norm(v.flatten() - a_ws[0][k].flatten()) / np_norm(v.flatten()) >= 1e-3:
              print('ERROR: Weights diverged too far')
              import pdb; pdb.set_trace()
              raise Exception('Weights not consistent')

        if violation:
            params_send_qs[i + 1].put(a_ws[0])
        else:
            params_send_qs[i + 1].put(None)

      if any_agent_violation:
        params_send_qs[0].put(a_ws[0])
      else:
        params_send_qs[0].put(None)

  def handle_summaries(self):

    self.init_summary()
    n_agents = self.n_agents
    group_is = {}
    for i, m in enumerate(self.baseline_mask):
      if m not in group_is:
        group_is[m] = []
      group_is[m].append(i)

    while True:
      summs = []
      for i in range(n_agents):
        summs += [self.summ_recv_qs[i].get()]

      for i in range(n_agents):
        self.summ_send_qs[i].put(True)

      for m, l_i in group_is.items():
          combine_d = {}
          eps = []
          for i in range(n_agents):
            if i not in l_i:
              continue
            d, ep, is_eval = summs[i]
            eps.append(ep)
            for k, v in d.items():
              if k not in combine_d:
                combine_d[k] = [v]
              else:
                combine_d[k].append(v)

          for k, v in combine_d.items():
            if k in self.avg_aggs:
              combine_d[k] = np.average(v)
            elif k in self.min_aggs:
              combine_d[k] = np.min(v)
            else:
              raise Exception('unrecognized combine option')

          assert np.min(eps) == np.max(eps)

          writer = self.summ_writer[m]
          if is_eval:
            writer = self.eval_writer[m]

          writer.write(combine_d, eps[0])

  def handle_baselines(self):
    n_agents = self.n_agents
    id_to_center = {}
    center_to_bl_collections_mold = {}

    for i, m in enumerate(self.baseline_mask):
      id_to_center[i] = m
      center_to_bl_collections_mold[m] = []

    while True:
      center_to_bl_collections = copy.deepcopy(center_to_bl_collections_mold)

      for i in range(n_agents):
        if self.verbose:
            print('Coordinator: receiving baseline from agent %d' % i)
        bl = self.recv_baseline_qs[i].get()
        center_to_bl_collections[id_to_center[i]].append(bl)

      for cent, bl_collect in center_to_bl_collections.items():
          center_to_bl_collections[cent] = np.mean(bl_collect, axis=0, dtype=np.float64)

      for i in range(n_agents):
        bl = center_to_bl_collections[id_to_center[i]]
        if self.verbose:
            print('Coordinator: sending baseline from agent %d' % i)
        self.send_baseline_qs[i].put(bl)


  def init_summary(self):
    config = self.config
    summ_names = ['run_times/episode_end_rt', 'run_times/best_so_far', 'run_times/best_rew_rt', 'ent/tanhc_const', 'run_times/ep_best_rt']
    
    if not config['dont_sim_mem']:
      summ_names += ['run_times/rew_rt', 'mem/mem_util', 'mem/best_mem_util_so_far']

    if config['supervised']:
      summ_names += ['loss/loss', 'opt/logits', 'opt/lr']
    else:
      summ_names += ['rew/reward', 'loss/loss', 'rew/baseline', 'rew/advantage', 'loss/log_probs',\
              'opt/lr', 'ent/entropy', 'opt/grad_norm', 'ent/ent_dec', 'opt/pre_sync_grad_norm',
              'loss/entropy_grad_norm', 'loss/reinforce_grad_norm']

    self.avg_aggs = set(['run_times/episode_end_rt', 'loss/loss', 'opt/logits', 'opt/lr', 'rew/reward', 'rew/baseline', 'rew/advantage', 'loss/log_probs', 'ent/entropy', 'opt/grad_norm', 'ent/ent_dec', 'opt/pre_sync_grad_norm', 'ent/tanhc_const', 'mem/mem_util', 'run_times/ep_best_rt', 'run_times/rew_rt'])
    
    self.min_aggs = set(['run_times/best_so_far', 'run_times/best_rew_rt', 'run_times/argmax_ep_best_rt',  'mem/best_mem_util_so_far'])

    seen_masks = set()
    self.summ_writer, self.eval_writer = [], []
    for i, m in enumerate(self.baseline_mask):
      if m in seen_masks:
        continue

      seen_masks.add(m)
      g = self.graph_names[i]
      tb_dir = 'models/tb-logs/%s/%s' % (config['name'], g)

      if len(config['name']) > 0:
          os.system('rm -r %s 2> /dev/null' % tb_dir)

      os.system('mkdir -p %s 2> /dev/null' % tb_dir)

      tb_writer = tf.summary.FileWriter(tb_dir)
      self.summ_writer += [CustomTBSummary(tb_writer, summ_names)]
      self.eval_writer += [CustomTBSummary(tb_writer, ['run_times/argmax_ep_best_rt'])]


  def set_seeds(self, i = 0):
    if i is None:
      i = 0
    s = 42 + i
    np.random.seed(s)
    random.seed(s)

