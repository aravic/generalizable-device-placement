import networkx as nx
import sys, random, os, copy, json
from time import time
import numpy as np 
import tensorflow as tf
sys.path.append('./')
sys.path.append('progressive_placers/')
sys.path.append('../')
from cust_tb_summ import CustomTBSummary
from pgraph import *
from simplified_utils import *
import math
from numpy import average
from collections import deque, defaultdict
import matplotlib as mpl    
mpl.use('Agg')
import matplotlib.pyplot as plt
from heapq import heappush, heappushpop
import copy
import multiprocessing as mp
import threading
from tensorflow.python.framework import ops as tf_ops


class ProgressivePlacer:

  is_supervised = False
  '''
      G needs to be annotated with attributes: cost, output_shape
  '''
  def place(self, G, n_devs, nn_model, sim_eval, config, pptfitem):

    self.id = config.get('id', None)
    self.seed = self.id + config['seed']
    self.set_seeds(self.seed)
    self.params_send_q = config.get('params_send_q', None)
    self.params_recv_q = config.get('params_recv_q', None)
    self.grads_send_q = config.get('grads_send_q', None)
    self.grads_recv_q = config.get('grads_recv_q', None)
    self.summ_recv_q = config.get('summ_recv_q', None)
    self.summ_send_q = config.get('summ_send_q', None)
    self.send_baseline_q = config.get('send_baseline_q', None)
    self.recv_baseline_q = config.get('recv_baseline_q', None)
    self.bs = config['num_children']
    self.n_peers = config['n_peers']
    self.n_devs = n_devs
    self.sim_eval = sim_eval
    self.disc_fact = config['disc_factor']
    n_episodes = config['n_eps']
    self.max_rnds = config['max_rnds']
    self.print_freq = config['print_freq']
    self.discard_last_rnds = config['discard_last_rnds']
    self.tb_dir = 'models/tb-logs/%s' % config['name']
    self.eval_dir = 'models/eval-dir/%s'% config['name']
    self.fig_dir = '%s/figs/'% self.eval_dir
    self.record_best_pl_file = '%s/best_pl.json'% self.eval_dir
    self.tb_log_freq = 10

    self.eval_freq = config['eval_freq']
    self.restore_from = config['restore_from']
    self.save_freq = config['save_freq']
    self.m_save_path = 'models/tf-models/%s'% config['name']
    self.best_runtimes = []
    self.n_max_best_runtimes = 5
    self.record_pl_write_freq = 10
    self.ep2pl = {}
    self.debug_verbose = config['debug_verbose']
    self.dont_share_classifier = config['dont_share_classifier']
    self.eval_on_transfer = config['eval_on_transfer']
    self.dont_repeat_ff = config['dont_repeat_ff']
    self.log_tb_workers = config['log_tb_workers']
    self.gen_profile_timeline = config['gen_profile_timeline']
    self.profiling_chrome_trace = 'models/chrome-traces/%s/'% config['name']
    self.node_traversal_order = config['node_traversal_order']
    self.cache_eval_plts = deque(maxlen=5)
    self.pptfitem = pptfitem

    if self.gen_profile_timeline:
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    assert(config['turn_based_baseline'])
    
    if config['one_shot_episodic_rew']:
        assert config['zero_placement_init']
        assert not config['use_min_runtime']

    self.async_sim = (config['n_async_sims'] is not None) or (config['remote_async_addrs'] is not None)
    
    if len(config['name']) > 0:
      for d in [self.tb_dir]:
        rmdir(d)

    for d in [self.tb_dir, self.fig_dir, self.eval_dir, self.m_save_path, \
              self.profiling_chrome_trace]:
      mkdir_if_not_exists(d)

    self.log_config(config)
    
    pgs = self.pgs = [ProgressiveGraph(G, self.n_devs, self.node_traversal_order, seed=self.seed) for _ in range(self.bs)]

    if self.async_sim:
        self.setup_async_sim(config)
        
    if self.max_rnds is None:
      self.max_rnds = self.pgs[0].n_nodes()

    self.model = nn_model(self.pgs[0].get_emb_size(), 
                               self.pgs[0].n_nodes(), 
                               self.n_devs,
                               self.pgs[0],
                               config=config,
                               debug=config['debug'],
                               normalize_aggs=config['normalize_aggs'],
                               bn_pre_classifier=config['bn_pre_classifier'],
                               small_nn=config['small_nn'],
                               no_msg_passing=config['no_msg_passing'],
                               radial_mp=config['radial_mp'],
                               bs=self.bs)

    supervised = self.model.is_supervised
    rnd_cum_rewards = [deque(maxlen=config['bl_n_rnds']) for _ in range(self.max_rnds)]

    if not self.summ_send_q or self.log_tb_workers:
        self.tb_writer = tf.summary.FileWriter(self.tb_dir, flush_secs=30)
        summ_names = ['run_times/episode_end_rt', 'run_times/best_so_far', 'run_times/best_rew_rt', 'ent/tanhc_const', 'run_times/ep_best_rt']

        if not config['dont_sim_mem']:
          summ_names += ['run_times/rew_rt', 'mem/mem_util', 'mem/best_mem_util_so_far']

        if config['supervised']:
          summ_names += ['loss/loss', 'opt/logits', 'opt/lr']
        else:
          summ_names += ['rew/reward', 'loss/loss', 'rew/baseline', 'rew/advantage', 'loss/log_probs',\
                  'opt/lr', 'ent/entropy', 'opt/grad_norm', 'ent/ent_dec', 'opt/pre_sync_grad_norm']

        self.summ_writer = CustomTBSummary(self.tb_writer, summ_names)
        self.eval_writer = CustomTBSummary(self.tb_writer, ['run_times/argmax_ep_best_rt'])

    eval_summ_data = []

    self.save_saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2)

    vs = tf_ops.get_collection(tf_ops.GraphKeys.TRAINABLE_VARIABLES)

    if config['dont_restore_softmax']:
        vs = filter(lambda k: 'classifier' not in k.name, vs)
        vs = list(vs)
        self.restore_saver = tf.train.Saver(vs)
    else:
        self.restore_saver = tf.train.Saver()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True

    with tf.Session(config=sess_config) as sess:

      self.initialize_weights(sess, config['dont_restore_softmax'])

      best_rew_rt = 1e9

      zero_placement = self.pgs[0].get_zero_placement()
      rand_placement = self.pgs[0].get_random_placement(seed=0)
      null_placement = self.pgs[0].get_null_placement()

      disamb_nodes = []
      if config['disamb_pl']:
        assert False
        n = self.pgs[0].n_nodes()
        disamb_nodes = ['0', '1', str(int((n/2)-1)), str(n-1)]
        for i in disamb_nodes:
          rand_placement[str(i)] = 0

      nodes = [n for n in self.pgs[0].nodes() if n not in disamb_nodes]

      for ep in range(n_episodes):

        if config['vary_init_state']:
          init_pl = self.pgs[0].get_random_placement(seed=ep)
        elif config['init_best_pl']:
          init_pl = best_pl
        elif config['zero_placement_init']:
          init_pl = zero_placement
        elif config['null_placement_init']:
          init_pl = null_placement
        else:
          init_pl = rand_placement

        self.init_pl = init_pl

        if self.eval_on_transfer is not None:
          is_eval_on_transfer = (ep == self.eval_on_transfer)
        else:
          is_eval_on_transfer = False
        is_eval_ep = (ep % self.eval_freq == 0)
        is_eval_ep = is_eval_ep or is_eval_on_transfer
        is_save_ep = (ep % self.save_freq == 0 and ep > 0)

        if (1 + ep) % self.print_freq == 0:
          print('Episode starting pl',  self.pgs[0].get_placement())
          print("Episode #:", ep)
          print("Nodes: ", list2str(nodes))

        _, epbest_run_time, epbest_mem_util, run_times, mem_utils, states, explor_acts, pls =\
              self.run_episode(sess, init_pl, nodes, is_eval_ep, ep, config)

        batched_cum_rewards = []
        rew_rts = []
        for i, rt in enumerate(run_times):
          cum_rewards, rew_rt = self.compute_rewards(copy.deepcopy(rt), mem_utils[i], config)
          batched_cum_rewards.append(cum_rewards)
          rew_rts.append(rew_rt)

        ep_best_pl, epbest_rew_rt, epbest_pl_rt, epbest_pl_mem = self.identify_best_pl(rew_rts, run_times, mem_utils, pls)

        if epbest_rew_rt < best_rew_rt and not is_eval_ep:
          best_rew_rt = epbest_rew_rt
          best_pl = ep_best_pl
          best_rt = epbest_pl_rt
          best_mem = epbest_pl_mem

        self.update_best_pl(epbest_rew_rt, ep_best_pl, ep, is_eval_ep)
        
        if ep % self.record_pl_write_freq == 0:
              self.record_pl(ep)

        if is_eval_ep:
          eval_summ_data += [({'run_times/argmax_ep_best_rt': epbest_pl_rt}, ep)]
          self.plot_eval_stats(ep, run_times, mem_utils, rew_rts, init_pl, ep_best_pl)
          if is_save_ep:
            self.possibly_save_model(sess, ep)

          if is_eval_on_transfer:
            print('Is eval_on_transfer runtimes:')
            print(run_times)
            self.send_is_eval_on_transfer(config['br_send_q'],
                                                    run_times)
            return

          continue

        batched_cum_rewards = np.float32(batched_cum_rewards)
        avg_cum_rewards = np.mean(batched_cum_rewards, axis=0)
      
        for i in range(len(avg_cum_rewards)):
            rnd_cum_rewards[i].append(avg_cum_rewards[i])

        baselines = self.get_baseline(rnd_cum_rewards, config)

        if supervised:
          summ = self.improve_supervised_placement(sess, states, batched_cum_rewards, baselines, ep)
        else:
          summ = self.improve_placement(sess, states, batched_cum_rewards, baselines, ep)

        if is_save_ep:
          self.possibly_save_model(sess, ep)

        if (ep < 100 and ep % 10 == 0) or ep % 1000 == 0:
          self.plot_rnd_run_times(ep, rnd_cum_rewards)

        if ep % self.tb_log_freq == 0 or len(summ) > 0:
            d = {'run_times/episode_end_rt': float(np.average([rt[-1] for rt in run_times])),
                 'run_times/ep_best_rt': epbest_pl_rt,
                 'run_times/best_so_far': best_rt,
                 'run_times/best_rew_rt': best_rew_rt}

            if not config['dont_sim_mem']:
                d['mem/mem_util'] = epbest_pl_mem / 1e9
                d['mem/best_mem_util_so_far'] = best_mem / 1e9
                d['run_times/rew_rt'] = epbest_rew_rt

            self.tb_write({**d, **summ}, ep)

            for d, ep in eval_summ_data:
                self.tb_write(d, ep, True)

            eval_summ_data = []

    if self.gen_profile_timeline:
        self.save_profiling_info()
        
  def save_profiling_info(self):
    from tensorflow.python.client import timeline

    fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(self.profiling_chrome_trace + '/chrome_trace.trace', 'w') as f:
        f.write(chrome_trace)

  def tb_write(self, d, ep, eval=False):
    if self.summ_send_q:
      self.summ_send_q.put([d, ep, eval])
      assert self.summ_recv_q.get()

    if not self.summ_send_q or self.log_tb_workers:
      if eval:
        self.eval_writer.write(d, ep)
      else:
        self.summ_writer.write(d, ep)

  def setup_async_sim(self, config):

      self.async_send_pls_q = []
      self.async_recv_pls_q = []

      if config['n_async_sims']:

        for i in range(config['n_async_sims']):
          self.async_send_pls_q.append(mp.Queue(1000))
          self.async_recv_pls_q.append(mp.Queue(1000))

          d = {'id': i, 
                'recv_q': self.async_send_pls_q[-1], 
                'send_q': self.async_recv_pls_q[-1], 
                'G': self.pgs[0].get_G()}
          mp.Process(target=self.async_process_func, args=(d,)).start()


      if config['remote_async_addrs'] is not None:

        from remote_async_sim_q import RemoteAsyncSimClient
        import queue

        self.remote_async_addrs = config['remote_async_addrs']
        self.remote_async_start_ports = config['remote_async_start_ports']
        self.remote_async_n_sims = config['remote_async_n_sims']

        assert len(self.pgs) == 1

        self.remote_async_send_pls_q = []
        self.remote_async_recv_pls_q = []

        for i, (addr, start_port) in enumerate(zip(self.remote_async_addrs, self.remote_async_start_ports)):

          for j in range(self.remote_async_n_sims[i]):

            port = start_port + j

            ungroup_mapping = self.pptfitem.get_ungroup_map()

            pkl_file = config['pickled_inp_file'][0]
            if config['remote_prefix'] is not None:
                pkl_file = config['remote_prefix'] + '/' + pkl_file

            register_args = [pkl_file, self.n_devs, True, ungroup_mapping]

            cl = RemoteAsyncSimClient(addr, port, register_args)

            send_q = queue.Queue(1000)
            self.async_send_pls_q.append(send_q)
            self.remote_async_send_pls_q.append(send_q)

            recv_q = queue.Queue(1000)
            self.async_recv_pls_q.append(recv_q)
            self.remote_async_recv_pls_q.append(recv_q)

            d = {'id': len(self.remote_async_send_pls_q) - 1,
                'send_q': recv_q,
                'recv_q': send_q,
                'remote_async_sim_client': cl}

            threading.Thread(target=self.remote_async_thread_func, args=(d,)).start()

      self.n_async_sims = len(self.async_send_pls_q)
      assert self.n_async_sims == len(self.async_recv_pls_q)


  def send_is_eval_on_transfer(self, q, rs):
    zero_rt, _, mem_util = self.eval_placement(self.pgs[0].get_zero_placement())
    init_rt, _, mem_util = self.eval_placement(self.init_pl)
    q.put([zero_rt, init_rt, rs])


  def initialize_weights(self, sess, dont_restore_softmax):
    if self.restore_from is not None:
        if dont_restore_softmax:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        self.restore_saver.restore(sess, self.restore_from)
        print('Model successfully restored!')
    else:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())

    if self.restore_from is None or dont_restore_softmax:
        if self.params_send_q is not None:
          x = self.get_vars(sess)
          self.params_send_q.put(x) 

        if self.params_recv_q is not None:
          var_vals = self.params_recv_q.get()
          self.set_vars(sess, var_vals)

        if self.params_recv_q is not None or \
           self.params_send_q is not None:
          print('Agent initialization sequence finished')

    sess.run(self.model.init_global_step)


  def get_vars(self, sess):
    return self.model.get_vars(sess, not self.dont_share_classifier)

  def set_vars(self, sess, var_vals):
    self.model.set_vars(sess, var_vals, not self.dont_share_classifier)

  def possibly_save_model(self, sess, ep):
    save = True
    if self.params_send_q is not None:
      if self.id > 0:
        save = False

    if save:
      self.save_model(sess, ep)

  def save_model(self, sess, ep):
    self.save_saver.save(sess, self.m_save_path, global_step=ep, write_meta_graph=False)

  def update_best_pl(self, epbest_rew_rt, epbest_pl, ep, is_eval_ep):
    if len(self.best_runtimes) < self.n_max_best_runtimes:
      f = lambda *args: (heappush(*args), None)
    elif -self.best_runtimes[0][0] > epbest_rew_rt:
      f = heappushpop
    else:
      return False

    _, del_ep = f(self.best_runtimes, (-epbest_rew_rt, ep))
    if del_ep is not None:
      self.ep2pl.pop(del_ep, None)

    self.ep2pl[ep] = [epbest_pl, is_eval_ep]

  def record_pl(self, ep):
      j = []
      for (r, ep) in self.best_runtimes:
        r = -r
        # make things json serializable
        pl, is_eval_ep = self.ep2pl[ep]
        for k, v in pl.items():
          pl[k] = int(v)
        j.append({'runtime': r, 
                  'pl': pl, 
                  'ep': int(ep), 
                  'is_eval_ep': int(is_eval_ep)})

      with open(self.record_best_pl_file, 'w') as f:
        json.dump(j, f)


  def run_episode(self, sess, init_pl, nodes, is_eval_ep, ep, config):

    for pg in self.pgs:
        pg.reset_placement(init_pl)
        pg.new_episode()

    start_times = [np.array([[-1]* pg.n_nodes()]) for pg in self.pgs]

    if config['one_shot_episodic_rew']:
      run_time = [math.inf]* self.bs
      mem_util = [[math.inf]* self.n_devs]* self.bs
    else:
      if self.async_sim:
        run_time, _, mem_util = self.eval_placement()
      else:
        run_time, start_times, mem_util = self.eval_placement()

      for i, pg in enumerate(self.pgs):
        pg.set_start_times(start_times[i])
    
    if (1 + ep) % self.print_freq == 0:
      print(run_time, end=' ')
      print(list2str([init_pl[n] for n in nodes]))

    episode_best_time = min(run_time)
    episode_best_pl_mem = mem_util[run_time.index(min(run_time))]
    episode_best_pl = init_pl
    run_times = []
    mem_utils = []
    states = []
    explor_acts = []
    async_record = []
    pls = [[init_pl] for _ in range(self.bs)]

    run_times.append(run_time)
    mem_utils.append(mem_util)

    nn_time = 0
    s1 = time()
    for i in range(self.max_rnds):
      is_last_rnd = (i == self.max_rnds - 1)
      node = nodes[i% len(nodes)]
      for pg in self.pgs: pg.set_cur_node(node)

      s2 = time()
      d, lo, feed, expl, train_outs = self.get_improvement(
                            sess, node, start_times, is_eval_ep)
      nn_time += (time() - s2)

      explor_acts.append(expl)
      for j, pg in enumerate(self.pgs):
        pg.refresh([node], [d[j]])

      for j, pg in enumerate(self.pgs):
          pls[j].append(pg.get_placement())

      if not config['one_shot_episodic_rew'] or is_last_rnd:
        if self.async_sim:
          j = i % self.n_async_sims
          self.eval_placement(async=j)
          async_record.append(j)
        else:
          run_time, start_times, mem_util = self.eval_placement()
          for st, pg in zip(start_times, self.pgs):
              pg.update_start_times(st)

      # add infs if one shot
      if not self.async_sim or config['one_shot_episodic_rew']:
        run_times.append(run_time)
        mem_utils.append(mem_util)

      states.append([feed, d, lo, train_outs])

      for pg in self.pgs:
          pg.inc_done_node(node)

    if self.async_sim:
      for j in async_record:
        run_time, mem_util = self.eval_placement(retreive=j)
        run_times.append(run_time)
        mem_utils.append(mem_util)

    for i, rnd_rt in enumerate(run_times):
      for j, rt in enumerate(rnd_rt):
          if episode_best_time > rt:
            episode_best_time = rt
            episode_best_pl_mem = mem_utils[i][j]
            episode_best_pl = pls[j][i]

    run_times = np.transpose(run_times)
    mem_utils = np.array(mem_utils)
    mem_utils = mem_utils.transpose(1, 0, 2)

    if ep < 20:
      print('Total time: ', time() - s1)
      print('NN time: ', nn_time)

    return episode_best_pl, episode_best_time, episode_best_pl_mem, run_times, mem_utils, states, explor_acts, pls

  def get_baseline(self, rnd_cum_rewards, config):
    baselines = np.zeros(len(rnd_cum_rewards))
    
    if config['turn_based_baseline']:
      for i in range(len(rnd_cum_rewards)):
        baselines[i] = np.mean(rnd_cum_rewards[i], dtype=np.float64)

    if self.send_baseline_q:
      if self.debug_verbose:
          print('Agent %d submitting baseline '% self.id)
      self.send_baseline_q.put(baselines)
      if self.debug_verbose:
          print('Agent %d getting baseline '% self.id)
      baselines = self.recv_baseline_q.get()

    return baselines


  def compute_rewards(self, run_times, mem_utils, config):

    if config['log_runtime']:
      run_times = np.log(run_times)

    if config['mem_penalty'] > 0:
      for i, mem_util in enumerate(mem_utils):
        mem_excess = max(mem_util)/1e9 - config['max_mem']
        mem_excess = max(0, mem_excess)
        run_times[i] += config['mem_penalty'] * mem_excess
        run_times[i] = min(run_times[i], config['max_runtime_mem_penalized'])
        # print(mem_excess, max(mem_util) / 1e9, config['mem_penalty']* mem_excess, run_times[i])

    if config['use_min_runtime']:
      cum_rewards = list(run_times[0: len(run_times) - 1])

      for i in range(len(cum_rewards)):
        cum_rewards[i] -= min(run_times[i + 1:])

    elif config['one_shot_episodic_rew']:
      r = run_times[-1]
      cum_rewards = [-1.* r]* (len(run_times) - 1)

    else:
      cum_rewards = []
      for i in range(len(run_times) - 1):
        cum_rewards.append([j - i for i, j in zip(run_times[i + 1], run_times[i])])

      for i in reversed(range(len(cum_rewards)-1)):
        cum_rewards[i] += (self.disc_fact* cum_rewards[i+1])

    return cum_rewards, run_times


  def identify_best_pl(self, rew_rts, run_times, mem_utils, pls):

      best_rew_rt = 1e20
      best_pl = None
      best_pl_rts = []
      best_pl_mems = []

      for i, rew_rt in enumerate(rew_rts):
          j = np.argmin(rew_rt)
          if best_rew_rt > rew_rt[j]:
            best_rew_rt = rew_rt[j]
            best_pl = pls[i][j]
            best_pl_rt = run_times[i][j]
            best_pl_mem = max(mem_utils[i][j])

      return best_pl, best_rew_rt, best_pl_rt, best_pl_mem


  def plot_rnd_run_times(self, ep, rnd_cum_rewards):
    
    def get_means_stds(ls):
      means = [np.mean(l) for l in ls]
      stds = [np.std(l) for l in ls]
      return means, stds

    plt.figure('rnd_run_times')
    means, stds = get_means_stds(rnd_cum_rewards)
    plt.errorbar(range(len(means)), means, stds,
                                  fmt='o', ecolor='g') 
    plt.xlabel('round #')
    plt.ylabel('cum_reward ')
    plt.savefig('%s/%d.pdf' % (self.fig_dir, ep))
    plt.clf()

  def log_config(self, config):
    jsonable_config = {}
    for k, v in config.items():
      if type(v).__name__ != 'Queue':
        jsonable_config[k] = v

    with open('%s/config.txt'% (self.eval_dir), 'w') as f:
      f.write(' '.join(sys.argv) + '\n')
      f.write('PID: %d\n' % os.getpid())
      json.dump(jsonable_config, f, indent=4, sort_keys=True)

  def plot_eval_stats(self, ep, run_times, mem_utils, rews, init_pl, epbest_pl):

    cache_eval_plts = self.cache_eval_plts
    cache_eval_plts.append([ep, run_times[0]])

    with open('%s/%d.txt'% (self.eval_dir, ep), 'w') as f:
      f.write(json.dumps(init_pl))
      f.write('\nRun times:\n')
      f.write('\n'.join(map(str, run_times[0])))
      f.write('\nmem util:\n')
      f.write('\n'.join(map(str, [max(m) for m in mem_utils[0]])))
      f.write('\nreward:\n')
      f.write('\n'.join(map(str, rews[0])))
      
      for k, v in epbest_pl.items():
        f.write('\n%s: %s'% (k, v))

    plt.figure('eval_stats')
    plt.clf()
    plt.cla()
    for ep, run_times in cache_eval_plts:
      plt.plot(run_times, label=str(ep))

    plt.xlabel('round #')
    plt.ylabel('Runtime')
    plt.legend()
    plt.savefig('%s/eval-stats.pdf'% self.eval_dir)
    

  def eval_placement(self, pl=None, async=None, retreive=None):

    pls = []
    if pl is None:
      for pg in self.pgs:
        pls.append(copy.copy(pg.get_placement()))
    else:
      pls.append(copy.copy(pl))

    if async is not None:
      self.async_send_pls_q[async].put(pls)
      return 

    elif retreive is not None:
      return self.async_recv_pls_q[retreive].get()

    else:

      run_times = []
      start_times = []
      mem_utils = []

      for pg, pl in zip(self.pgs, pls):
          run_time, start_time, mem_util = self.sim_eval(pl, pg.get_G())
          start_time = np.array([start_time[n] for n in pg.nodes()], ndmin=2)

          run_times.append(run_time)
          start_times.append(start_time)
          mem_utils.append(mem_util)

    return run_times, start_times, mem_utils

  def async_process_func(self, config):
    recv_q = config['recv_q']
    send_q = config['send_q']
    G = config['G']
    id = config['id']

    while True:
      run_times = []
      mem_utils = []

      pls = recv_q.get()

      for pl in pls:
        rt, _, mem_util = self.sim_eval(pl, G)
        run_times.append(rt)
        mem_utils.append(mem_util) 

      send_q.put([copy.deepcopy(run_times), copy.deepcopy(mem_utils)])


  def remote_async_thread_func(self, kwargs):
        recv_q = kwargs['recv_q']
        send_q = kwargs['send_q']
        client = kwargs['remote_async_sim_client']

        while True:
            run_times = []
            mem_utils = []

            pls = recv_q.get()

            assert len(pls) == 1

            pl = pls[0]

            send_pl = {k: int(v) for k, v in pl.items()}

            rt, mem_util = client.request(send_pl)

            send_q.put([copy.deepcopy([rt]), copy.deepcopy([mem_util])])

  def get_improvement(self, sess, node, start_times, is_eval_ep):
    model = self.model

    feed = model.get_feed_dict(self.pgs, node, start_times, self.n_peers)
    if is_eval_ep:
      feed[model.is_eval_ph] = 1.0

    train_ops = []
    if self.dont_repeat_ff:
      train_ops = [model.logprob_grad_outs, model.ent_grad_outs,\
                 model.log_probs, model.sample,\
                 model.pl_ent_loss, model.log_prob_loss,\
                 model.no_noise_logits, model.entropy, \
                 model.ent_dec]
    
    kwargs = {}
    if self.gen_profile_timeline:
        kwargs = {'run_metadata': self.run_metadata,
                  'options': self.run_options,}

    s, lo, lp, expl, *train_outs = sess.run(model.get_eval_ops() +\
                                      [model.expl_act] +\
                                      train_ops,
                                      feed_dict=feed, **kwargs)

    return s, lo, feed, expl, train_outs

  def improve_supervised_placement(self, sess, states, cum_rewards, baselines, ep):
    model = self.model

    for i, [feed, ln, sample, lo] in enumerate(states):
      _, loss, lot, lr = sess.run(
            [model.train_op, model.loss, model.logits, model.lr], feed_dict=feed)

      if (1 + ep) % self.print_freq == 0:
        print('loss, logits: ', loss, lot)

    return {}

  def improve_placement(self, sess, states, cum_rewards, baselines, ep):
    
    model = self.model
    batched_feed = {}
    sum_grads = None
    rews, losses, bls, advs, tlps, lrs, ents, ent_decs, tanhcs \
                              = [], [], [], [], [], [], [], [], []

    for rnd, [feed, sample, lo, train_outs] in enumerate(states):
      if self.discard_last_rnds:
        if len(states) - rnd < self.pgs[0].n_nodes():
          break

      rew = cum_rewards[:, rnd]
      bl = baselines[rnd]

      if self.dont_repeat_ff:
        lp_grads, ent_grads, \
          tlp, ts, pl_ent_loss, log_prob_loss, nnl, ent, ent_dec = train_outs

        lr = sess.run(model.lr)
        if model.tanhc_decay_func is not None:
          tanhc = sess.run(model.tanhc_decay_func)

        adv = rew - bl

        logi_grad = [lp_grads[0]* adv[:, None] + ent_grads[0]]
        grads = []
        for i in range(self.bs):
          for j, (g1, g2) in enumerate(zip(lp_grads[i], ent_grads[i])):
            g1 = np.float64(g1)
            g2 = np.float64(g2)
            g = (adv[i]* g1) + g2
            if i == 0:
              grads.append(g)
            else:
              grads[j] += g

        loss = log_prob_loss* adv + pl_ent_loss
        
        # for pg in self.pgs: print(pg.get_placement())
        # test_loss, test_grad_outs = \
        #   sess.run([model.loss, model.grad_outs], 
        #             feed_dict={**feed, **model.get_train_specific_fd(rew, bl, ln, sample)})

        # if np.any(np.abs(loss - test_loss) > 1e-3):
        #   print(loss, test_loss, adv)
        #   import pdb; pdb.set_trace()
        #   ...

        # if (1 + ep) % self.print_freq == 0:
        #   print(np.mean(np.abs(test_grad_outs[0])))

        # for i in range(len(grads)):
        #   if np.any(np.abs((test_grad_outs[i] - grads[i])/(1+grads[i])) > 1e-3):
        #     print(grads[i], test_grad_outs[i], adv)
        #     import pdb; pdb.set_trace()
        #     ...

      else:
        raise Exception('Dont repeat ff option locked')
        i = rnd
        tlp, ts, loss, rew, bl, adv, nnl, grads, ent, lr, logi_grad, ent_dec, rgn, egn, \
            test_lp_loss, pl_ent_loss, logprob_grad_outs, ent_grad_outs \
           = sess.run(
            [model.train_log_probs, model.train_sample, \
             model.loss, model.rew, model.baseline, model.adv, \
             model.no_noise_logits, model.grad_outs, \
             model.entropy, model.lr, model.logits_train_grad, model.ent_dec, \
             model.reinforce_grad_norm, model.entropy_grad_norm, \
             model.test_log_prob_loss, model.pl_ent_loss,
             model.logprob_grad_outs, model.ent_grad_outs],
            feed_dict={**feed, **model.get_train_specific_fd(rew, bl, sample)})

        assert np.all(ts == sample)
        # print(test_lp_loss, pl_ent_loss, adv, loss)
        # print(logprob_grad_outs[0] * adv, ent_grad_outs[0], grads[0])
        assert test_lp_loss* adv + pl_ent_loss == loss
        if not np.all(np.abs(logprob_grad_outs[0] * adv + ent_grad_outs[0] - grads[0]) < 1e-3):
          print(test_lp_loss, pl_ent_loss, adv, loss)
          print(logprob_grad_outs[0] * adv, ent_grad_outs[0], grads[0])
          import pdb; pdb.set_trace()
          ...

      if (1 + ep) % self.print_freq == 0:
        print('loss: %.2f, rew: %.2f, bl: %.2f, adv: %.2f'% (average(loss), average(rew), bl, average(adv)))
        # if len(ts) <= 1:
        #   print(', action: %d, logits: %s , logits_grad: %s'% \
        #         (ts, fl2str(lo), fl2str(logi_grad)))
        # else:
        #   print('')
        #   for i, s in enumerate(ts):
        #     print('action: %s, logits: %s, logits_grad: %s'%\
        #           (ts[i], fl2str(lo[i, :]), fl2str(logi_grad[0][i, :])))

      if sum_grads:
        for i, g in enumerate(grads):
          sum_grads[i] += g
      else:
        sum_grads = []
        for g in grads:
          sum_grads.append(np.float64(g))

      if ep % self.tb_log_freq == 0:
        rews.append(rew)
        losses.append(loss)
        bls.append(bl)
        advs.append(adv)
        tlps.append(tlp)
        lrs.append(lr)
        ents.append(ent)
        ent_decs.append(ent_dec)
        if model.tanhc_decay_func is not None:
          tanhcs.append(tanhc)

    if ep % self.tb_log_freq == 0:
      a_rew = average(rews)
      a_loss = average(losses)
      a_bl = average(bls)
      a_adv = average(advs)
      a_tlp = average(tlps)
      a_ent = average(ents)
      a_lr = average(lrs)
      a_entdec = average(ent_decs)
      if len(tanhcs) > 0:
        a_tanhc = average(tanhcs)

    d = model.get_apply_grad_feed(sum_grads)

    pre_gn = sess.run(model.grad_norm, feed_dict=d)

    if self.grads_send_q is not None:
      if self.debug_verbose:
        print('Agent %d Submitting Grads'% self.id)
        sys.stdout.flush()
      send_d = {}
      for k, v in d.items():
        if self.dont_share_classifier and 'classifier' in k.name:
          continue
        send_d[k.name] = v
      self.grads_send_q.put(send_d)
      recv_d = self.grads_recv_q.get()

      d = {}
      for k, v in recv_d.items():
        d[k] = v

    gn, _, global_step = sess.run([model.grad_norm, model.train_op, model.global_step], 
                                                                            feed_dict=d)

    if self.params_send_q is not None:
      self.params_send_q.put(self.get_vars(sess)) 
      var_vals = self.params_recv_q.get()
      if var_vals is not None:
        self.set_vars(sess, var_vals)

    if self.debug_verbose:
      print('Episode sequence finished for agent %d'% self.id)
      sys.stdout.flush()

    summ = {}
    if ep % self.tb_log_freq == 0:
      summ = {'rew/reward': a_rew, 'loss/loss': a_loss, 'rew/baseline': a_bl, 
              'rew/advantage': a_adv, 'loss/log_probs': a_tlp, 'opt/lr': a_lr, 'ent/entropy': a_ent,
              'opt/grad_norm': gn, 'ent/ent_dec': a_entdec, 'opt/pre_sync_grad_norm': pre_gn,}

      if len(tanhcs) > 0:
        summ['ent/tanhc_const'] = a_tanhc

    return summ

  def set_seeds(self, i = 0):
    if i is None:
      i = 0
    s = 42 + i
    np.random.seed(s)
    tf.set_random_seed(s)
    random.seed(s)
