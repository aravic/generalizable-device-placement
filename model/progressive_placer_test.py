import os
import networkx as nx
import numpy as np
import tensorflow as tf
import sys
import pickle
from progressive_placer import *
sys.path.append('./')
sys.path.append('progressive_placers/')
sys.path.append('sim/')
import argparse
import rl_params
from itertools import chain
from local_progressive_nn import LocalProgressiveNN
from mp_progressive_nn import MessagePassingProgressiveNN, MessagePassingOneRewardNN
from simple_nn import *
from simple_graphs import *
from pp_item import *
from utils import *


class ProgressivePlacerTest(object):

  @staticmethod
  def sim_reward(n_devs, sim, p, G):
    # Penalty for one unit of outshape transfer
    l_fact = 1

    costs = nx.get_node_attributes(G, 'cost')
    op_memories = nx.get_node_attributes(G, 'out_size')

    run_time, dft, du, tom, nt, start_times = sim.get_runtime(G, n_devs,
                                              costs, op_memories, p)

    assert tom >= 0
    run_time += l_fact* tom

    return run_time, start_times

  '''
  place everything on the last gpu (id: n_devs-1)
  '''
  @staticmethod
  def sim_single_gpu(n_devs, sim, p, G):
    start_times = {}
    for n in G.nodes():
      start_times[n] = 0.
    run_time = 0
    for _, d in p.items():
      if d != n_devs-1:
        run_time += 1

    return run_time, start_times

  @staticmethod
  def sim_neigh_placement(n_devs, sim, p, G):
    start_times = {}
    for n in G.nodes():
      start_times[n] = 0.
    run_time = 0
    for n, d in p.items():
      for neigh in chain(G.neighbors(n), G.predecessors(n)):
        if p[neigh] != p[n]:
          run_time += 1

    return run_time, start_times

  @staticmethod
  def choose_model(model_name):

    if model_name == 'supervised':
      nn_model = SupervisedSimpleNN
    elif model_name == 'simple_nn':
      nn_model = SimpleNN
    elif model_name == 'local_nn':
      nn_model = LocalProgressiveNN
    elif model_name == 'mp_nn':
      nn_model = MessagePassingProgressiveNN
    elif model_name == 'or':
      nn_model = MessagePassingOneRewardNN
    else:
      raise Exception('%s not implemented model'% model_name)

    return nn_model

  def test(self, config):
    graph = config['graph']
    N = config['graph_size']
    n_devs = config['n_devs']
    m_name = config['m_name']
    f = None
    sim = None
    if graph in ['chain', 'crown', 'edge']:
      from old_simulator import LegacySimulator
      sim = LegacySimulator(None, False, n_devs=n_devs, override_ban=True)
    if graph == 'chain':
      G = makeChainGraph(N, n_devs)
    elif graph == 'crown':
      G = makeCrownGraph(N, n_devs)
    elif graph == 'edge':
      G = makeEdgeGraph(N)
    else:
      inp_file = config['pickled_inp_file'][0]
      if config['local_prefix'] is not None:
        inp_file = config['local_prefix'] + '/' + config['pickled_inp_file'][0]

      pptf = PPTFItem(inp_file, n_devs,
          simplify_tf_rew_model = config['simplify_tf_rew_model'],
          use_new_sim = config['use_new_sim'],
          sim_mem_usage = True,
          final_size = config['prune_final_size'])

      if not config['simplify_tf_rew_model']:
        f = lambda _, __, p, ___: pptf.simulate(p)
      G = pptf.get_grouped_graph()

    if not f:
      if config['rew_singlegpu']:
        f = ProgressivePlacerTest.sim_single_gpu
      elif config['rew_neigh_pl']:
        f = ProgressivePlacerTest.sim_neigh_placement
      else:
        f = ProgressivePlacerTest.sim_reward

    if config['eval'] is not None:
      _, r, ss, p = pptf.eval(config['eval'])

      fname = 'models/chrome-traces/%s/timeline.json' % (config['name'])
      timeline_to_json(ss, p, fname)
    else:
      ProgressivePlacer().place(
        G, n_devs, ProgressivePlacerTest.choose_model(m_name),
        lambda *args, **kwargs: f(n_devs, sim, *args, **kwargs),
        config, pptf)

  def mul_graphs(self, config):
    from coord import Coordinator
    Coordinator().start(config, self.test)

  def benchmark_policy(self, config):
    from policy_benchmarker import PolicyBenchmarker
    PolicyBenchmarker().start(config, self.test)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--name', '-n', type=str, default='test')
  parser.add_argument('--graph', '-g', type=str, default=None)
  parser.add_argument('--id', type=int, default=None)
  # for synthetic non-tensorflow graphs
  parser.add_argument('--graph-size', '-N', type=int, default=4)
  parser.add_argument('--pickled-inp-file', '-i', type=str, default=None, nargs='+')
  parser.add_argument('--mul-graphs', type=str, default=None, nargs='+')

  parser.add_argument('--n-devs', type=int, default=2)

  # progressive placer model args
  parser.add_argument('--m-name', type=str, default='mp_nn')
  parser.add_argument('--n-peers', type=int, default=None)
  parser.add_argument('--agg-msgs', dest='agg_msgs', action='store_true')
  parser.add_argument('--no-msg-passing', action='store_true', dest='no_msg_passing')
  parser.add_argument('--radial-mp', type=int, default=None)
  parser.add_argument('--tri-agg', dest='tri_agg', action='store_true')

  # training args
  parser.add_argument('--n-eps', type=int, default=int(1e9))
  parser.add_argument('--max-rnds', type=int, default=None)
  parser.add_argument('--disc-factor', type=float, default=1.)
  parser.add_argument('--vary-init-state', dest='vary_init_state', action='store_true')
  parser.add_argument('--zero-placement-init', dest='zero_placement_init', action='store_true')
  parser.add_argument('--null-placement-init', dest='null_placement_init', action='store_true')
  parser.add_argument('--init-best-pl', dest='init_best_pl', action='store_true')
  parser.add_argument('--one-shot-episodic-rew', dest='one_shot_episodic_rew', action='store_true')
  parser.add_argument('--ep-decay-start', type=float, default=1e3)
  parser.add_argument('--bl-n-rnds', type=int, default=1000)
  parser.add_argument('--rew-singlegpu', dest='rew_singlegpu', action='store_true')
  parser.add_argument('--rew-neigh-pl', dest='rew_neigh_pl', action='store_true')
  parser.add_argument('--supervised', dest='supervised', action='store_true')
  parser.add_argument('--use-min-runtime', dest='use_min_runtime', action='store_true')
  parser.add_argument('--discard-last-rnds', dest='discard_last_rnds', action='store_true')
  parser.add_argument('--turn-based-baseline', dest='turn_based_baseline', action='store_true')
  parser.add_argument('--dont-repeat-ff', action='store_true', dest='dont_repeat_ff')
  parser.add_argument('--small-nn', action='store_true', dest='small_nn')
  parser.add_argument('--dont-restore-softmax', dest='dont_restore_softmax', action='store_true')
  parser.add_argument('--restore-from', type=str, default=None)

  # report/log args
  parser.add_argument('--print-freq', type=int, default=50)
  parser.add_argument('--save-freq', type=int, default=100)
  parser.add_argument('--eval-freq', type=int, default=999)
  parser.add_argument('--log-tb-workers', dest='log_tb_workers', action='store_true')
  parser.add_argument('--debug', dest='debug', action='store_true')
  parser.add_argument('--debug-verbose', dest='debug_verbose', action='store_true')
  parser.add_argument('--disamb-pl', dest='disamb_pl', action='store_true')
  parser.add_argument('--eval', type=str, default=None)
  parser.add_argument('--simplify-tf-rew-model', action='store_true', dest='simplify_tf_rew_model')
  parser.add_argument('--log-runtime', dest='log_runtime', action='store_true')
  parser.add_argument('--use-new-sim', action='store_true', dest='use_new_sim')
  parser.add_argument('--gen-profile-timeline', dest='gen_profile_timeline', action='store_true')
  parser.add_argument('--mem-penalty', type=float, default=0.)
  parser.add_argument('--max-mem', type=float, default=11., help='Default Max Memory of GPU (in GB)')
  parser.add_argument('--max-runtime-mem-penalized', type=float, default=10., 
                        help='Instantaneous runtime of the placement after adding the memory penalty has to be lower than this number. Note that improvement in this memory penalized runtime metric is used to compute intermediate rewards')

  # dist training params
  parser.add_argument('--use-threads', dest='use_threads', action='store_true')
  parser.add_argument('--scale-norm', dest='scale_norm', action='store_true')
  parser.add_argument('--dont-share-classifier', action='store_true', dest='dont_share_classifier')
  parser.add_argument('--use-gpus', type=str, nargs='+', default=None)
  parser.add_argument('--eval-on-transfer', type=int, default=None, help='Number of episodes to transfer train before reporting eval runtime')
  parser.add_argument('--normalize-aggs', dest='normalize_aggs', action='store_true')
  parser.add_argument('--bn-pre-classifier', dest='bn_pre_classifier', action='store_true')
  parser.add_argument('--bs', type=int, default=None)
  parser.add_argument('--num-children', type=int, default=1)
  parser.add_argument('--disable-profiling', action='store_true', dest='disable_profiling')
  parser.add_argument('--n-async-sims', type=int, default=None)
  parser.add_argument('--baseline-mask', type=int, nargs='+', default=None)
  parser.add_argument('--n-workers', type=int, default=1)
  parser.add_argument('--node-traversal-order', default='topo', help='Options: topo, random')
  parser.add_argument('--prune-final-size', type=int, default=None)
  parser.add_argument('--dont-sim-mem', dest='dont_sim_mem', action='store_true')

  parser.add_argument('--remote-async-addrs', type=str, default=None, nargs='+')
  parser.add_argument('--remote-async-start-ports', type=int, default=None, nargs='+')
  parser.add_argument('--remote-async-n-sims', type=int, default=None, nargs='+')
  parser.add_argument('--local-prefix', type=str, default=None)
  parser.add_argument('--remote-prefix', type=str, default=None)
  parser.add_argument('--shuffle-gpu-order', dest='shuffle_gpu_order', action='store_true')
  
  args, unknown = parser.parse_known_args()

  assert args.dont_repeat_ff

  if args.one_shot_episodic_rew and args.n_async_sims is not None:
    raise Exception('Input setting leads to deadlock')
  
  if args.eval_freq% 10 == 0:
    print('Eval freq cannot be divisible by 10')
    sys.exit(0)


  for option in unknown:
    for i in range(len(option)):
      if option[i] != '-':
        break
    if i > 0:
      option = option[i:].replace('-', '_')
      if option not in rl_params.args.__dict__:
        print(option)
        # pass
        raise Exception("Passed unknown option in dict : %s" % option)

  if args.use_gpus is not None:
      os.environ['CUDA_VISIBLE_DEVICES'] = ' '.join(args.use_gpus)

  if args.eval_on_transfer is not None:
    ProgressivePlacerTest().benchmark_policy(args.__dict__)
  elif args.n_workers > 1:
    ProgressivePlacerTest().mul_graphs(args.__dict__)
  else:
    ProgressivePlacerTest().test(args.__dict__)
