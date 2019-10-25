import networkx as nx
import sys, random, os, copy, json
import numpy as np 
import global_config as config

''' uses static object n_devs'''
class NodeEmbeddings(object):
  def __init__(self, cost, out_size, mem, placement=None):
    self.cost = cost
    self.out_size = out_size
    self.placement = placement
    self.curr_bit = 0
    self.done_bit = 0
    self.start_time = None
    if config.use_mem_attr:
      self.mem = mem

  def placement_to_one_hot(self, p):
    try:
      ret = [0]* NodeEmbeddings.n_devs
      if p is not None:
        ret[p] = 1
      return ret
    except Exception:
      import pdb; pdb.set_trace()
      print(p)

  def set_curr_bit(self):
    self.curr_bit = 1

  def inc_done_bit(self):
    self.done_bit += 1

  def reset_curr_bit(self):
    self.curr_bit = 0

  def reset_done_bit(self):
    self.done_bit = 0

  def update_start_time(self, t):
    self.start_time = t

  def update_placement(self, new_p):
    self.placement = new_p

  def get_embedding(self):
    n_devs = NodeEmbeddings.n_devs
    l = [self.cost, self.out_size, self.done_bit, self.curr_bit, self.start_time]
           
    if config.use_mem_attr:
      l.append(self.mem)

    l = l + self.placement_to_one_hot(self.placement)

    return l

  @staticmethod
  def get_emb_size():
    if config.use_mem_attr:
      return 6 + NodeEmbeddings.n_devs
    else:
      return 5 + NodeEmbeddings.n_devs

  @staticmethod
  def normalize(E, factors):
    # normalize cost, out_size, start_time
    for i in [0, 1, 4]:
      if factors[i] != 0:
        E[:, i] /= factors[i]

    if config.use_mem_attr:
      if factors[5] != 0:
        E[:, 5] /= factors[5]

    return E

  @staticmethod
  def normalize_start_time(E, cur_node):
    E[:, 4] -= E[cur_node, 4]
    return E

# Graph object to be used for progressive placers.
class ProgressiveGraph(object):

  def __init__(self, G, n_devs, node_traversal_order, seed=42):
    # Networkx graph object with cost attribute set for each node
    random.seed(seed)
    G = copy.deepcopy(G)
    self.G = G
    self.n_devs = n_devs
    NodeEmbeddings.n_devs = n_devs
    self.seed = seed

    if node_traversal_order == 'topo':
        self.node_traversal_order = list(nx.topological_sort(self.G))
    elif node_traversal_order == 'random':
        self.node_traversal_order = list(self.G.nodes())
        random.shuffle(self.node_traversal_order)
    else:
        raise Exception('Node traversal order not specified correctly')

    d = {}
    for i, node in enumerate(self.node_traversal_order):
      d[node] = i
    nx.set_node_attributes(G, d, 'idx')

    for n in self.nodes():
      assert G.node[n]['cost'] is not None
      assert G.node[n]['out_size'] is not None
      assert G.node[n]['mem'] is not None

    self.init_node_embeddings()
    self.init_positional_mats()
    self.init_adj_mat()
    self.init_badj_fadj()

  def init_random_placement(self):
    random.seed(self.seed)
    d = {}
    for i, n in enumerate(self.nodes()):
      d[n] = random.randint(0, self.n_devs-1)
      self.node_embeddings[i].update_placement(d[n])
    nx.set_node_attributes(self.G, d, 'placement')

  def init_zero_placement(self):
    d = self.get_zero_placement()
    for i, n in enumerate(self.nodes()):
      self.node_embeddings[i].update_placement(0)
    nx.set_node_attributes(self.G, d, 'placement')

  def refresh(self, nodes, new_p):
    for p, node in zip(new_p, nodes):
      self.G.node[node]['placement'] = p
      i = self.get_idx(node)
      self.node_embeddings[i].update_placement(p)

  def init_node_embeddings(self):
    E = []
    G = self.G
    for n in self.nodes():
      e = NodeEmbeddings(G.node[n]['cost'], G.node[n]['out_size'], G.node[n]['mem'])
      E.append(e)
    self.node_embeddings = E

  def get_embeddings(self):
    E = []
    for node_emb in self.node_embeddings:
      E.append(node_emb.get_embedding())

    E = np.array(E, dtype=np.float32)
    E = NodeEmbeddings.normalize_start_time(E, self.cur_node)
    self.embeddings = NodeEmbeddings.normalize(E, np.amax(E, axis=0))

    return self.embeddings

  def init_positional_mats(self):
    # finds the shortest path between all nodes
    # referred to as path matrix
    path_mat = nx.floyd_warshall_numpy(self.G, 
                        nodelist=self.nodes())
    peer_mat = np.isinf(path_mat)
    for i in range(len(peer_mat)):
      for j in range(len(peer_mat)):
        if i != j:
          peer_mat[i, j] &= peer_mat[j, i]
        else:
          peer_mat[i, j] = False

    self.peer_mat = peer_mat
    self.progenial_mat = np.logical_not(np.isinf(path_mat))
    np.fill_diagonal(self.progenial_mat, 0)
    self.ancestral_mat = self.progenial_mat.T

  def init_adj_mat(self):
    self.adj_mat = nx.to_numpy_matrix(self.G, nodelist=self.nodes())
    self.undirected_adj_mat = np.array(self.adj_mat)

    for i in range(len(self.adj_mat)):
      for j in range(len(self.adj_mat)):
        self.undirected_adj_mat[i, j] = max(self.undirected_adj_mat[i, j],
                                            self.undirected_adj_mat[j, i])

  def init_badj_fadj(self):
    self.badj = np.float32(nx.to_numpy_matrix(self.G, self.nodes()))
    self.fadj = self.badj.transpose()

  def get_neighbor_mask(self, node):
    return np.expand_dims(self.undirected_adj_mat[self.G.node[node]['idx'], :], axis=0)

  def get_self_mask(self, node):
    m = np.zeros((1, len(self.nodes())))
    m[:, self.get_idx(node)] = 1.
    return m

  def set_cur_node(self, node):
    for e in self.node_embeddings:
      e.reset_curr_bit()

    i = self.get_idx(node)
    self.node_embeddings[i].set_curr_bit()
    self.cur_node = i

  def inc_done_node(self, node):
    i = self.get_idx(node)
    self.node_embeddings[i].inc_done_bit()

  def new_episode(self):
    for e in self.node_embeddings:
      e.reset_curr_bit()
      e.reset_done_bit()

  def get_peer_mask(self, node, start_times, n_peers):
    i = self.G.node[node]['idx']
    start_times = np.abs(start_times - start_times[:, i])
    start_times += (np.logical_not(self.peer_mat[i, :])* int(1e9))
    if n_peers:
      peer_idx = np.argpartition(-start_times[0], -n_peers)[-n_peers:]
    else:
      peer_idx = range(0, self.n_nodes())
    peer_idx = filter(lambda i: start_times[:, i] < int(1e9), peer_idx)
    peer_mask = np.zeros_like(start_times)
    for peer in peer_idx:
      peer_mask[:, peer] = 1.
    return peer_mask

  # get all immediate parents and ancestors
  def get_ancestral_mask(self, node):
    return self.ancestral_mat[self.get_idx(node), :]

  def get_progenial_mask(self, node):
    return self.progenial_mat[self.get_idx(node), :]

  def get_placement(self):
    return nx.get_node_attributes(self.G, 'placement')

  def reset_placement(self, pl):
    for i, n in enumerate(self.nodes()):
      self.node_embeddings[i].update_placement(pl[n])
    nx.set_node_attributes(self.G, pl, 'placement')

  def get_badj(self):
    return self.badj

  def get_fadj(self):
    return self.fadj


  def update_start_times(self, start_times):
    for i, n in enumerate(self.nodes()):
      self.node_embeddings[i].update_start_time(start_times[0, i])

  def get_zero_placement(self):
    d = {}
    for n in self.nodes():
      d[n] = 0
    return d

  def get_random_placement(self, seed=None):
    d = {}
    if seed:
      random.seed(seed)
    for n in self.nodes():
      d[n] = random.randint(0, self.n_devs-1)
    return d

  def get_null_placement(self):
    d = {}
    for n in self.nodes():
      d[n] = None
    return d

  def reset_random_placement(self):
    d = self.get_random_placement()
    for i, n in enumerate(self.nodes()):
      self.node_embeddings[i].update_placement(d[n])
    nx.set_node_attributes(self.G, d, 'placement')
    return d

  def set_start_times(self, d):
    for i in range(self.n_nodes()):
      self.node_embeddings[i].update_start_time(d[0, i])

  def get_source_nodes(self):
    src_nodes = []
    for n in self.nodes():
      has_parent = False
      for _ in self.G.predecessors(p):
        has_parent = True
        break

      if not has_parent:
        src_nodes.append(n)

    return src_nodes

  def get_idx(self, node):
    return self.G.node[node]['idx']
    
  def neighbors(self, node):
    return self.G.neighbors(node)

  def predecessors(self, node):
    return self.G.predecessors(node)

  def nodes(self):
    return self.node_traversal_order

  def n_nodes(self):
    return len(self.G)

  def get_emb_size(self):
    return NodeEmbeddings.get_emb_size()

  def get_G(self):
    return self.G

  def get_optimal_runtime(self):
    return len(self.G)/ self.G.d + 1
