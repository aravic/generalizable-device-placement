import networkx as nx

# d is the number of chains
def makeChainGraph(N, d=2):
  G = nx.DiGraph()
  def add_edge(i, j):
    G.add_edge(str(i), str(j))

  '''
  for  N = 4, d = 2
    1 2 3 4 
  0         9
    5 6 7 8
  Lowest Runtime: (N+2) + l_fact* 2
  '''

  n = 1
  for i in range(d):
    add_edge(0, n)
    for j in range(N-1):
      add_edge(n, n+1)
      n += 1

    add_edge(n, N*d + 1)
    n += 1

  assert n == N*d + 1

  cost = {}
  out_size = {}
  for i in G.nodes():
    cost[i] = 1
    out_size[i] = 1
  nx.set_node_attributes(G, cost, 'cost')
  nx.set_node_attributes(G, out_size, 'out_size')
  G.d = d

  return G

def makeEdgeGraph(N):
  G = nx.DiGraph()
  for i in range(N):
    G.add_edge(2*i, 2*i + 1)

  cost = {}
  out_size = {}
  for i in G.nodes():
    cost[i] = 1
    out_size[i] = 1
  nx.set_node_attributes(G, cost, 'cost')
  nx.set_node_attributes(G, out_size, 'out_size')
  return G

def makeCrownGraph(N, d=2):
  G = nx.DiGraph()
  def add_edge(i, j):
    G.add_edge(str(i), str(j))

  '''
  for N = 4, d = 2

         8
      /  /\   \
     /  /  \   \
   /   /    \   \
  4 -> 5 -> 6 -> 7
  ^    ^    ^    ^
  |    |    |    |
  0 -> 1 -> 2 -> 3
  '''

  for i in range(d):
    for j in range(N):
      n = N*i + j
      if j != (N - 1):
        add_edge(n, n + 1)
      if i > 0:
        add_edge(N* (i-1) + j, n)
      if i == d - 1:
        add_edge(n, N* d)
        
  cost = {}
  out_size = {}
  for i in G.nodes():
    cost[i] = 1
    out_size[i] = .5
  nx.set_node_attributes(G, cost, 'cost')
  nx.set_node_attributes(G, out_size, 'out_size')
  G.d = d

  return G


