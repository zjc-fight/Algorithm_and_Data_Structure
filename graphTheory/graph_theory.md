Graph theory

> Graph therory is the mathematical theory of the properties and application of graphs (networks)

# 图的分类及表示

## 基本图类型

### 无向图

无向图中边没有方向，边$(u,v)$等价于边$(v,u)$

![](./image/undirected_graph_1.png)

### 有向图

有向图中，边是有方向的，边$(u,v)$表示节点u到节点v的有向边。

![](./image/directed_graph_1.png)

### 权重图

权重图中，每条边都附带了一个权重，用于表示消耗、距离、数量等。

![](./image/weighted_graphs_1.png)

## 特殊图类型

### 树

树是一种无向无环图，等价低，它是一种由N个节点与N-1条边组成的连通图。

![](./image/tree_1.png)

**二叉搜索树**

![](./image/binary_search_tree.png)

**root-tree**

![](./image/root_tree_1.png)

### 有向无环图（DAG）

![](./image/dag_1.png)

### 二部图

![](./image/bipartitle_graph_1.png)

### 全连接图

![](./image/complete_graph.png)

## 图的表示

### 邻接矩阵

用一个n*n的矩阵来表示图，矩阵的元素表示边的权重，邻接矩阵用于表示稠密矩阵比较高效。

![](./image/adjacency_matrix.png)

### 邻接链表

邻接链表用一系列链表来表示图的链接，每条链表都对应一个节点以及与之相连的所有边

![](./image/adjacency_list.png)

### 边的集合（edge list）

![](./image/edge_list.png)

# 常见图论应用

当遇到图论相关应用时，需要明确以下几个问题：

1. 有向图还是无向图

2. 图的边带权重吗

3. 图是稠密还是稀疏

4. 应该怎么表示这个图，邻接矩阵？邻接链表？edge list？

## 最短路径问题

给定一个权重图，找出节点A到节点B的最短路径

![](./image/short_path_problem.png)

## 图的连通性

给定一张图，判断两个节点之间是否联通

![](./image/connectivity.png)

## 负向环（Negtive cycles）

给定一个带权重的有向图，判断其中是否存在权重和为负数的环

![](./image/negative_cycles.png)

## 强连通图

强连通部分是指图中的的环，其中环的每一个节点都能到达该环的其他节点

![](./image/strongly_connected_components.png)

## 旅行商问题

![](./image/traveling_salesman_problem.png)

## bridge (cut edge)

![](./image/bridges.png)

## cut vertex

![](./image/cut_verticles.png)

## 最小生成树

![](./image/minimum_spanning_tree_1.png)

![](./image/minimum_spanning_tree_2.png)

## 网络流量

![](./image/network_flow.png)

# 常用图算法

## 深度优先搜索（Depth First Search）

深度优先搜索算法是常用的图搜索算法，其算法基本框架如下：

```python
n = number of nodes in the graph
g = adjacency list representing graph
visited = [false,...,false]  # size n
function dfs(at):
    if visited[at]: return
    visited[at] = true
    neighbours = graph[at]
    for next in neighbours:
        dfs(next)
start_node = 0
dfs(start_node)
```

深度优先搜索算法有非常广泛的应用：

1. 寻找图中的联通分量

2. 计算图的最小生成树

3. 检测图中是否存在环

4. 判断图是否是二部图

5. 图的拓扑排序

6. 寻找图中的桥以及关节点

7. 寻找流量网络中的增广路径

8. 寻找迷宫中的有效路径

## 广度优先搜索（Breadth First Search）

广度优先搜索类似于扩散的过程，先搜索附近的邻接点，再向远处扩散。

通常，广度优先搜索最长用的场景是寻找无向图节点间的最短路径，会借助于Queue来实现：

```python
n = number of nodes in the graph
g = adjacency list representing graph
function bfs(s, e):
    prev = solve(s)
    return reconstructPath(s, e, prev)

function solve(s):
    q = queue data structure with enqueue and dequeue
    q.enqueue(s)
    visited = [false,...,false]  # size n
    visited[s] = true

    prev = [null, ..., null]   # size n 
    while !q.isEmpty():
        node = q.dequeue()
        neighbours = g.get(node)
        for (next : neighbours):
            if !visited[next]:
                q.enqueue(next)
                visited[next] = true
                prev[next] = node
    return prev

function reconstructPath(s, e, prev):
    path = []
    for (at = e; at != null; at = prev[at]):
        path.add(at)
    path.reverse()
    if path[0] == s:
        return path
    return []
```

## 树算法

### 树的存储与表示

除了常见的用指针来表示树以外，另一种比较方便的是用数组来表示树：

![](./image/array_represent_tree.png)

### 树算法应用

#### 叶节点数字求和

对树的叶节点求和

![](/Users/bytedance/workspace/Algorithm_and_Data_Structure/graphTheory/image/sum_of_tree_leaf.png)

伪代码：

```python
function leafSum(node):
    if node == null:
        return 0
    if isLeaf(node):
        return node.getValue()
    total = 0
    for child in node.getChildNodes():
        total += leafSum(child)
    return total


function isLeaf(node):
    return node.getChildNodes().size() == 0
```

#### 树高度

![](./image/heigh_of_tree.png)

伪代码如下：

```python
function treeHeight(node):
    if node == null:
        return -1
    if node.left == null and node.right == null:
        return 0
    return max(treeHeight(node.left),
               treeHeight(node.right)) + 1
```

#### 寻找树的根节点

给定一个无向图，以某个节点为根节点构造一颗树

具体操作是从根节点出发，利用DFS算法逐步构建出一颗完整的树

#### 寻找树的中心节点

给定代表树的无向图，找出图中的中心节点

![](./image/center_of_tree.png)

一个可行的做法是，逐步删除外围节点（即度为1的节点），则最后删除的节点便是中心节点

伪代码：

```python
function treeCenters(g):
    n = g.numberOfNodes()
    degree = [0,0,...,n]
    leaves = []
    for (int i = 0; i < n; ++i):
        degree[i] = q[i].size()
        if degree[i] == 0 or degree[i] == 1:
            leaves.add(i)
            degree[i] = 0
    count = leaves.size()
    while count < n:
        new_leaves = []
        for (node : leaves):
            for (neighbor : g[node]):
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    new_leaves.add(neighbor)
            degree[node] = 0
        count += new_leaves.size()
        leaves = new_leaves
    return leaves
```

#### 同构树（isomorphism）

同构树：两个树的结构相同，用数学语言表示就是，针对两个图$G_1(V_1, E_1)$和$G_2(V_2,E_2)$。如果两张图的节点间存在一种映射关系$\phi$，满足：

$$
\forall u,v \in V_1,(u,v) \in E_1 <=> (\phi(u),\phi(v)) \in E_2
$$

则称两个图结构相同。

常用的判断两个树是否同构的算法是，先对两颗树进行某种编码，将树映射成一个字符串，然后对比两个字符串是否相同，如果相同，则为同构树。

算法分为以下步骤：

1. 首先将图转变成一颗带根节点的树

2. 应用AHU算法对树进行编码：首先每个叶节点编码成'()'，然后逐层向上编码，对每个中间节点，先将所有子树进行编码，然后将所有子树的编码进行字典排序，最后连接起来，并在外围增加一个'(xxxx)'

![](./image/isomorphism_tree.png)

伪代码如下：

```python
function treesAreIsomorphic(tree1, tree2):
  tree1_centers = treeCenters(tree1)
  tree2_centers = treeCenters(tree2)

  tree1_rooted = rootTree(tree1, tree1_centers[0])
  tree1_encoded = encode(tree1_rooted)

  for center in tree2_conters:
    tree2_rooted = rootTree(tree2, center)
    tree2_encoded = encode(tree2_rooted)
    if tree1_encodes == tree2_encoded:
      return true
  return false


function encode(treeNode):
  if treeNode == null:
    return ""
  labels = []
  for child in node.children():
    labels.add(encode(child))
  sort(labels)

  result = ""
  for label in labels:
    result += label
  return "(" + result + ")"
```

#### 最低公共祖先

给定一颗root tree，找出树中任意两个节点的最低公共祖先(Lowest Common Ancestor)

这里介绍一种基于Eulerian tour的方法，首先从根节点出发，遍历每个节点，最后回到根节点，生成访问的路径（即Eulerian tour）。生成路径的过程中需要记录tour中每个节点对应的树的节点，以及在树中的深度（即需要两个数组nodes、depth来记录）。

![](./image/eulerian_tour_tree_1.png)



![](./image/eulerian_tour_tree_2.png)



生成depth 和nodes之后，接下来按以下不准寻找两个节点的最低公共祖先：

1. 寻找两个树节点在tour路径中对应的index，由于tour中同一个树节点会访问多次，所以只需要找最后访问的index即可，因此还需要另一个last数组，来记录每个树节点最后被访问的index。

2. 在depth中，找出第一步得到的两个节点的index之间的最小深度对应的index

3. 得到最小深度的index之后，到nodes节点取出对应index的树节点，就找到了最低公共祖先



算法伪代码如下：

```python
function setup(n, root):
  nodes = ... # array of nodes of size 2n-1
  depth = ... # array of integers of size 2n-1
  last = ... # node index -> euler tour index

  # do eulerian tour around the tree
  dfs(root)
  sparse_table = CreateMinSparseTable(depth)

tour_index = 0
function dfs(node, node_depth=0):
  if node == null:
    return
  visit(node, node_depth)
  for (child in node.children):
    dfs(child, node_depth + 1)
    visit(node, node_depth)

function visit(node, node_depth):
  nodes[tour_index] = node
  depth[tour_index] = node_depth
  last[node.index] = tour_index
  tour_index += 1


function lca(index1, index2):
  l = min(last[index1], last[index2])
  r = max(last[index1], last[index2])
  
  i = sparse_table.queryIndex(l, r)
  return nodes[i]
```

伪代码中，CreateMinSparseTable函数构建了一个表格，可以快速查找arr[i]、arr[j]之间的最小值：

![](./image/min_sparse_table.png)



## 拓扑排序

拓扑排序是指将一个有向无环图的节点排列成一个序列，如果存在A->B的边，则A需要排在B前面。拓扑排序有非常多的应用，比如程序编译的依赖、课程排列。



DFS recursion对DAG进行拓扑排序：

1. 从一个未被访问的节点开始进行DFS，记录dfs得到的路径

2. 将dfs得到的路径加入拓扑序列中

3. 重复1、2步骤，知道所有节点都被访问

伪代码如下：

```python
function topsort(graph):
  N = graph.numberOfNode()
  V = [false, false, ..., false] # size of N
  ordering = [0,..,0]    # size of N
  i = N-1
  for (at = 0; at < N; at++):
    if V[at] == false:
      visitedNodes = []
      dfs(at, V, visitedNodes, graph)
        for nodeId in visitedNodes:
          ordering[i] = nodeId
          i -= 1
  return ordering
```



## 最短/最长路径

### 单源最短路径（SSSP）

DAG图中的Single Source Shortest Path问题能够在$O(V+E)$的时间复杂度下求解，因为DAG图能够在$O(V+E)$内完成拓扑排序，得到序列之后，从头到位遍历每个节点，并基于源节点到当前节点的最短距离，更新源节点到当前节点的邻接点的最短路径。



![](./image/single_source_shortest_path.png)



相应地，求DAG中的单源最长距离，只需要把每条边的权重变成其相反数，再求解SSSP即可。



#### Dijkstra 算法

Dijkstra算法能够在$O(V*log(E))$的复杂度下求解SSSP问题，但有个前提是要求图中不能有负权重的边，这个前提保证了，一旦某个节点的最短距离确定了，就不再改变了，这是Dijkstra算法的核心思想。

Dijkstra算法维护了一个dist数组，存储源节点到当前节点的最短路径，然后从源节点开始，每次选择dist最小的节点，并更新其邻节点的最小距离。

```python
function dijkstra(g, n, s):
  vis = [false, false,..., false] # size n
  dist = [max, max,..., max] # size n
  prev = [null, null,...,null] # size
  dist[s] = 0
  pq = empty # priority_queue
  pq.insert((s,0))
  while pq.size() != 0:
    index, minval = pq.poll()
    vis[index] = true
    for edge : g[index]:
      if vis[edge.to] == true:    continue
      newDist = dist[index] + edge.cost
      if newDist < dist[edge.to]:
        prev[edge.to] = index
        dist[edge.to] = newDist
        pq.insert((edge.to, newDist))
  return dist, prev
```



如果需要求解具体的路径，可以增加一个prev数组来记录最短路径中每个节点的前置节点。

```python
function findShorteastPath(g,n,s,e):
  dist, prev = dijkstra(g,n,s)
  path = []
  if (dist[e] == maxvalue)    return path
  for (at = e; at != null; at = prev[at])
    path.add(at)
  path.reverse()
  return path
```



如果需要求解到指定目的节点的最短距离，则当遍历到目的节点时，可以提前退出，不再需要遍历其他节点。



当图的边比较稠密时，上面的dijkstra算法会设计比较多的priority_queue操作，这时候可以用**D-ary heap**来进行优化，即优先队列中的每个节点含有多个子节点，即多叉树



![](./image/d_ary_tree.png)

知识点：Fibonacci Tree可以进一步提升Dijkstra算法的性能到$O(V + log(V))$，但是比较难以实现，除非图非常大，否则不建议使用。



#### Bellman-Ford(BF)算法

当图中含有负权重的边时，Dijkstra算法无法使用，这时候可以使用BF算法。BF算法可以用来检测negative cycles。

BF算法步骤如下：

1. 每个节点的最短路径D设置为+maxvalue

2. D[s] = 0

3. 遍历图中的边，依次更新每个节点的最短路径，改操作重复$V-1$次。



```python
for (i = 0; i < V-1; ++i):
  for (edge in graph.edges):
    if (D[edge.from] + edge.cost < D[edge.to])
      D[edge.to] = D[edge.from] + edge.cost

# repeat to find nodes in negative cycle
for (i = 0; i < V-1; ++i):
  for (edge in graph.edges):
    if (D[edge.from] + edge.cost < D[edge.to])
      D[edge.to] = -maxvalue
```



![](./image/Bellman-Ford.png)





### All pair最短路径(APSP)

APSP问题是需要找出图中所有节点间的最短路径，现有的Shortest Path相关算法对比：

![](./image/shortest_path_compare.png)



Floyd-Warshall算法

FW算法用动态规划的思想来求解节点间的最短路径。

dp[i][j]表示节点i到节点j的最短路径，状态转移方程

$dp[i][j] = min(dp[i][k] + dp[k][j]) for k \in \{1,2,...,n\}$

FW用邻接矩阵来表示图，伪代码如下：

```python
function floydWarshall(m):
  setup(m)   # 初始化dp矩阵
  for (int k = 0; k < n; ++k):
    for(int i = 0; i < n; ++i):
      for(int j = 0; j < n; ++j):
        if (dp[i][k] + dp[k][j] < dp[i][j]):
          dp[i][j] = dp[i][k] + dp[k][j]
          next[i][j] = next[i][k]
  propagateNegativeCycles(dp, n)
  return dp
```

```python
# 初始化dp矩阵，以及next数组表示i->j的最短路径中i的后一个节点
function setup(m):
  dp = empty matrix of size n x n
  next = empty integer matrix of size n x n
  for (int i = 0; i < n; ++i):
    for (int j = 0; j < n; ++j):
      dp[i][j] = m[i][j]
      if m[i][j] != maxvalue:
        next[i][j] = j
```

```python
function propagateNegativeCycles(dp, n):
  for (int k = 0; k < n; ++k):
    for(int i = 0; i < n; ++i):
      for(int j = 0; j < n; ++j):
        if (dp[i][k] + dp[k][j] < dp[i][j]):
          dp[i][j] = -maxvalue
          next[i][j] = -1
```

```python
# 重建节点start->end的最短路径
function reconstructPaht(start, end):
 path = []
if dp[start][end] == +infinity: return path
at = start
for (; at != end; at = next[at][end]):
  if at == -1: return null
  path.add(at)
if next[at][end] == -1:    return null
path.add(end)
return path
```



## 强连通图(SCC)

强连通图（Strongly Connected Components）可以认为是一个自包含的有向图，图中的任意两个节点之间都有路径可以到达。

![](./image/strongly_connected_components.png)



### Tarjan 算法

Tarjan算法的核心是找到每个节点的**low-link value**，节点x的low-link表示用dfs算法从节点x开始搜索，x所能到达的最小的node id。

当每个节点都遍历完之后，low-link value相同的节点就组成了一个强连通图，如下：

![](./image/low_link_value_in_scc.png)

用dfs搜索low-link value时需要注意以下几点：

1. 当遍历节点x的邻接节点y时，如果节点y没有在当前搜索路径中（stack中），则不能用y的low-link value来更新节点x的low-link value

Tarjan算法流程如下：

1. 当遍历到节点x时，为x分配一个node_id，并将node_id设置为节点x的low-link value

2. 遍历节点x的邻居节点，调用dfs，并将节点x的low-link value更新为邻居节点的low-link value的最小值

3. 当遍历完节点x的左右邻居节点后，如果节点x的low-link value等于其node id，则找到了一个x作为入口的强连通图

伪代码如下：

```python
UNVISITED = -1
n = number of nodes in graph
g = adjacency list with directed edges
id = 0
sccCount = 0
ids = [0, 0, ..., 0]    # length n
low = [0, 0, ..., 0]    # length n
onStack = [false, false, ..., false]    # length n
stack = an empty stack

function findSccs():
  for (i = 0; i < n; i++):    ids[i] = UNVISITED
  for (i = 0; i < n; i++):
    if (ids[i] == UNVISITED):
      dfs[i]
  return low
```

```python
function dfs(at):
  stack.push(at)
  onStack[at] = true
  ids[at] = low[at] = id++
  # visit all neighbours & min low-link
  for(to : g[at]):
    if (ids[to] == UNVISITED):    dfs(ids[to])
    if (onStack[to]):    low[at] = min(low[at], low[to])
  
  if (ids[at] == low[at]):
    for (node = stack.pop();; node = stack.pop()):
      onStack[node] = false
      low[node] = ids[at]
      if (node == at):    break
    sccCount++
```



## 旅行商问题


