Graph theory

> Graph therory is the mathematical theory of the properties and application of graphs (networks)



# 图的分类及表示

## 基本图类型

### 无向图

无向图中边没有方向，边$(u,v)$等价于边$(v,u)$

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/undirected_graph_1.png)



### 有向图

有向图中，边是有方向的，边$(u,v)$表示节点u到节点v的有向边。

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/directed_graph_1.png)



### 权重图

权重图中，每条边都附带了一个权重，用于表示消耗、距离、数量等。

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/weighted_graphs_1.png)



## 特殊图类型

### 树

树是一种无向无环图，等价低，它是一种由N个节点与N-1条边组成的连通图。

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/tree_1.png)



**root-tree**

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/root_tree_1.png)



### 有向无环图（DAG）

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/dag_1.png)



### 二部图

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/bipartitle_graph_1.png)



### 全连接图

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/complete_graph.png)





## 图的表示

### 邻接矩阵

用一个n*n的矩阵来表示图，矩阵的元素表示边的权重，邻接矩阵用于表示稠密矩阵比较高效。

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/adjacency_matrix.png)



### 邻接链表

邻接链表用一系列链表来表示图的链接，每条链表都对应一个节点以及与之相连的所有边

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/adjacency_list.png)



### 边的集合（edge list）

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/edge_list.png)



# 常见图论应用

当遇到图论相关应用时，需要明确一下几个问题：

1. 有向图还是无向图

2. 图的边带权重吗

3. 图是稠密还是稀疏

4. 应该怎么表示这个图，邻接矩阵？邻接链表？edge list？



## 最短路径问题

给定一个权重图，找出节点A到节点B的最短路径

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/short_path_problem.png)



## 图的连通性

给定一张图，判断两个节点之间是否联通

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/connectivity.png)



## 负向环（Negtive cycles）

给定一个带权重的有向图，判断其中是否存在权重和为负数的环

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/negative_cycles.png)



## 强连通图

强连通部分是指图中的的环，其中环的每一个节点都能到达该环的其他节点

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/strongly_connected_components.png)



## 旅行商问题



![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/traveling_salesman_problem.png)



## bridge (cut edge)



![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/bridges.png)



## cut vertex



![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/cut_verticles.png)



## 最小生成树



![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/minimum_spanning_tree_1.png)

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/minimum_spanning_tree_2.png)

## 网络流量

![](/Users/bytedance/workspace/dailyNotes/source/graphTheory/image/network_flow.png)



# 常用图算法

## 深度优先搜索（Depth First Search）

深度优先搜索算法是常用的图搜索算法，其算法基本框架如下：

```
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




