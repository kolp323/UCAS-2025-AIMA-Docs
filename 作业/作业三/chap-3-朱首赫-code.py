import math
import heapq
import time
from typing import List, Tuple, Set
import random
import matplotlib.pyplot as plt


# TSP问题实例生成器
def generate_cities(n: int, seed: int = None) -> List[Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)
    return [(random.random(), random.random()) for _ in range(n)]


# 计算距离矩阵
def distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        x1, y1 = cities[i]
        for j in range(i+1, n):
            dx = x1 - cities[j][0]
            dy = y1 - cities[j][1]
            dist[i][j] = dist[j][i] = math.hypot(dx, dy)
    return dist

# Prim算法实现MST
def prim_mst(dist: List[List[float]], nodes: Set[int]) -> float:
    if len(nodes) < 1:
        return 0.0
    
    visited = set()
    heap = []
    start_node = next(iter(nodes))
    visited.add(start_node)
    mst_cost = 0.0
    
    # 初始化堆
    for node in nodes - visited:
        heapq.heappush(heap, (dist[start_node][node], node))
    
    while heap and len(visited) < len(nodes):
        cost, node = heapq.heappop(heap)
        if node not in visited:
            visited.add(node)
            mst_cost += cost
            for neighbor in nodes - visited:
                heapq.heappush(heap, (dist[node][neighbor], neighbor))
    
    return mst_cost

# A*搜索节点
class Node:
    def __init__(self, path: List[int], cost: float, remaining: Set[int]):
        self.path = path
        self.cost = cost
        self.remaining = remaining
        self.heuristic = 0.0
        self._key = None
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)
    
    @property
    def key(self):
        if self._key is None:
            self._key = (tuple(self.path), frozenset(self.remaining))
        return self._key

# TSP求解器
class TSPSolver:
    def __init__(self, cities: List[Tuple[float, float]]):
        self.n = len(cities)
        self.dist = distance_matrix(cities)
        self.start = 0  # 固定起点
    
    def mst_heuristic(self, remaining: Set[int], current_pos: int) -> float:
        if not remaining:
            return self.dist[current_pos][self.start]
        
        # 计算剩余节点的MST
        mst = prim_mst(self.dist, remaining)
        
        # 找到连接到当前节点和起点的最小边
        min_to_current = min(self.dist[current_pos][node] for node in remaining) if remaining else 0
        min_to_start = min(self.dist[self.start][node] for node in remaining) if remaining else 0
        
        return mst + min_to_current + min_to_start
    
    def solve(self) -> Tuple[List[int], float]:
        initial_remaining = set(range(self.n)) - {self.start}
        initial_node = Node([self.start], 0.0, initial_remaining)
        initial_node.heuristic = self.mst_heuristic(initial_remaining, self.start)
        
        heap = [initial_node]
        visited = {}
        best_cost = float('inf')
        best_path = []
        
        while heap:
            node = heapq.heappop(heap)
            
            if node.cost >= best_cost:
                continue
            
            # 终止条件：返回起点
            if not node.remaining and len(node.path) == self.n:
                final_cost = node.cost + self.dist[node.path[-1]][self.start]
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_path = node.path + [self.start]
                continue
            
            # 扩展子节点
            current = node.path[-1]
            for next_node in node.remaining:
                new_cost = node.cost + self.dist[current][next_node]
                new_remaining = node.remaining - {next_node}
                new_path = node.path + [next_node]
                
                # 计算启发值
                heuristic = self.mst_heuristic(new_remaining, next_node)
                
                # 剪枝
                if new_cost + heuristic >= best_cost:
                    continue
                
                new_node = Node(new_path, new_cost, new_remaining)
                new_node.heuristic = heuristic
                
                # 检查是否已访问过更优路径
                if new_node.key in visited:
                    if visited[new_node.key] <= new_node.cost:
                        continue
                visited[new_node.key] = new_node.cost
                
                heapq.heappush(heap, new_node)
        
        return best_path, best_cost

def plot_tsp_instance(cities: List[Tuple[float, float]]) -> None:
    """可视化城市分布"""
    plt.figure(figsize=(8, 8))
    x = [c[0] for c in cities]
    y = [c[1] for c in cities]
    
    plt.scatter(x, y, c='red', s=40)
    plt.plot(x + [x[0]], y + [y[0]], 'b--', alpha=0.3)  # 连接首尾点
    
    plt.title(f"TSP Problem Instance ({len(cities)} Cities)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 示例使用
if __name__ == "__main__":
    city_num = 20
    seed = 42

    # 生成测试数据
    cities = generate_cities(city_num, seed = seed)

    # 可视化展示
    plot_tsp_instance(cities)

    # 初始化求解器
    solver = TSPSolver(cities)
    
    # 求解并计时
    start_time = time.time()
    path, cost = solver.solve()
    elapsed = time.time() - start_time
    
    print(f"最优路径: {path}")
    print(f"总成本: {cost:.4f}")
    print(f"计算时间: {elapsed:.2f}秒")