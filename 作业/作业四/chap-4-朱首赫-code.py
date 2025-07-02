
import numpy as np
import matplotlib.pyplot as plt
import random
import math

prblem_types = ["puzzle8", "queens8"]
algorithms = ["steepest", "first_choice", "random_restart", "annealing"]
optimal_puzzle8 = 22
optimal_queens8 = 4

# ---------------------------8-puzzle---------------------------------------
class puzzle8:
    def __init__(self, state=None):
        self.goal = np.array([[1, 2, 3], # 目标状态
                        [4, 5, 6],
                        [7, 8, 0]])
        if state is None:
            self.state = self.generate_solvable_state() # 随机生成可解问题
        else:
            self.state = np.array(state)

    def generate_solvable_state(self):
        while True:
            arr = list(range(9))
            random.shuffle(arr)
            state = np.array(arr).reshape(3,3)
            if self.is_solvable(state):
                return state 
    
    def is_solvable(self, state):
        inversions = 0
        flat = state.flatten()
        for i in range(len(flat)):
            for j in range(i+1, len(flat)):
                if flat[i] != 0 and flat[j] !=0 and flat[i] > flat[j]:
                    inversions +=1
        return inversions % 2 == 0 # 逆序数为偶数则可解
    
    def get_heuristic(self): # 采用曼哈顿距离启发式
        h = 0
        for i in range(3):
            for j in range(3):
                if self.state[i,j] != 0:
                    x_goal, y_goal = np.where(self.goal == self.state[i,j])
                    h += abs(i - x_goal) + abs(j - y_goal)
        return h

    def find_blank(self): # 找到空白格
        for i in range(3):
            for j in range(3):
                if self.state[i,j] == 0:
                    return i, j

    def get_successors(self): # 生成所有可能的合法移动
        moves = []
        blank_row, blank_col = self.find_blank()
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        for dx, dy in directions:
            new_row = blank_row + dx
            new_col = blank_col + dy
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = self.state.copy()
                new_state[blank_row, blank_col], new_state[new_row, new_col] = \
                    new_state[new_row, new_col], new_state[blank_row, blank_col] # 交换空白格和相邻格的值
                moves.append(puzzle8(new_state))
        return moves


# ---------------------------8-queens---------------------------------------
class queens8:
    # 长度为8的数组，每个元素表示第i列的皇后所在的行数（即初始时保证皇后不在同一行）
    def __init__(self, state=None):
        if state is None:
            self.state = self.generate_random_state()  
        else:
            self.state = state
        self.heuristic = self.calculate_heuristic()
        
    def calculate_heuristic(self): # 计算冲突数
        conflicts = 0
        for col in range(8):
            for other_col in range(col+1, 8):
                row, other_row = self.state[col], self.state[other_col]
                if row == other_row or \
                   abs(row - other_row) == abs(col - other_col):
                    conflicts +=1
        return conflicts
    
    def generate_random_state(self): # 随机生成一个初始状态
        state = np.random.permutation(8)
        return state
    
    def get_successors(self):  # 生成所有邻居状态（只移动单个皇后的位置）
        neighbors = []
        for col in range(8):
            for row in range(8):
                if row != self.state[col]:  # 选取一行进行改变，保证不是原来的位置
                    new_state = self.state.copy()
                    new_state[col] = row
                    neighbors.append(queens8(new_state))
        return neighbors


#---------------------------problem solver---------------------------------
# 最陡上升爬山法
def steepest_ascent(problem, maxsteps=1000): # 超过最大步数则停止，认为求解失败
    current = problem
    steps = 0
    for _ in range(maxsteps):
        neighbors = current.get_successors()
        if not neighbors:
            break # 没有邻居则停止
        best_neighbor = min(neighbors, key=lambda x: x.get_heuristic() if isinstance(x, puzzle8) else x.calculate_heuristic())
        if (isinstance(best_neighbor, puzzle8) and best_neighbor.get_heuristic() >= current.get_heuristic()) or \
           (isinstance(best_neighbor, queens8) and best_neighbor.calculate_heuristic() >= current.calculate_heuristic()):
            break  # 没有更优解则停止
        current = best_neighbor
        steps +=1
    return current, steps

# 首选爬山法
def first_choice_hill_climbing(problem, max_steps=1000):
    current = problem
    steps = 0
    for _ in range(max_steps):
        neighbors = current.get_successors()
        random.shuffle(neighbors) # 随机打乱邻居顺序
        improved = False
        for neighbor in neighbors:
            if (isinstance(neighbor, puzzle8) and neighbor.get_heuristic() < current.get_heuristic()) or \
               (isinstance(neighbor, queens8) and neighbor.calculate_heuristic() < current.calculate_heuristic()):
                current = neighbor
                steps +=1
                improved = True
                break
        if not improved: # 在所有邻居中都没有找到更优解，则停止搜索
            break
    return current, steps

# 重启爬山法（最大重启次数100）
def random_restart_hill_climbing(problem_generator, max_restarts=100): 
    restarts = 0
    total_steps = 0
    while restarts < max_restarts:
        current = problem_generator() # 每次重启生成新问题
        solution, steps = steepest_ascent(current, maxsteps=1000)  
        total_steps += steps
        restarts +=1
        if (isinstance(solution, puzzle8) and np.array_equal(solution.state, solution.goal)) or \
           (isinstance(solution, queens8) and solution.calculate_heuristic() == 0):
            return solution, total_steps
    return None, total_steps

# 模拟退火算法
def simulated_annealing(problem, initial_temperature=1000, cooling_rate=0.95, max_steps=10000):
    current = problem
    current_heuristic = current.get_heuristic() if isinstance(current, puzzle8) else current.calculate_heuristic()
    temperature = initial_temperature
    steps = 0
    for _ in range(max_steps):
        if temperature <= 1e-6:
            break
        neighbors = current.get_successors()
        if not neighbors: # 没有邻居则停止
            break
        next_state = random.choice(neighbors) # 随机选择一个邻居
        next_heuristic = next_state.get_heuristic() if isinstance(next_state, puzzle8) else next_state.calculate_heuristic() 
        delta_E = next_heuristic - current_heuristic
        delta_E = delta_E.item() if isinstance(delta_E, np.ndarray) else delta_E  # 提取单个元素
        if delta_E < 0 or random.random() < math.exp(-delta_E/temperature): # 接受更优解 或 以一定概率接受更差解
            current = next_state
            current_heuristic = next_heuristic
        temperature *= cooling_rate
        steps +=1
    return current, steps

# 实验100次，获取search cost、percentage of solved problems、optimal cost
def experiment(problem_type, algorithm, problem_num = 100):
    solved = 0
    total_cost = 0
    for _ in range(problem_num):
        if problem_type == "puzzle8":
            problem = puzzle8()
        elif problem_type == "queens8":
            problem = queens8()
        steps = 0
        solution = None
        if algorithm == "random_restart":
            solution, steps = random_restart_hill_climbing(lambda: queens8() if problem_type == "queens8" else puzzle8())
        else:
            if algorithm == "steepest":
                solution, steps = steepest_ascent(problem)
            elif algorithm == "first_choice":
                solution, steps = first_choice_hill_climbing(problem)
            elif algorithm == "annealing":
                solution, steps = simulated_annealing(problem)

        total_cost += steps
        if (problem_type == "puzzle8" and solution != None and np.array_equal(solution.state, solution.goal)) or \
           (problem_type == "queens8" and solution != None and solution.calculate_heuristic() == 0):
            solved += 1

    return solved/problem_num*100, total_cost/problem_num

# 可视化结果
def plot_results(problem_type, success_rates, avg_costs): 
    if problem_type == "puzzle8":
        avg_costs.append(optimal_puzzle8)
    else:
        avg_costs.append(optimal_queens8)

    extended_algorithms = algorithms + ["optimal"] 
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.bar(algorithms, success_rates)
    plt.title(f"{problem_type.capitalize()} Success Rate")
    plt.ylabel("Success Rate (%)")
    
    plt.subplot(1,2,2)
    plt.bar(extended_algorithms, avg_costs)
    plt.title(f"{problem_type.capitalize()} Average Costs")
    plt.ylabel("Average Costs")
    

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    for pro in prblem_types:
        print(f"Running {pro} experiments...")
        success_rates = []
        avg_costs = []
        
        for alg in algorithms:
            rate, costs = experiment(pro, alg)
            success_rates.append(rate)
            avg_costs.append(costs)
        print(success_rates)
        print(avg_costs)
        plot_results(pro, success_rates, avg_costs)
    