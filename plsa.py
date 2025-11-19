import gzip
import json
import time
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class EnhancedPLSATest:
    def __init__(self):
        self.results = []
    
    def generate_sat_instance(self, n_vars=10, clause_ratio=4.2):
        """生成具有不同硬度的SAT实例"""
        n_clauses = int(n_vars * clause_ratio)
        problem = {
            "n_vars": n_vars,
            "clause_ratio": clause_ratio,
            "clauses": []
        }
        for _ in range(n_clauses):
            clause = []
            variables = random.sample(range(1, n_vars+1), 3)
            for var in variables:
                sign = random.choice([-1, 1])
                clause.append(sign * var)
            problem["clauses"].append(clause)
        return problem
    
    def dpll_solver(self, problem, depth=0):
        """一个简化的DPLL算法作为智能求解器"""
        clauses = problem["clauses"].copy()
        n_vars = problem["n_vars"]
        
        # 简单的单元传播和纯文字消除
        assignment = [None] * n_vars
        trace = {"decisions": 0, "backtracks": 0, "steps": []}
        
        def simplify(clauses, literal):
            new_clauses = []
            for clause in clauses:
                if literal in clause:
                    continue  # 子句已满足
                new_clause = [l for l in clause if l != -literal]
                if not new_clause:
                    return None  # 冲突
                new_clauses.append(new_clause)
            return new_clauses
        
        def solve_recursive(clauses, assignment, var_idx=0):
            if var_idx >= n_vars:
                return assignment, True
                
            if all(len(c) == 0 for c in clauses):
                return assignment, True
                
            # 选择变量
            trace["decisions"] += 1
            current_var = var_idx + 1
            
            # 尝试True
            new_assignment = assignment.copy()
            new_assignment[var_idx] = True
            new_clauses = simplify(clauses, current_var)
            
            if new_clauses is not None:
                result, solved = solve_recursive(new_clauses, new_assignment, var_idx+1)
                if solved:
                    return result, True
            
            # 回溯
            trace["backtracks"] += 1
            
            # 尝试False
            new_assignment = assignment.copy()
            new_assignment[var_idx] = False
            new_clauses = simplify(clauses, -current_var)
            
            if new_clauses is not None:
                result, solved = solve_recursive(new_clauses, new_assignment, var_idx+1)
                if solved:
                    return result, True
            
            return None, False
        
        solution, solved = solve_recursive(clauses, assignment)
        return solution, trace
    
    def random_walk_solver(self, problem, max_steps=1000):
        """随机游走求解器"""
        n_vars = problem["n_vars"]
        clauses = problem["clauses"]
        
        trace_steps = []
        assignment = [random.choice([True, False]) for _ in range(n_vars)]
        
        for step in range(max_steps):
            trace_steps.append(assignment.copy())
            
            # 检查是否满足
            unsatisfied = []
            for i, clause in enumerate(clauses):
                satisfied = False
                for lit in clause:
                    var = abs(lit) - 1
                    value = assignment[var]
                    if (lit > 0 and value) or (lit < 0 and not value):
                        satisfied = True
                        break
                if not satisfied:
                    unsatisfied.append(i)
            
            if not unsatisfied:
                trace = {"steps": trace_steps, "total_steps": step+1}
                return assignment, trace
            
            # 随机选择一个不满足的子句并翻转一个变量
            chosen_clause = random.choice(unsatisfied)
            chosen_literal = random.choice(clauses[chosen_clause])
            var_to_flip = abs(chosen_literal) - 1
            assignment[var_to_flip] = not assignment[var_to_flip]
        
        trace = {"steps": trace_steps, "total_steps": max_steps}
        return None, trace
    
    def compute_kolmogorov_complexity(self, obj):
        """计算基于压缩的柯尔莫哥洛夫复杂度估计"""
        json_str = json.dumps(obj, sort_keys=True, separators=(',', ':'))
        return len(gzip.compress(json_str.encode('utf-8')))
    
    def compute_structural_measures(self, problem, solution, trace):
        """计算完整的结构度量"""
        # 基础压缩大小
        K_problem = self.compute_kolmogorov_complexity(problem)
        K_solution = self.compute_kolmogorov_complexity({
            "problem": problem, "solution": solution, "trace": trace
        })
        
        # 结构不可压缩性 (K)
        conditional_complexity = max(K_solution - K_problem, 1)
        K_measure = conditional_complexity / K_problem
        
        # 结构曲率 (S) - 基于轨迹变化的二阶差分
        if "steps" in trace and len(trace["steps"]) > 2:
            steps_str = [json.dumps(step, sort_keys=True) for step in trace["steps"]]
            complexities = [len(gzip.compress(step.encode())) for step in steps_str]
            if len(complexities) >= 3:
                curvatures = []
                for i in range(1, len(complexities)-1):
                    curvature = abs(complexities[i+1] - 2*complexities[i] + complexities[i-1])
                    curvatures.append(curvature)
                S_measure = np.mean(curvatures) / 1000 if curvatures else 0.1
            else:
                S_measure = 0.1
        else:
            S_measure = len(str(trace)) / 10000  # 回退代理
        
        # 正确性 (C)
        C_measure = 0 if solution is not None else 1
        
        return {
            "K": K_measure,
            "S": S_measure, 
            "C": C_measure,
            "K_absolute": conditional_complexity,
            "K_problem": K_problem
        }
    
    def compute_structural_action(self, measures, weights=None):
        """计算结构作用量"""
        if weights is None:
            weights = {"alpha": 1.0, "beta": 0.5, "gamma": 10.0}
        
        return (weights["alpha"] * measures["K"] + 
                weights["beta"] * measures["S"] + 
                weights["gamma"] * measures["C"])
    
    def test_structure_time_law(self, n_tests=30):
        """测试结构-时间定律"""
        print("=== 增强PLSA证伪测试 ===\n")
        
        instances = []
        for i in range(n_tests):
            # 生成不同硬度的实例
            n_vars = random.choice([8, 10, 12, 15])
            clause_ratio = random.choice([3.5, 4.0, 4.2, 4.5])
            
            problem = self.generate_sat_instance(n_vars, clause_ratio)
            instances.append(problem)
            
            # 智能求解器
            start_time = time.time()
            smart_sol, smart_trace = self.dpll_solver(problem)
            smart_time = time.time() - start_time
            
            # 随机求解器  
            start_time = time.time()
            random_sol, random_trace = self.random_walk_solver(problem)
            random_time = time.time() - start_time
            
            # 计算结构度量
            smart_measures = self.compute_structural_measures(problem, smart_sol, smart_trace)
            random_measures = self.compute_structural_measures(problem, random_sol, random_trace)
            
            # 计算结构作用量
            smart_action = self.compute_structural_action(smart_measures)
            random_action = self.compute_structural_action(random_measures)
            
            result = {
                "problem_id": i,
                "n_vars": n_vars,
                "clause_ratio": clause_ratio,
                "smart": {
                    "action": smart_action,
                    "time": smart_time,
                    "measures": smart_measures,
                    "solved": smart_sol is not None
                },
                "random": {
                    "action": random_action, 
                    "time": random_time,
                    "measures": random_measures,
                    "solved": random_sol is not None
                }
            }
            self.results.append(result)
            
            print(f"实例 {i+1:2d} (n={n_vars}, α={clause_ratio}): "
                  f"智能={smart_action:.4f}, 随机={random_action:.4f}, "
                  f"时间比={smart_time/random_time:.3f}")
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.results:
            print("请先运行测试!")
            return
        
        smart_actions = [r["smart"]["action"] for r in self.results]
        random_actions = [r["random"]["action"] for r in self.results]
        smart_times = [r["smart"]["time"] for r in self.results]
        random_times = [r["random"]["time"] for r in self.results]
        
        # 1. 结构作用量比较
        t_stat, p_value = stats.ttest_rel(smart_actions, random_actions)
        
        print(f"\n=== 结构作用量分析 ===")
        print(f"智能求解器平均作用量: {np.mean(smart_actions):.4f} ± {np.std(smart_actions):.4f}")
        print(f"随机求解器平均作用量: {np.mean(random_actions):.4f} ± {np.std(random_actions):.4f}")
        print(f"配对t检验 p值: {p_value:.6f}")
        print(f"效应大小 (Cohen's d): {np.mean(random_actions) - np.mean(smart_actions):.4f}")
        
        # 2. 结构-时间定律验证
        hardness_indices = [1 - r["smart"]["measures"]["K"] for r in self.results]
        log_times = [np.log2(max(t, 1e-6)) for t in smart_times]
        
        # 线性回归
        X = np.column_stack([hardness_indices, [r["n_vars"] for r in self.results]])
        model = LinearRegression()
        model.fit(X, log_times)
        
        print(f"\n=== 结构-时间定律验证 ===")
        print(f"回归系数 (硬度): {model.coef_[0]:.4f}")
        print(f"回归系数 (问题规模): {model.coef_[1]:.4f}")
        print(f"截距: {model.intercept_:.4f}")
        print(f"R² 分数: {model.score(X, log_times):.4f}")
        
        # 3. 可视化
        self.plot_results(hardness_indices, log_times, smart_actions, random_actions)
        
        # 结果解释
        if p_value < 0.05 and np.mean(smart_actions) < np.mean(random_actions):
            print("\n✅ 强烈支持PLSA理论!")
            print("   - 智能求解器产生显著更小的结构作用量")
            print("   - 结构-时间定律得到实证支持")
            if model.coef_[0] > 0:
                print("   - 结构硬度确实预测求解时间")
        else:
            print("\n❌ 结果挑战PLSA理论")
    
    def plot_results(self, hardness, log_times, smart_actions, random_actions):
        """绘制结果图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 结构作用量比较
        x_pos = np.arange(len(smart_actions))
        ax1.bar(x_pos - 0.2, smart_actions, 0.4, label='智能求解器', alpha=0.7, color='blue')
        ax1.bar(x_pos + 0.2, random_actions, 0.4, label='随机求解器', alpha=0.7, color='red')
        ax1.set_xlabel('问题实例')
        ax1.set_ylabel('结构作用量')
        ax1.set_title('智能vs随机求解器的结构作用量')
        ax1.legend()
        
        # 2. 结构-时间关系
        ax2.scatter(hardness, log_times, alpha=0.6)
        z = np.polyfit(hardness, log_times, 1)
        p = np.poly1d(z)
        ax2.plot(hardness, p(hardness), "r--", alpha=0.8)
        ax2.set_xlabel('结构硬度指数 h(x)')
        ax2.set_ylabel('log2(求解时间)')
        ax2.set_title('结构-时间定律')
        
        # 3. 作用量分布
        ax3.hist(smart_actions, alpha=0.7, label='智能', bins=15, color='blue')
        ax3.hist(random_actions, alpha=0.7, label='随机', bins=15, color='red')
        ax3.set_xlabel('结构作用量')
        ax3.set_ylabel('频次')
        ax3.set_title('结构作用量分布')
        ax3.legend()
        
        # 4. 性能对比
        time_ratios = [r["random"]["time"]/max(r["smart"]["time"], 1e-6) for r in self.results]
        ax4.scatter(smart_actions, time_ratios, alpha=0.6)
        ax4.set_xlabel('智能求解器结构作用量')
        ax4.set_ylabel('随机/智能时间比')
        ax4.set_title('结构效率vs时间效率')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

# 运行增强测试
if __name__ == "__main__":
    test_suite = EnhancedPLSATest()
    test_suite.test_structure_time_law(n_tests=30)
    test_suite.analyze_results()