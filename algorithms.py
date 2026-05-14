import numpy as np
import time
from typing import Callable, Tuple, List, Dict, Any


class BaseOptimizer:
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, seed: int = None):
        self.objective_func = objective_func
        self.dim = dim
        self.lb, self.ub = bounds
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.seed = seed
        if not isinstance(self.lb, np.ndarray):
            self.lb = np.array([self.lb] * dim) if np.isscalar(self.lb) else np.array(self.lb)
        if not isinstance(self.ub, np.ndarray):
            self.ub = np.array([self.ub] * dim) if np.isscalar(self.ub) else np.array(self.ub)
        if seed is not None:
            np.random.seed(seed)
        self.history = []
        self.time_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.execution_time = 0
        self.function_evaluations = 0

    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        fitness = []
        for pos in positions:
            val = self.objective_func(pos)
            if np.isnan(val) or np.isinf(val):
                val = 1e6
            fitness.append(val)
        self.function_evaluations += len(positions)
        return np.array(fitness)

    def _record_time(self, start_time):
        self.time_history.append(time.time() - start_time)

    def optimize(self):
        raise NotImplementedError

    def get_metrics(self):
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'function_evaluations': self.function_evaluations,
            'convergence_history': self.history,
            'time_history': self.time_history,
            'population_size': self.pop_size,
            'iterations': self.max_iter
        }


class PSO(BaseOptimizer):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, w: float = 0.729,
                 c1: float = 1.49445, c2: float = 1.49445, seed: int = None):
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        self.w = w; self.c1 = c1; self.c2 = c2

    def optimize(self):
        start_time = time.time()
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = np.argmin(pbest_fitness)
        self.best_fitness = pbest_fitness[gbest_idx]
        self.best_solution = pbest_positions[gbest_idx].copy()
        self.history.append(self.best_fitness)
        self._record_time(start_time)

        for _ in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (pbest_positions - positions) +
                          self.c2 * r2 * (self.best_solution - positions))
            v_max = 0.1 * (self.ub - self.lb)
            velocities = np.clip(velocities, -v_max, v_max)
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)
            fitness = self._evaluate(positions)
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved] = fitness[improved]
            curr_best = np.argmin(pbest_fitness)
            if pbest_fitness[curr_best] < self.best_fitness:
                self.best_fitness = pbest_fitness[curr_best]
                self.best_solution = pbest_positions[curr_best].copy()
            self.history.append(self.best_fitness)
            self._record_time(start_time)
            if self.best_fitness < 1e-10:
                break
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class GWO(BaseOptimizer):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, seed: int = None):
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)

    def optimize(self):
        start_time = time.time()
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        idx = np.argsort(fitness)
        alpha_pos, beta_pos, delta_pos = positions[idx[0]].copy(), positions[idx[1]].copy(), positions[idx[2]].copy()
        alpha_score, beta_score, delta_score = fitness[idx[0]], fitness[idx[1]], fitness[idx[2]]
        self.best_solution, self.best_fitness = alpha_pos.copy(), alpha_score
        self.history.append(self.best_fitness)
        self._record_time(start_time)

        for t in range(self.max_iter):
            a = 2.0 - t * (2.0 / self.max_iter)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1, C1 = 2*a*r1 - a, 2*r2
                X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2, C2 = 2*a*r1 - a, 2*r2
                X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3, C3 = 2*a*r1 - a, 2*r2
                X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])
                positions[i] = (X1 + X2 + X3) / 3.0
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            fitness = self._evaluate(positions)
            for i in range(self.pop_size):
                if fitness[i] < alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness[i], positions[i].copy()
                elif fitness[i] < beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness[i], positions[i].copy()
                elif fitness[i] < delta_score:
                    delta_score, delta_pos = fitness[i], positions[i].copy()
            self.best_solution, self.best_fitness = alpha_pos.copy(), alpha_score
            self.history.append(self.best_fitness)
            self._record_time(start_time)
            if self.best_fitness < 1e-10:
                break
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class WOA(BaseOptimizer):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, seed: int = None):
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)

    def optimize(self):
        start_time = time.time()
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.best_solution = best_pos.copy()
        self.history.append(self.best_fitness)
        self._record_time(start_time)

        for t in range(self.max_iter):
            a = 2.0 - t * (2.0 / self.max_iter)
            a2 = -1.0 + t * (-1.0 / self.max_iter)
            for i in range(self.pop_size):
                r1, r2, p = np.random.rand(), np.random.rand(), np.random.rand()
                A = 2*a*r1 - a
                C = 2*r2
                l = (a2 - 1)*np.random.rand() + 1
                b = 1
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_idx = np.random.randint(0, self.pop_size)
                        D = np.abs(C * positions[rand_idx] - positions[i])
                        positions[i] = positions[rand_idx] - A * D
                    else:
                        D = np.abs(C * best_pos - positions[i])
                        positions[i] = best_pos - A * D
                else:
                    dist = np.abs(best_pos - positions[i])
                    positions[i] = dist * np.exp(b*l) * np.cos(2*np.pi*l) + best_pos
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            fitness = self._evaluate(positions)
            curr_best = np.argmin(fitness)
            if fitness[curr_best] < self.best_fitness:
                self.best_fitness = fitness[curr_best]
                best_pos = positions[curr_best].copy()
                self.best_solution = best_pos.copy()
            self.history.append(self.best_fitness)
            self._record_time(start_time)
            if self.best_fitness < 1e-10:
                break
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class HHO(BaseOptimizer):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, seed: int = None):
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)

    def optimize(self):
        start_time = time.time()
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        best_idx = np.argmin(fitness)
        rabbit_pos = positions[best_idx].copy()
        rabbit_fitness = fitness[best_idx]
        self.best_solution, self.best_fitness = rabbit_pos.copy(), rabbit_fitness
        self.history.append(self.best_fitness)
        self._record_time(start_time)

        for t in range(self.max_iter):
            E0 = 2*np.random.rand() - 1
            E = 2*E0*(1 - t/self.max_iter)
            for i in range(self.pop_size):
                q = np.random.rand()
                if q >= 0.5:
                    positions[i] = (rabbit_pos - positions.mean(axis=0)) - np.random.rand() * (self.lb + np.random.rand()*(self.ub - self.lb))
                if abs(E) >= 1:
                    if np.random.rand() >= 0.5:
                        positions[i] = rabbit_pos - E * abs(rabbit_pos - positions[i])
                    else:
                        positions[i] = rabbit_pos - E * abs(rabbit_pos - positions[i])
                else:
                    J = 2*(1 - np.random.rand())
                    if np.random.rand() >= 0.5:
                        positions[i] = rabbit_pos - E * abs(J*rabbit_pos - positions[i])
                    else:
                        positions[i] = (rabbit_pos - E * abs(J*rabbit_pos - positions[i])) + np.random.randn(self.dim)*0.01
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            fitness = self._evaluate(positions)
            for i in range(self.pop_size):
                if fitness[i] < rabbit_fitness:
                    rabbit_fitness = fitness[i]
                    rabbit_pos = positions[i].copy()
            self.best_solution, self.best_fitness = rabbit_pos.copy(), rabbit_fitness
            self.history.append(self.best_fitness)
            self._record_time(start_time)
            if self.best_fitness < 1e-10:
                break
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class SMA(BaseOptimizer):
    def __init__(self, objective_func: Callable, dim: int, bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100, pop_size: int = 30, z: float = 0.03, seed: int = None):
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        self.z = z

    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)

        # Лучшая позиция
        best_idx = np.argmin(fitness)
        best_pos = positions[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.best_solution = best_pos.copy()
        self.history.append(self.best_fitness)
        self._record_time(start_time)

        for t in range(self.max_iter):
            # Сортировка по фитнесу
            idx = np.argsort(fitness)
            best_f = fitness[idx[0]]
            worst_f = fitness[idx[-1]]

            # Вычисление весов W с защитой от деления на ноль и логарифма от нуля
            W = np.zeros(self.pop_size)
            if best_f == worst_f:
                W[:] = 1.0
            else:
                for i in range(self.pop_size):
                    if i <= self.pop_size // 2:
                        diff = best_f - fitness[idx[i]]
                        # Защита от логарифма <=0
                        val = max(diff / (best_f - worst_f), 1e-10)
                        W[idx[i]] = 1 + np.random.rand() * np.log10(val + 1e-10)
                    else:
                        diff = fitness[idx[i]] - worst_f
                        val = max(diff / (best_f - worst_f), 1e-10)
                        W[idx[i]] = 1 - np.random.rand() * np.log10(val + 1e-10)
                # Нормализация весов
                maxW = np.max(W)
                if maxW > 0:
                    W = W / maxW
                else:
                    W = np.ones(self.pop_size)

            # Параметры a и b (ограничиваем по модулю 10)
            a = np.arctanh(1 - t / self.max_iter)
            a = np.clip(a, -10, 10)
            b = 1 - t / self.max_iter
            b = np.clip(b, -10, 10)

            new_positions = positions.copy()
            for i in range(self.pop_size):
                if np.random.rand() < self.z:
                    # Случайная инициализация (exploration)
                    new_positions[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    p = np.tanh(abs(fitness[i] - best_f))
                    # Генерация векторов vb и vc (для каждого измерения)
                    vb = np.random.uniform(-abs(a), abs(a), self.dim)
                    vc = np.random.uniform(-abs(b), abs(b), self.dim)
                    A = np.random.randint(0, self.pop_size)
                    B = np.random.randint(0, self.pop_size)
                    if np.random.rand() < p:
                        new_positions[i] = best_pos + vb * (W[i] * positions[A] - positions[B])
                    else:
                        new_positions[i] = vc * positions[i]

                # Проверка на NaN и Inf
                if np.any(np.isnan(new_positions[i])) or np.any(np.isinf(new_positions[i])):
                    new_positions[i] = np.random.uniform(self.lb, self.ub, self.dim)

                # Ограничение границами
                new_positions[i] = np.clip(new_positions[i], self.lb, self.ub)

            # Оценка новых позиций
            new_fitness = self._evaluate(new_positions)

            # Отбор (жадный)
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    positions[i] = new_positions[i]
                    fitness[i] = new_fitness[i]

            # Обновление лучшего решения
            curr_best = np.argmin(fitness)
            if fitness[curr_best] < self.best_fitness:
                self.best_fitness = fitness[curr_best]
                best_pos = positions[curr_best].copy()
                self.best_solution = best_pos.copy()

            self.history.append(self.best_fitness)
            self._record_time(start_time)

            if self.best_fitness < 1e-10:
                break

        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

