"""
Модуль с реализациями метаэвристических алгоритмов оптимизации.
Все алгоритмы имеют единый интерфейс для сравнения.
"""

import numpy as np
import math
import time
from typing import Callable, Tuple, List, Dict, Any

class BaseOptimizer:
    """Базовый класс для всех оптимизаторов"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 seed: int = None):
        """
        Args:
            objective_func: Целевая функция для минимизации
            dim: Размерность задачи
            bounds: Кортеж (lower_bounds, upper_bounds)
            max_iter: Максимальное количество итераций
            pop_size: Размер популяции
            seed: Seed для воспроизводимости
        """
        self.objective_func = objective_func
        self.dim = dim
        self.lb, self.ub = bounds
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.seed = seed
        
        # Проверка границ
        if not isinstance(self.lb, np.ndarray):
            self.lb = np.array([self.lb] * dim) if np.isscalar(self.lb) else np.array(self.lb)
        if not isinstance(self.ub, np.ndarray):
            self.ub = np.array([self.ub] * dim) if np.isscalar(self.ub) else np.array(self.ub)
        
        # Установка seed
        if seed is not None:
            np.random.seed(seed)
            
        # Метрики
        self.history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.execution_time = 0
        self.function_evaluations = 0
        
    def _evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Вычисление значений целевой функции для всех позиций"""
        fitness = np.array([self.objective_func(pos) for pos in positions])
        self.function_evaluations += len(positions)
        return fitness
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Основной метод оптимизации (должен быть переопределен)"""
        raise NotImplementedError("Метод optimize должен быть реализован")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики алгоритма"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'function_evaluations': self.function_evaluations,
            'convergence_history': self.history,
            'population_size': self.pop_size,
            'iterations': self.max_iter
        }


class PSO(BaseOptimizer):
    """Алгоритм роевой оптимизации (Particle Swarm Optimization)"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 w: float = 0.729,      # Инерция
                 c1: float = 1.49445,   # Когнитивный коэффициент
                 c2: float = 1.49445,   # Социальный коэффициент
                 seed: int = None):
        
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        
        # Оценка начальной популяции
        fitness = self._evaluate(positions)
        
        # Лучшие позиции для каждой частицы
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        # Глобальная лучшая позиция
        gbest_idx = np.argmin(pbest_fitness)
        self.best_fitness = pbest_fitness[gbest_idx]
        self.best_solution = pbest_positions[gbest_idx].copy()
        
        # Основной цикл
        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # Обновление скоростей
            velocities = (self.w * velocities + 
                         self.c1 * r1 * (pbest_positions - positions) + 
                         self.c2 * r2 * (self.best_solution - positions))
            
            # Ограничение скоростей
            velocity_max = 0.1 * (self.ub - self.lb)
            velocities = np.clip(velocities, -velocity_max, velocity_max)
            
            # Обновление позиций
            positions += velocities
            positions = np.clip(positions, self.lb, self.ub)
            
            # Оценка новых позиций
            fitness = self._evaluate(positions)
            
            # Обновление лучших позиций
            improved_idx = fitness < pbest_fitness
            pbest_positions[improved_idx] = positions[improved_idx]
            pbest_fitness[improved_idx] = fitness[improved_idx]
            
            # Обновление глобальной лучшей позиции
            current_best_idx = np.argmin(pbest_fitness)
            if pbest_fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = pbest_fitness[current_best_idx]
                self.best_solution = pbest_positions[current_best_idx].copy()
            
            # Сохранение истории
            self.history.append(self.best_fitness)
            
            # Ранняя остановка при достижении нужной точности
            if self.best_fitness < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class GWO(BaseOptimizer):
    """Алгоритм серых волков (Grey Wolf Optimizer)"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 seed: int = None):
        
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        
        # Индексы трех лучших волков (альфа, бета, дельта)
        sorted_idx = np.argsort(fitness)
        alpha_pos = positions[sorted_idx[0]].copy()
        beta_pos = positions[sorted_idx[1]].copy()
        delta_pos = positions[sorted_idx[2]].copy()
        
        alpha_score = fitness[sorted_idx[0]]
        beta_score = fitness[sorted_idx[1]]
        delta_score = fitness[sorted_idx[2]]
        
        self.best_solution = alpha_pos.copy()
        self.best_fitness = alpha_score
        
        # Основной цикл
        for iteration in range(self.max_iter):
            a = 2.0 - iteration * (2.0 / self.max_iter)  # a линейно уменьшается от 2 до 0
            
            for i in range(self.pop_size):
                # Коэффициенты для альфа, бета и дельта
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                X1 = alpha_pos - A1 * D_alpha
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                
                D_beta = np.abs(C2 * beta_pos - positions[i])
                X2 = beta_pos - A2 * D_beta
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                
                D_delta = np.abs(C3 * delta_pos - positions[i])
                X3 = delta_pos - A3 * D_delta
                
                # Обновление позиции волка
                positions[i] = (X1 + X2 + X3) / 3.0
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            # Оценка новых позиций
            fitness = self._evaluate(positions)
            
            # Обновление альфа, бета и дельта
            for i in range(self.pop_size):
                if fitness[i] < alpha_score:
                    alpha_score = fitness[i]
                    alpha_pos = positions[i].copy()
                elif fitness[i] < beta_score:
                    beta_score = fitness[i]
                    beta_pos = positions[i].copy()
                elif fitness[i] < delta_score:
                    delta_score = fitness[i]
                    delta_pos = positions[i].copy()
            
            self.best_solution = alpha_pos.copy()
            self.best_fitness = alpha_score
            
            # Сохранение истории
            self.history.append(self.best_fitness)
            
            # Ранняя остановка
            if self.best_fitness < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class WOA(BaseOptimizer):
    """Алгоритм китов (Whale Optimization Algorithm)"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 seed: int = None):
        
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        
        # Лучший кит
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.best_solution = best_position.copy()
        
        # Основной цикл
        for iteration in range(self.max_iter):
            a = 2.0 - iteration * (2.0 / self.max_iter)  # a уменьшается от 2 до 0
            a2 = -1.0 + iteration * (-1.0 / self.max_iter)  # a2 уменьшается от -1 до -2
            
            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                
                b = 1.0  # Константа для логарифмической спирали
                l = (a2 - 1.0) * np.random.rand() + 1.0
                
                p = np.random.rand()
                
                if p < 0.5:
                    if abs(A) >= 1:
                        # Случайный поиск
                        rand_idx = np.random.randint(0, self.pop_size)
                        rand_position = positions[rand_idx]
                        D = np.abs(C * rand_position - positions[i])
                        positions[i] = rand_position - A * D
                    else:
                        # Окружение добычи
                        D = np.abs(C * best_position - positions[i])
                        positions[i] = best_position - A * D
                else:
                    # Нападение по спирали
                    distance_to_best = np.abs(best_position - positions[i])
                    positions[i] = distance_to_best * np.exp(b * l) * np.cos(2.0 * np.pi * l) + best_position
                
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            # Оценка новых позиций
            fitness = self._evaluate(positions)
            
            # Обновление лучшей позиции
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                best_position = positions[current_best_idx].copy()
                self.best_solution = best_position.copy()
            
            # Сохранение истории
            self.history.append(self.best_fitness)
            
            # Ранняя остановка
            if self.best_fitness < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class HHO(BaseOptimizer):
    """Алгоритм ястребов Харриса (Harris Hawks Optimizer)"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 seed: int = None):
        
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        
        # Лучшая позиция
        best_idx = np.argmin(fitness)
        rabbit_pos = positions[best_idx].copy()
        rabbit_fitness = fitness[best_idx]
        
        self.best_solution = rabbit_pos.copy()
        self.best_fitness = rabbit_fitness
        
        # Основной цикл
        for iteration in range(self.max_iter):
            E0 = 2.0 * np.random.rand() - 1.0  # Начальная энергия
            E = 2.0 * E0 * (1.0 - iteration / self.max_iter)  # Энергия добычи
            
            for i in range(self.pop_size):
                q = np.random.rand()
                r = np.random.rand()
                
                # Пробег от добычи
                if q >= 0.5:
                    # Высокий полет
                    positions[i] = (rabbit_pos - positions.mean(axis=0)) - np.random.rand() * (
                        self.lb + np.random.rand() * (self.ub - self.lb))
                
                # Жесткая и мягкая осада
                if abs(E) >= 1:
                    # Жесткая осада
                    if r >= 0.5:
                        positions[i] = (rabbit_pos - positions[i]) - E * abs(
                            rabbit_pos - positions[i])
                    else:
                        positions[i] = rabbit_pos - E * abs(rabbit_pos - positions[i])
                else:
                    # Мягкая осада
                    J = 2.0 * (1.0 - np.random.rand())  # Случайная сила прыжка
                    if r >= 0.5:
                        positions[i] = rabbit_pos - E * abs(J * rabbit_pos - positions[i])
                    else:
                        positions[i] = (rabbit_pos - E * abs(J * rabbit_pos - positions[i]) + 
                                       np.random.randn(self.dim) * 0.01)
                
                positions[i] = np.clip(positions[i], self.lb, self.ub)
            
            # Оценка новых позиций
            fitness = self._evaluate(positions)
            
            # Обновление позиции кролика (добычи)
            for i in range(self.pop_size):
                if fitness[i] < rabbit_fitness:
                    rabbit_fitness = fitness[i]
                    rabbit_pos = positions[i].copy()
            
            self.best_solution = rabbit_pos.copy()
            self.best_fitness = rabbit_fitness
            
            # Сохранение истории
            self.history.append(self.best_fitness)
            
            # Ранняя остановка
            if self.best_fitness < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness


class SMA(BaseOptimizer):
    """Алгоритм слизевиков (Slime Mould Algorithm)"""
    
    def __init__(self, 
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_iter: int = 100,
                 pop_size: int = 30,
                 z: float = 0.03,  # Параметр z
                 seed: int = None):
        
        super().__init__(objective_func, dim, bounds, max_iter, pop_size, seed)
        self.z = z
        
    def optimize(self) -> Tuple[np.ndarray, float]:
        start_time = time.time()
        
        # Инициализация популяции
        positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = self._evaluate(positions)
        
        # Лучшая позиция
        best_idx = np.argmin(fitness)
        best_position = positions[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.best_solution = best_position.copy()
        
        # Основной цикл
        for iteration in range(self.max_iter):
            # Сортировка по фитнесу
            sorted_idx = np.argsort(fitness)
            best_fitness = fitness[sorted_idx[0]]
            worst_fitness = fitness[sorted_idx[-1]]
            
            # Вычисление веса W (с защитой от деления на ноль)
            W = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                if i <= self.pop_size // 2:
                    diff = best_fitness - fitness[sorted_idx[i]]
                    b = np.log10(abs(diff) + 1e-10)  # Защита от отрицательных значений
                    W[sorted_idx[i]] = 1.0 + np.random.rand() * b
                else:
                    diff = fitness[sorted_idx[i]] - worst_fitness
                    W[sorted_idx[i]] = 1.0 - np.random.rand() * np.log10(abs(diff) + 1e-10)
            
            # Нормализация весов (с защитой от деления на ноль)
            max_W = np.max(W)
            if max_W > 0:
                W = W / max_W
            else:
                W = np.ones_like(W)
            
            # Параметры для обновления позиций
            # Ограничиваем a, чтобы избежать переполнения
            a = np.arctanh(1.0 - (iteration / max(self.max_iter, 1)))
            b = 1.0 - iteration / self.max_iter
            
            # Ограничиваем a разумными пределами для предотвращения переполнения
            a = np.clip(a, -10, 10)
            
            new_positions = positions.copy()
            for i in range(self.pop_size):
                if np.random.rand() < self.z:
                    # Эксплорация
                    new_positions[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    p = np.tanh(abs(fitness[i] - best_fitness))
                    
                    # Ограничиваем диапазоны vb и vc
                    vb = np.random.uniform(-abs(a), abs(a), self.dim)
                    vc = np.random.uniform(-abs(b), abs(b), self.dim)
                    
                    A = np.random.randint(0, self.pop_size)
                    B = np.random.randint(0, self.pop_size)
                    
                    if np.random.rand() < p:
                        new_positions[i] = best_position + vb * (
                            W[i] * positions[A] - positions[B])
                    else:
                        new_positions[i] = vc * positions[i]
                
                # Ограничиваем позиции границами
                new_positions[i] = np.clip(new_positions[i], self.lb, self.ub)
            
            # Оценка новых позиций
            new_fitness = self._evaluate(new_positions)
            
            # Выбор лучших позиций
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    positions[i] = new_positions[i]
                    fitness[i] = new_fitness[i]
            
            # Обновление лучшей позиции
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = fitness[current_best_idx]
                best_position = positions[current_best_idx].copy()
                self.best_solution = best_position.copy()
            
            # Сохранение истории
            self.history.append(self.best_fitness)
            
            # Ранняя остановка
            if self.best_fitness < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness