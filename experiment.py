"""
Основной скрипт для проведения сравнительных экспериментов.
Запускает все алгоритмы на всех задачах и собирает метрики.
Теперь поддерживает:
    - Два уровня успешности: Feasible и Acceptable.
    - Агрегирование истории сходимости по всем запускам.
    - Метрики по числу вычислений функции (FE до порога, J@FE) с интерполяцией.
    - Запись результатов для алгоритмов без допустимых запусков (Feasible_Rate_% = 0).
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Any
import scipy.integrate  # для odeint в evaluate_solution

from algorithms import PSO, GWO, WOA, HHO, SMA
from problems import get_problem_info, _rng_seed
from simulation import simulate_dc_motor_pid, compute_step_metrics


class ComparativeExperiment:
    """Класс для проведения сравнительных экспериментов"""
    
    def __init__(self, 
                 num_runs: int = 10,
                 max_iter: int = 100,
                 pop_size: int = 30,
                 output_dir: str = "experiment_results",
                 auto_visualize: bool = True):
        self.num_runs = num_runs
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.output_dir = output_dir
        self.auto_visualize = auto_visualize
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.algorithms = {
            'PSO': PSO,
            'GWO': GWO,
            'WOA': WOA,
            'HHO': HHO,
            'SMA': SMA
        }
        
        self.problems = {
            'dc_motor_pid': 'Оптимизация ПИД-регулятора (двигатель)',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Управление уровнем жидкости'
        }
        
        self.problem_titles = {
            'dc_motor_pid': 'ПИД-регулятор двигателя',
            'inverted_pendulum': 'Балансировка маятника',
            'liquid_level': 'Уровень жидкости'
        }
        
        self.results = {}
        self.convergence_data = {}   # для каждого алгоритма и задачи храним список историй
        self.start_time = None
    
    def evaluate_solution(self, problem_name: str, solution: np.ndarray) -> Dict[str, Any]:
        """
        Пост-проверка решения: определяет допустимость и приемлемость,
        а также вычисляет инженерные метрики.
        """
        if problem_name == 'dc_motor_pid':
            Kp, Ki, Kd = solution
            # Допустимость: параметры в границах и моделирование прошло без ошибок
            if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            try:
                t, y = simulate_dc_motor_pid(solution, t_end=5, n_points=500)
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                metrics = compute_step_metrics(t, y)
                feasible = True
                # Приемлемость: перерегулирование ≤ 10%, время установления ≤ 2 с, уст. ошибка ≤ 0.02
                acceptable = (metrics['overshoot'] <= 10.0 and 
                              metrics['settling_time'] <= 2.0 and
                              metrics['steady_state_error'] <= 0.02)
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}
            except:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
        
        elif problem_name == 'inverted_pendulum':
            # Параметры маятника (те же, что в problems.py)
            M, m, b, l, g = 1.0, 0.1, 0.1, 0.5, 9.81
            A = np.array([
                [0, 1, 0, 0],
                [0, -b/M, -m*g/M, 0],
                [0, 0, 0, 1],
                [0, -b/(M*l), (M+m)*g/(M*l), 0]
            ])
            B = np.array([[0], [1/M], [0], [1/(M*l)]])
            K = np.array([[solution[0], solution[1], solution[2], solution[3]]])
            A_closed = A - B @ K
            eigvals = np.linalg.eigvals(A_closed)
            feasible = np.all(np.real(eigvals) < 0)
            if not feasible:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            # Моделируем
            def dyn(x, t):
                return A_closed.dot(x)
            x0 = [0, 0, 0.1, 0]
            t_span = np.linspace(0, 5, 500)
            x = scipy.integrate.odeint(dyn, x0, t_span)
            # Приемлемость: максимальное отклонение угла < 0.2 рад и финальное отклонение < 0.05
            final_angle = np.abs(x[-1, 2])
            max_angle = np.max(np.abs(x[:, 2]))
            acceptable = (final_angle < 0.05) and (max_angle < 0.2)
            return {'feasible': feasible, 'acceptable': acceptable, 'metrics': {'final_angle': final_angle, 'max_angle': max_angle}}
        
        elif problem_name == 'liquid_level':
            try:
                # Детерминированная симуляция без шума для оценки
                def simulate_no_noise(params):
                    Kp1, Ki1, Kp2, Ki2 = params
                    # Параметры системы (те же, что в objective)
                    A1, A2 = 2.0, 1.5
                    R1_base, R2_base = 0.5, 0.7
                    dt = 0.1
                    steps = 200
                    h1, h2 = 0.5, 0.3
                    I1, I2 = 0.0, 0.0
                    u1_prev, u2_prev = 0.0, 0.0
                    # Для хранения траекторий
                    h1_traj = [h1]
                    h2_traj = [h2]
                    times = [0.0]
                    
                    for k in range(steps):
                        t = k * dt
                        if t < 10:
                            h1_ref, h2_ref = 1.0, 0.8
                        else:
                            h1_ref, h2_ref = 0.9, 0.7
                        
                        # Ошибки без шума
                        e1 = h1_ref - h1
                        e2 = h2_ref - h2
                        I1 += e1 * dt
                        I2 += e2 * dt
                        u1 = np.clip(Kp1 * e1 + Ki1 * I1, 0, 2)
                        u2 = np.clip(Kp2 * e2 + Ki2 * I2, 0, 2)
                        
                        # Нелинейные потоки
                        R1 = R1_base * (1 + 0.5 * h1)
                        R2 = R2_base * (1 + 0.5 * h2)
                        q12 = max(0, (h1 - h2) / R1) * np.sqrt(abs(h1 - h2) + 1e-6)
                        q2out = (h2 / R2) * np.sqrt(h2 + 1e-6)
                        
                        h1 += (u1 - q12) / A1 * dt
                        h2 += (q12 - q2out + u2) / A2 * dt
                        h1 = max(0, h1)
                        h2 = max(0, h2)
                        
                        h1_traj.append(h1)
                        h2_traj.append(h2)
                        times.append(t + dt)
                    
                    return times, h1_traj, h2_traj
                
                # Выполняем симуляцию
                times, h1_traj, h2_traj = simulate_no_noise(solution)
                
                # Проверка допустимости
                feasible = (not np.any(np.isnan(h1_traj)) and 
                            not np.any(np.isnan(h2_traj)) and
                            not np.any(np.isinf(h1_traj)) and
                            not np.any(np.isinf(h2_traj)) and
                            np.all(np.array(h1_traj) >= -0.01) and
                            np.all(np.array(h2_traj) >= -0.01))
                
                if not feasible:
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                
                # Вычисляем метрики для приемлемости
                last_t = times[-1]
                start_eval = last_t * 0.8  # последние 20% времени
                indices = [i for i, t in enumerate(times) if t >= start_eval]
                if not indices:
                    indices = range(len(times))
                
                # Определяем уставки для каждого момента
                h1_ref_traj = []
                h2_ref_traj = []
                for t in times:
                    if t < 10:
                        h1_ref_traj.append(1.0)
                        h2_ref_traj.append(0.8)
                    else:
                        h1_ref_traj.append(0.9)
                        h2_ref_traj.append(0.7)
                
                # Ошибки
                e1 = np.array([h1_ref_traj[i] - h1_traj[i] for i in range(len(times))])
                e2 = np.array([h2_ref_traj[i] - h2_traj[i] for i in range(len(times))])
                
                # Средняя абсолютная ошибка на последних 20%
                mean_abs_error1 = np.mean(np.abs(e1[indices]))
                mean_abs_error2 = np.mean(np.abs(e2[indices]))
                
                # Финальная ошибка
                final_error1 = abs(h1_ref_traj[-1] - h1_traj[-1])
                final_error2 = abs(h2_ref_traj[-1] - h2_traj[-1])
                
                # Приемлемость: средняя ошибка на последних 20% < 0.05 и финальная ошибка < 0.05
                acceptable = (mean_abs_error1 < 0.05 and mean_abs_error2 < 0.05 and
                              final_error1 < 0.05 and final_error2 < 0.05)
                
                metrics = {
                    'mean_abs_error1': mean_abs_error1,
                    'mean_abs_error2': mean_abs_error2,
                    'final_error1': final_error1,
                    'final_error2': final_error2
                }
                
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}
                
            except Exception as e:
                print(f"Ошибка при оценке liquid_level: {e}")
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
    
    def run_single_experiment(self, 
                             algorithm_class,
                             problem_info: Dict,
                             algorithm_name: str,
                             problem_name: str,
                             run_id: int) -> Dict[str, Any]:
        """Запуск одного эксперимента с возвратом метрик, включая флаги допустимости"""
        try:
            seed = 42 + run_id * 100
            dim = problem_info['dim']
            bounds = problem_info['bounds']
            objective_func = problem_info['objective_func']
            
            # Устанавливаем глобальный seed для задачи уровня жидкости (чтобы шум был фиксирован)
            if problem_name == 'liquid_level':
                import problems
                problems._rng_seed = seed
            
            if algorithm_name == 'PSO':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                    seed=seed
                )
            elif algorithm_name == 'SMA':
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    z=0.03,
                    seed=seed
                )
            else:
                algorithm = algorithm_class(
                    objective_func=objective_func,
                    dim=dim,
                    bounds=bounds,
                    max_iter=self.max_iter,
                    pop_size=self.pop_size,
                    seed=seed
                )
            
            best_solution, best_fitness = algorithm.optimize()
            metrics = algorithm.get_metrics()
            
            # Пост-проверка решения
            eval_result = self.evaluate_solution(problem_name, best_solution)
            
            # Вычисляем FE до порога (если порог достигнут) с использованием интерполяции
            target_fitness = 1e-3  # порог для всех задач
            fe_to_target = None
            if 'convergence_history' in metrics and metrics['convergence_history']:
                hist = metrics['convergence_history']  # список лучших фитнесов после каждой оценки (включая начальную)
                # FE для каждого элемента history: (i+1)*pop_size, где i начинается с 0
                fe_points = np.array([self.pop_size * (i+1) for i in range(len(hist))])
                # Если целевой фитнес достигнут, находим минимальное FE
                for i, val in enumerate(hist):
                    if val <= target_fitness:
                        # Интерполируем между текущим и предыдущим, если нужно
                        if i > 0 and val < target_fitness:
                            # Линейная интерполяция между (fe_points[i-1], hist[i-1]) и (fe_points[i], hist[i])
                            fe1, fe2 = fe_points[i-1], fe_points[i]
                            v1, v2 = hist[i-1], hist[i]
                            if v2 != v1:
                                fe_to_target = fe1 + (target_fitness - v1) * (fe2 - fe1) / (v2 - v1)
                            else:
                                fe_to_target = fe1
                        else:
                            fe_to_target = fe_points[i]
                        break
            
            # Значения на отсечках бюджета с линейной интерполяцией
            J_at_500 = None
            J_at_1000 = None
            J_at_2000 = None
            if 'convergence_history' in metrics and metrics['convergence_history']:
                hist = metrics['convergence_history']
                fe_points = np.array([self.pop_size * (i+1) for i in range(len(hist))])
                
                def interpolate_fitness(target_fe):
                    if target_fe <= fe_points[0]:
                        return hist[0]
                    if target_fe >= fe_points[-1]:
                        return hist[-1]
                    idx = np.searchsorted(fe_points, target_fe)
                    if idx == 0:
                        return hist[0]
                    # Линейная интерполяция между fe_points[idx-1] и fe_points[idx]
                    fe1, fe2 = fe_points[idx-1], fe_points[idx]
                    v1, v2 = hist[idx-1], hist[idx]
                    if fe2 == fe1:
                        return v1
                    return v1 + (target_fe - fe1) * (v2 - v1) / (fe2 - fe1)
                
                J_at_500 = interpolate_fitness(500)
                J_at_1000 = interpolate_fitness(1000)
                J_at_2000 = interpolate_fitness(2000)
            
            metrics.update({
                'algorithm': algorithm_name,
                'problem': problem_name,
                'run_id': run_id,
                'seed': seed,
                'solution': best_solution.tolist(),
                'best_fitness': float(best_fitness),
                'feasible': eval_result['feasible'],
                'acceptable': eval_result['acceptable'],
                'fe_to_target': fe_to_target,
                'J_at_500': J_at_500,
                'J_at_1000': J_at_1000,
                'J_at_2000': J_at_2000
            })
            return metrics
            
        except Exception as e:
            print(f"   Ошибка в {algorithm_name}: {str(e)[:100]}...")
            return {
                'algorithm': algorithm_name,
                'problem': problem_name,
                'run_id': run_id,
                'best_fitness': float('inf'),
                'execution_time': 0,
                'function_evaluations': 0,
                'feasible': False,
                'acceptable': False,
                'error': str(e)
            }
    
    def print_progress_bar(self, current, total, bar_length=50):
        percent = current / total
        arrow = '=' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(arrow))
        
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = (elapsed / current) * (total - current)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "??:??:??"
        
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
        sys.stdout.write(f"\r[{arrow}{spaces}] {current}/{total} "
                        f"({percent*100:.1f}%) | "
                        f"Прошло: {elapsed_str} | Осталось: {eta_str}")
        sys.stdout.flush()
    
    def run_all_experiments(self):
        self.start_time = time.time()
        
        print("=" * 90)
        print("  ЗАПУСК СРАВНИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        print(f" Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Количество запусков: {self.num_runs}")
        print(f" Максимальное количество итераций: {self.max_iter}")
        print(f" Размер популяции: {self.pop_size}")
        print(f" Директория результатов: {self.output_dir}")
        print("=" * 90)
        
        total_experiments = len(self.algorithms) * len(self.problems) * self.num_runs
        current_experiment = 0
        
        problem_times = {}
        
        for problem_idx, (problem_name, problem_description) in enumerate(self.problems.items(), 1):
            problem_start = time.time()
            
            print(f"\n{'='*60}")
            print(f" ЗАДАЧА {problem_idx}/{len(self.problems)}: {problem_description}")
            print(f"{'='*60}")
            
            problem_info = get_problem_info(problem_name)
            if problem_info is None:
                print(f" Ошибка: задача '{problem_name}' не найдена")
                continue
            
            self.results[problem_name] = {}
            self.convergence_data[problem_name] = {}  # здесь будем хранить список историй для каждого алгоритма
            
            for algo_idx, (algorithm_name, algorithm_class) in enumerate(self.algorithms.items(), 1):
                print(f"\n  Алгоритм {algo_idx}/{len(self.algorithms)}: {algorithm_name}")
                print("  " + "-" * 40)
                
                all_metrics = []
                convergence_histories = []  # список списков best_fitness по запускам
                
                for run_id in range(self.num_runs):
                    current_experiment += 1
                    self.print_progress_bar(current_experiment, total_experiments)
                    
                    metrics = self.run_single_experiment(
                        algorithm_class=algorithm_class,
                        problem_info=problem_info,
                        algorithm_name=algorithm_name,
                        problem_name=problem_name,
                        run_id=run_id
                    )
                    all_metrics.append(metrics)
                    if 'convergence_history' in metrics and metrics['convergence_history']:
                        convergence_histories.append(metrics['convergence_history'])
                    else:
                        convergence_histories.append([])
                
                print()
                
                # Фильтруем успешные (feasible) запуски для вычисления статистики
                feasible_runs = [m for m in all_metrics if m.get('feasible', False)]
                acceptable_runs = [m for m in all_metrics if m.get('acceptable', False)]
                
                if feasible_runs:
                    best_fitness_values = [m['best_fitness'] for m in feasible_runs]
                    execution_times = [m['execution_time'] for m in feasible_runs]
                    
                    mean_fitness = np.mean(best_fitness_values)
                    std_fitness = np.std(best_fitness_values)
                    median_fitness = np.median(best_fitness_values)
                    q25_fitness = np.percentile(best_fitness_values, 25)
                    q75_fitness = np.percentile(best_fitness_values, 75)
                    
                    mean_time = np.mean(execution_times)
                    std_time = np.std(execution_times)
                    
                    # FE до порога
                    fe_to_target_list = [m.get('fe_to_target', None) for m in feasible_runs if m.get('fe_to_target') is not None]
                    median_fe_to_target = np.median(fe_to_target_list) if fe_to_target_list else np.nan
                    
                    # J@FE
                    J500_list = [m.get('J_at_500', None) for m in feasible_runs if m.get('J_at_500') is not None]
                    median_J500 = np.median(J500_list) if J500_list else np.nan
                    J1000_list = [m.get('J_at_1000', None) for m in feasible_runs if m.get('J_at_1000') is not None]
                    median_J1000 = np.median(J1000_list) if J1000_list else np.nan
                    J2000_list = [m.get('J_at_2000', None) for m in feasible_runs if m.get('J_at_2000') is not None]
                    median_J2000 = np.median(J2000_list) if J2000_list else np.nan
                    
                    feasible_rate = len(feasible_runs) / len(all_metrics) * 100
                    acceptable_rate = len(acceptable_runs) / len(all_metrics) * 100
                    
                    # Вывод
                    status = "" if acceptable_rate > 50 else "" if feasible_rate > 0 else ""
                    print(f"  {status} {algorithm_name}: "
                          f"Фитнес = {mean_fitness:.4e} ± {std_fitness:.4e}, "
                          f"Время = {mean_time:.3f} ± {std_time:.3f} с, "
                          f"Feasible={feasible_rate:.1f}%, Acceptable={acceptable_rate:.1f}%")
                    
                    self.results[problem_name][algorithm_name] = {
                        'best_fitness_mean': float(mean_fitness),
                        'best_fitness_std': float(std_fitness),
                        'best_fitness_median': float(median_fitness),
                        'best_fitness_q25': float(q25_fitness),
                        'best_fitness_q75': float(q75_fitness),
                        'execution_time_mean': float(mean_time),
                        'execution_time_std': float(std_time),
                        'feasible_rate': feasible_rate,
                        'acceptable_rate': acceptable_rate,
                        'median_fe_to_target': median_fe_to_target,
                        'median_J_at_500': median_J500,
                        'median_J_at_1000': median_J1000,
                        'median_J_at_2000': median_J2000,
                        'all_runs': all_metrics
                    }
                    
                    # Сохраняем истории сходимости для всех запусков
                    self.convergence_data[problem_name][algorithm_name] = convergence_histories
                else:
                    # Нет допустимых запусков – записываем нули и NaN
                    print(f"   {algorithm_name}: Нет допустимых запусков")
                    self.results[problem_name][algorithm_name] = {
                        'best_fitness_mean': np.nan,
                        'best_fitness_std': np.nan,
                        'best_fitness_median': np.nan,
                        'best_fitness_q25': np.nan,
                        'best_fitness_q75': np.nan,
                        'execution_time_mean': np.nan,
                        'execution_time_std': np.nan,
                        'feasible_rate': 0.0,
                        'acceptable_rate': 0.0,
                        'median_fe_to_target': np.nan,
                        'median_J_at_500': np.nan,
                        'median_J_at_1000': np.nan,
                        'median_J_at_2000': np.nan,
                        'all_runs': all_metrics
                    }
                    self.convergence_data[problem_name][algorithm_name] = []
            
            problem_times[problem_name] = time.time() - problem_start
            print(f"\n   Время выполнения задачи: {problem_times[problem_name]:.2f} с")
        
        self.save_results()
        self.print_final_statistics(problem_times)
        
        if self.auto_visualize:
            self.run_visualization()
        
        return self.results
    
    def save_results(self):
        print("\n" + "=" * 60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        # Сводная таблица CSV
        summary_data = []
        for problem_name, algorithms in self.results.items():
            for algorithm_name, metrics in algorithms.items():
                # Если есть 'error', пропускаем (но теперь у нас всегда есть метрики)
                if 'error' in metrics:
                    continue
                summary_data.append({
                    'Problem': self.problem_titles.get(problem_name, problem_name),
                    'Algorithm': algorithm_name,
                    'Best_Fitness_Mean': metrics.get('best_fitness_mean', np.nan),
                    'Best_Fitness_Std': metrics.get('best_fitness_std', np.nan),
                    'Best_Fitness_Median': metrics.get('best_fitness_median', np.nan),
                    'Best_Fitness_Q25': metrics.get('best_fitness_q25', np.nan),
                    'Best_Fitness_Q75': metrics.get('best_fitness_q75', np.nan),
                    'Execution_Time_Mean': metrics.get('execution_time_mean', np.nan),
                    'Execution_Time_Std': metrics.get('execution_time_std', np.nan),
                    'Feasible_Rate_%': metrics.get('feasible_rate', 0.0),
                    'Acceptable_Rate_%': metrics.get('acceptable_rate', 0.0),
                    'Median_FE_to_Target': metrics.get('median_fe_to_target', np.nan),
                    'Median_J@500': metrics.get('median_J_at_500', np.nan),
                    'Median_J@1000': metrics.get('median_J_at_1000', np.nan),
                    'Median_J@2000': metrics.get('median_J_at_2000', np.nan)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_dir, "summary_results.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f" Сводные результаты: {summary_path}")
        
        # Детальные результаты JSON
        detailed_path = os.path.join(self.output_dir, "results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        print(f" Детальные результаты: {detailed_path}")
        
        # Данные сходимости JSON (список историй по алгоритмам)
        convergence_path = os.path.join(self.output_dir, "convergence.json")
        with open(convergence_path, 'w', encoding='utf-8') as f:
            json.dump(self.convergence_data, f, indent=2, default=str, ensure_ascii=False)
        print(f" Данные сходимости: {convergence_path}")
        
        # Параметры эксперимента
        experiment_params = {
            'num_runs': self.num_runs,
            'max_iter': self.max_iter,
            'pop_size': self.pop_size,
            'timestamp': datetime.now().isoformat(),
            'total_time': time.time() - self.start_time
        }
        params_path = os.path.join(self.output_dir, "params.json")
        with open(params_path, 'w') as f:
            json.dump(experiment_params, f, indent=2)
        print(f" Параметры эксперимента: {params_path}")
    
    def print_final_statistics(self, problem_times):
        total_time = time.time() - self.start_time
        print("\n" + "=" * 90)
        print("ИТОГОВАЯ СТАТИСТИКА ЭКСПЕРИМЕНТОВ")
        print("=" * 90)
        print(f"\n ВРЕМЯ ВЫПОЛНЕНИЯ:")
        print("-" * 40)
        for problem_name, p_time in problem_times.items():
            title = self.problem_titles.get(problem_name, problem_name)
            print(f"  {title}: {p_time:.2f} с")
        print(f"  {'='*30}")
        print(f"  ВСЕГО: {total_time:.2f} с ({total_time/60:.2f} мин)")
        
        print(f"\n ЛУЧШИЕ АЛГОРИТМЫ ПО ЗАДАЧАМ (по медиане фитнеса):")
        print("-" * 40)
        for problem_name in self.problems.keys():
            if problem_name in self.results:
                problem_results = self.results[problem_name]
                best_algo = None
                best_median = float('inf')
                for algo_name, metrics in problem_results.items():
                    if 'error' not in metrics and not np.isnan(metrics.get('best_fitness_median', np.nan)):
                        if metrics['best_fitness_median'] < best_median:
                            best_median = metrics['best_fitness_median']
                            best_algo = algo_name
                if best_algo:
                    title = self.problem_titles.get(problem_name, problem_name)
                    print(f"  {title}: {best_algo} (медиана фитнеса={best_median:.4e})")
        print("\n" + "=" * 90)
        print(" ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print(f" Все результаты в папке: {self.output_dir}")
        print("=" * 90)
    
    def run_visualization(self):
        print("\n" + "=" * 60)
        print(" АВТОМАТИЧЕСКИЙ ЗАПУСК ВИЗУАЛИЗАЦИИ")
        print("=" * 60)
        viz_files = ['visualization.py', 'plot_step_responses.py']
        for viz_file in viz_files:
            if os.path.exists(viz_file):
                print(f"\n  Запуск {viz_file}...")
                try:
                    import subprocess
                    result = subprocess.run([sys.executable, viz_file], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   {viz_file} выполнен успешно")
                        if viz_file == 'visualization.py':
                            print(f"     Графики сохранены в папке: plots/")
                        elif viz_file == 'plot_step_responses.py':
                            print(f"     Графики: step_response_comparison.png")
                            print(f"     Метрики: step_response_metrics.csv")
                    else:
                        print(f"   Ошибка в {viz_file}:")
                        print(f"     {result.stderr[:200]}")
                except Exception as e:
                    print(f"   Не удалось запустить {viz_file}: {e}")
            else:
                print(f"   Файл {viz_file} не найден, пропускаем")
        print("\n" + "=" * 60)
        print(" ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

def main():
    experiment = ComparativeExperiment(
        num_runs=10,
        max_iter=100,
        pop_size=30,
        output_dir="experiment_results",
        auto_visualize=True
    )
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()