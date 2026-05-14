import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Any
import scipy.integrate

from algorithms import PSO, GWO, WOA, HHO, SMA
from problems import get_problem_info, _rng_seed
from simulation import simulate_dc_motor_pid, compute_step_metrics


class ComparativeExperiment:
    def __init__(self, num_runs: int = 20, max_iter: int = 100, pop_size: int = 30,
                 output_dir: str = "results", auto_visualize: bool = True):
        self.num_runs = num_runs
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.output_dir = output_dir
        self.auto_visualize = auto_visualize
        os.makedirs(output_dir, exist_ok=True)

        self.algorithms = {'PSO': PSO, 'GWO': GWO, 'WOA': WOA, 'HHO': HHO, 'SMA': SMA}
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
        self.convergence_data = {}
        self.convergence_time_data = {}
        self.start_time = None

    def evaluate_solution(self, problem_name: str, solution: np.ndarray) -> Dict[str, Any]:
        if problem_name == 'dc_motor_pid':
            Kp, Ki, Kd = solution
            if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            try:
                t, y = simulate_dc_motor_pid(solution, t_end=5, n_points=500)
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                metrics = compute_step_metrics(t, y)
                feasible = True
                acceptable = (metrics['overshoot'] <= 10.0 and
                              metrics['settling_time'] <= 2.0 and
                              metrics['steady_state_error'] <= 0.02)
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}
            except:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}

        elif problem_name == 'inverted_pendulum':
            M, m, b, l, g = 1.0, 0.1, 0.1, 0.5, 9.81
            A = np.array([[0,1,0,0],[0,-b/M,-m*g/M,0],[0,0,0,1],[0,-b/(M*l),(M+m)*g/(M*l),0]])
            B = np.array([[0],[1/M],[0],[1/(M*l)]])
            K = np.array([[solution[0], solution[1], solution[2], solution[3]]])
            A_closed = A - B @ K
            feasible = np.all(np.real(np.linalg.eigvals(A_closed)) < 0)
            if not feasible:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
            def dyn(x, t): return A_closed.dot(x)
            x0 = [0,0,0.1,0]
            t_span = np.linspace(0,5,500)
            x = scipy.integrate.odeint(dyn, x0, t_span)
            final_angle = np.abs(x[-1,2])
            max_angle = np.max(np.abs(x[:,2]))
            acceptable = (final_angle < 0.05) and (max_angle < 0.2)
            metrics = {'final_angle': final_angle, 'max_angle': max_angle}
            return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}

        elif problem_name == 'liquid_level':
            try:
                Kp1, Ki1, Kp2, Ki2 = solution
                A1, A2 = 2.0, 1.5
                R1_base, R2_base = 0.5, 0.7
                dt = 0.1; steps = 200
                h1, h2 = 0.5, 0.3
                I1, I2 = 0.0, 0.0
                h1_traj, h2_traj = [h1], [h2]
                times = [0.0]
                total_iae = 0.0
                for k in range(steps):
                    t = k*dt
                    h1_ref, h2_ref = (1.0, 0.8) if t<10 else (0.9, 0.7)
                    e1, e2 = h1_ref - h1, h2_ref - h2
                    I1 += e1*dt; I2 += e2*dt
                    u1 = np.clip(Kp1*e1 + Ki1*I1, 0, 2)
                    u2 = np.clip(Kp2*e2 + Ki2*I2, 0, 2)
                    R1 = R1_base*(1+0.5*h1); R2 = R2_base*(1+0.5*h2)
                    q12 = max(0, (h1-h2)/R1)*np.sqrt(abs(h1-h2)+1e-6)
                    q2out = (h2/R2)*np.sqrt(h2+1e-6)
                    h1 += (u1 - q12)/A1*dt
                    h2 += (q12 - q2out + u2)/A2*dt
                    h1, h2 = max(0,h1), max(0,h2)
                    h1_traj.append(h1); h2_traj.append(h2)
                    times.append(t+dt)
                    total_iae += (abs(e1)+abs(e2))*dt
                feasible = (not np.any(np.isnan(h1_traj)) and not np.any(np.isnan(h2_traj)) and
                            not np.any(np.isinf(h1_traj)) and not np.any(np.isinf(h2_traj)))
                if not feasible:
                    return {'feasible': False, 'acceptable': False, 'metrics': {}}
                start_eval = times[-1]*0.8
                indices = [i for i,t in enumerate(times) if t >= start_eval] or range(len(times))
                h1_ref_traj = [1.0 if t<10 else 0.9 for t in times]
                h2_ref_traj = [0.8 if t<10 else 0.7 for t in times]
                e1 = np.array([h1_ref_traj[i]-h1_traj[i] for i in range(len(times))])
                e2 = np.array([h2_ref_traj[i]-h2_traj[i] for i in range(len(times))])
                mean_abs_error1 = np.mean(np.abs(e1[indices]))
                mean_abs_error2 = np.mean(np.abs(e2[indices]))
                final_error1 = abs(h1_ref_traj[-1]-h1_traj[-1])
                final_error2 = abs(h2_ref_traj[-1]-h2_traj[-1])
                acceptable = (mean_abs_error1 < 0.05 and mean_abs_error2 < 0.05 and
                              final_error1 < 0.05 and final_error2 < 0.05)
                metrics = {
                    'mean_abs_error1': mean_abs_error1,
                    'mean_abs_error2': mean_abs_error2,
                    'final_error1': final_error1,
                    'final_error2': final_error2,
                    'total_iae': total_iae
                }
                return {'feasible': feasible, 'acceptable': acceptable, 'metrics': metrics}
            except Exception as e:
                return {'feasible': False, 'acceptable': False, 'metrics': {}}
        else:
            return {'feasible': False, 'acceptable': False, 'metrics': {}}

    def run_single_experiment(self, algorithm_class, problem_info: Dict,
                              algorithm_name: str, problem_name: str, run_id: int) -> Dict[str, Any]:
        try:
            seed = 42 + run_id * 100
            dim = problem_info['dim']
            bounds = problem_info['bounds']
            objective_func = problem_info['objective_func']

            if problem_name == 'liquid_level':
                import problems
                problems._rng_seed = seed

            if algorithm_name == 'PSO':
                algorithm = algorithm_class(objective_func, dim, bounds, self.max_iter, self.pop_size,
                                            w=0.7, c1=1.5, c2=1.5, seed=seed)
            elif algorithm_name == 'SMA':
                algorithm = algorithm_class(objective_func, dim, bounds, self.max_iter, self.pop_size,
                                            z=0.03, seed=seed)
            else:
                algorithm = algorithm_class(objective_func, dim, bounds, self.max_iter, self.pop_size, seed=seed)

            best_solution, best_fitness = algorithm.optimize()
            metrics = algorithm.get_metrics()
            eval_result = self.evaluate_solution(problem_name, best_solution)

            # J-метрики (относительные)
            history = metrics.get('convergence_history', [])
            total_iters = self.max_iter
            if len(history) > 0:
                idx50 = min(int(0.5*(total_iters+1)), len(history)-1)
                idx75 = min(int(0.75*(total_iters+1)), len(history)-1)
                idx100 = min(total_iters, len(history)-1)
                j50 = history[idx50] if idx50>=0 else np.nan
                j75 = history[idx75] if idx75>=0 else np.nan
                j100 = history[idx100] if idx100>=0 else np.nan
            else:
                j50 = j75 = j100 = np.nan

            time_hist = metrics.get('time_history', [])
            metrics.update({
                'algorithm': algorithm_name, 'problem': problem_name, 'run_id': run_id, 'seed': seed,
                'solution': best_solution.tolist(), 'best_fitness': float(best_fitness),
                'feasible': eval_result['feasible'], 'acceptable': eval_result['acceptable'],
                'time_history': time_hist, 'metrics': eval_result['metrics'],
                'j50': j50, 'j75': j75, 'j100': j100
            })
            return metrics
        except Exception as e:
            return {
                'algorithm': algorithm_name, 'problem': problem_name, 'run_id': run_id,
                'best_fitness': float('inf'), 'execution_time': 0, 'function_evaluations': 0,
                'feasible': False, 'acceptable': False, 'time_history': [], 'metrics': {},
                'j50': np.nan, 'j75': np.nan, 'j100': np.nan, 'error': str(e)
            }

    def print_progress_bar(self, current, total, bar_length=50):
        percent = current/total
        arrow = '='*int(round(percent*bar_length))
        spaces = ' '*(bar_length - len(arrow))
        elapsed = time.time() - self.start_time
        eta = (elapsed/current)*(total-current) if current>0 else 0
        sys.stdout.write(f"\r[{arrow}{spaces}] {current}/{total} ({percent*100:.1f}%) | "
                         f"Прошло: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                         f"Осталось: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
        sys.stdout.flush()

    def run_all_experiments(self):
        self.start_time = time.time()
        print("="*90)
        print(" ЗАПУСК СРАВНИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ")
        print("="*90)
        print(f" Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Количество запусков: {self.num_runs}")
        print(f" Максимальное количество итераций: {self.max_iter}")
        print(f" Размер популяции: {self.pop_size}")
        print(f" Директория результатов: {self.output_dir}")
        print("="*90)

        total_experiments = len(self.algorithms)*len(self.problems)*self.num_runs
        current_exp = 0
        problem_times = {}

        for problem_name, problem_desc in self.problems.items():
            problem_start = time.time()
            print(f"\n{'='*60}\n ЗАДАЧА: {problem_desc}\n{'='*60}")
            problem_info = get_problem_info(problem_name)
            if not problem_info:
                print(f" Ошибка: задача '{problem_name}' не найдена")
                continue
            self.results[problem_name] = {}
            self.convergence_data[problem_name] = {}
            self.convergence_time_data[problem_name] = {}

            for algo_name, algo_class in self.algorithms.items():
                print(f"\n Алгоритм {algo_name}")
                print(" " + "-"*40)
                all_metrics = []
                conv_histories = []
                time_histories = []
                for run_id in range(self.num_runs):
                    current_exp += 1
                    self.print_progress_bar(current_exp, total_experiments)
                    metrics = self.run_single_experiment(algo_class, problem_info, algo_name, problem_name, run_id)
                    all_metrics.append(metrics)
                    conv_histories.append(metrics.get('convergence_history', []))
                    time_histories.append(metrics.get('time_history', []))
                print()

                feasible_runs = [m for m in all_metrics if m.get('feasible', False)]
                acceptable_runs = [m for m in all_metrics if m.get('acceptable', False)]

                if feasible_runs:
                    best_vals = [m['best_fitness'] for m in feasible_runs]
                    exec_times = [m['execution_time'] for m in feasible_runs]
                    mean_f = np.mean(best_vals); std_f = np.std(best_vals)
                    median_f = np.median(best_vals); q25_f = np.percentile(best_vals,25); q75_f = np.percentile(best_vals,75)
                    mean_t = np.mean(exec_times); std_t = np.std(exec_times)
                    feas_rate = len(feasible_runs)/len(all_metrics)*100
                    acc_rate = len(acceptable_runs)/len(all_metrics)*100

                    # Сбор инженерных метрик
                    metric_keys = set()
                    for m in feasible_runs: metric_keys.update(m.get('metrics',{}).keys())
                    metric_keys = sorted(metric_keys)
                    metrics_summary = {}
                    for key in metric_keys:
                        vals = [m['metrics'][key] for m in feasible_runs if key in m.get('metrics',{})]
                        if vals:
                            metrics_summary[f'{key}_mean'] = np.mean(vals)
                            metrics_summary[f'{key}_std'] = np.std(vals)
                            metrics_summary[f'{key}_median'] = np.median(vals)
                            metrics_summary[f'{key}_q25'] = np.percentile(vals,25)
                            metrics_summary[f'{key}_q75'] = np.percentile(vals,75)

                    # J-метрики
                    j50_vals = [m['j50'] for m in feasible_runs if not np.isnan(m.get('j50',np.nan))]
                    j75_vals = [m['j75'] for m in feasible_runs if not np.isnan(m.get('j75',np.nan))]
                    j100_vals = [m['j100'] for m in feasible_runs if not np.isnan(m.get('j100',np.nan))]
                    j50_med = np.median(j50_vals) if j50_vals else np.nan
                    j75_med = np.median(j75_vals) if j75_vals else np.nan
                    j100_med = np.median(j100_vals) if j100_vals else np.nan

                    print(f" {algo_name}: Фитнес={mean_f:.4e}±{std_f:.4e}, Время={mean_t:.3f}±{std_t:.3f} с, "
                          f"Feasible={feas_rate:.1f}%, Acceptable={acc_rate:.1f}%")

                    self.results[problem_name][algo_name] = {
                        'best_fitness_mean': float(mean_f), 'best_fitness_std': float(std_f),
                        'best_fitness_median': float(median_f), 'best_fitness_q25': float(q25_f),
                        'best_fitness_q75': float(q75_f), 'execution_time_mean': float(mean_t),
                        'execution_time_std': float(std_t), 'feasible_rate': feas_rate,
                        'acceptable_rate': acc_rate, 'all_runs': all_metrics,
                        'metrics_summary': metrics_summary,
                        'Median_J@50%': j50_med, 'Median_J@75%': j75_med, 'Median_J@100%': j100_med
                    }
                    self.convergence_data[problem_name][algo_name] = conv_histories
                    self.convergence_time_data[problem_name][algo_name] = time_histories
                else:
                    print(f" {algo_name}: Нет допустимых запусков")
                    self.results[problem_name][algo_name] = {
                        'best_fitness_mean': np.nan, 'best_fitness_std': np.nan,
                        'best_fitness_median': np.nan, 'best_fitness_q25': np.nan,
                        'best_fitness_q75': np.nan, 'execution_time_mean': np.nan,
                        'execution_time_std': np.nan, 'feasible_rate': 0.0, 'acceptable_rate': 0.0,
                        'all_runs': all_metrics, 'metrics_summary': {},
                        'Median_J@50%': np.nan, 'Median_J@75%': np.nan, 'Median_J@100%': np.nan
                    }
                    self.convergence_data[problem_name][algo_name] = []
                    self.convergence_time_data[problem_name][algo_name] = []

            problem_times[problem_name] = time.time() - problem_start
            print(f"\n Время выполнения задачи: {problem_times[problem_name]:.2f} с")

        self.save_results()
        self.print_final_statistics(problem_times)
        if self.auto_visualize:
            self.run_visualization()
        return self.results

    def save_results(self):
        print("\n" + "="*60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)

        summary_data = []
        for problem_name, algorithms in self.results.items():
            for algo_name, metrics in algorithms.items():
                if 'error' in metrics:
                    continue
                row = {
                    'Problem': self.problem_titles.get(problem_name, problem_name),
                    'Algorithm': algo_name,
                    'Best_Fitness_Mean': metrics.get('best_fitness_mean', np.nan),
                    'Best_Fitness_Std': metrics.get('best_fitness_std', np.nan),
                    'Best_Fitness_Median': metrics.get('best_fitness_median', np.nan),
                    'Best_Fitness_Q25': metrics.get('best_fitness_q25', np.nan),
                    'Best_Fitness_Q75': metrics.get('best_fitness_q75', np.nan),
                    'Execution_Time_Mean': metrics.get('execution_time_mean', np.nan),
                    'Execution_Time_Std': metrics.get('execution_time_std', np.nan),
                    'Feasible_Rate_%': metrics.get('feasible_rate', 0.0),
                    'Acceptable_Rate_%': metrics.get('acceptable_rate', 0.0),
                    'Median_J@50%': metrics.get('Median_J@50%', np.nan),
                    'Median_J@75%': metrics.get('Median_J@75%', np.nan),
                    'Median_J@100%': metrics.get('Median_J@100%', np.nan),
                }
                # Добавляем инженерные метрики
                for key, val in metrics.get('metrics_summary', {}).items():
                    row[key] = val
                summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_dir, "summary_results.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f" Сводные результаты: {summary_path}")

        detailed_path = os.path.join(self.output_dir, "results.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        print(f" Детальные результаты: {detailed_path}")

        convergence_path = os.path.join(self.output_dir, "convergence.json")
        with open(convergence_path, 'w') as f:
            json.dump(self.convergence_data, f, indent=2, default=str)
        print(f" Данные сходимости: {convergence_path}")

        time_path = os.path.join(self.output_dir, "convergence_time.json")
        with open(time_path, 'w') as f:
            json.dump(self.convergence_time_data, f, indent=2, default=str)
        print(f" Данные времени итераций: {time_path}")

        params = {
            'num_runs': self.num_runs, 'max_iter': self.max_iter, 'pop_size': self.pop_size,
            'timestamp': datetime.now().isoformat(), 'total_time': time.time() - self.start_time
        }
        params_path = os.path.join(self.output_dir, "params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f" Параметры эксперимента: {params_path}")

    def print_final_statistics(self, problem_times):
        total_time = time.time() - self.start_time
        print("\n" + "="*90)
        print("ИТОГОВАЯ СТАТИСТИКА ЭКСПЕРИМЕНТОВ")
        print("="*90)
        print("\n ВРЕМЯ ВЫПОЛНЕНИЯ:")
        for problem_name, p_time in problem_times.items():
            print(f" {self.problem_titles[problem_name]}: {p_time:.2f} с")
        print(f" ВСЕГО: {total_time:.2f} с ({total_time/60:.2f} мин)")

        print("\n ЛУЧШИЕ АЛГОРИТМЫ ПО ЗАДАЧАМ (по медиане фитнеса):")
        for problem_name in self.problems.keys():
            if problem_name in self.results:
                best_algo = None
                best_median = float('inf')
                for algo_name, metrics in self.results[problem_name].items():
                    med = metrics.get('best_fitness_median', np.nan)
                    if not np.isnan(med) and med < best_median:
                        best_median = med
                        best_algo = algo_name
                if best_algo:
                    print(f" {self.problem_titles[problem_name]}: {best_algo} (медиана={best_median:.4e})")

        print("\n" + "="*90)
        print(" ЭКСПЕРИМЕНТЫ УСПЕШНО ЗАВЕРШЕНЫ")
        print(f" Все результаты в папке: {self.output_dir}")
        print("="*90)

    def run_visualization(self):
        print("\n" + "="*60)
        print(" АВТОМАТИЧЕСКИЙ ЗАПУСК ВИЗУАЛИЗАЦИИ")
        print("="*60)
        import subprocess
        for viz_file in ['visualization.py', 'plot_all_step_responses.py']:
            if os.path.exists(viz_file):
                print(f"\n Запуск {viz_file}...")
                try:
                    result = subprocess.run([sys.executable, viz_file, '--results_dir', self.output_dir],
                                            capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f" {viz_file} выполнен успешно")
                    else:
                        print(f" Ошибка в {viz_file}: {result.stderr[:200]}")
                except Exception as e:
                    print(f" Не удалось запустить {viz_file}: {e}")
            else:
                print(f" Файл {viz_file} не найден, пропускаем")
        print("\n" + "="*60)
        print(" ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("="*60)


def main():
    experiment = ComparativeExperiment(num_runs=20, max_iter=100, pop_size=30,
                                       output_dir="results", auto_visualize=True)
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()