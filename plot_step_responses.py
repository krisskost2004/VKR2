"""
Скрипт для построения графиков переходных процессов двигателя постоянного тока
с оптимальными параметрами ПИД-регулятора, найденными каждым алгоритмом.
Использует единую функцию моделирования из simulation.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from simulation import simulate_dc_motor_pid, compute_step_metrics

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (14, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300
})

colors = {
    'PSO': '#FF6B6B',
    'GWO': '#4ECDC4',
    'WOA': '#45B7D1',
    'HHO': '#96CEB4',
    'SMA': '#FFEAA7'
}

def load_best_solutions():
    """Загружает лучшие решения для задачи dc_motor_pid из результатов эксперимента"""
    results_file = "experiment_results/results.json"
    if not os.path.exists(results_file):
        print(f"❌ Файл {results_file} не найден. Сначала запустите experiment.py")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'dc_motor_pid' not in data:
        print("❌ Данные для задачи dc_motor_pid не найдены")
        return None
    
    motor_data = data['dc_motor_pid']
    best_solutions = {}
    
    print("\n📊 Загрузка лучших решений для каждого алгоритма:")
    print("-" * 60)
    
    for algo_name, algo_data in motor_data.items():
        if 'error' not in algo_data and 'all_runs' in algo_data:
            runs = algo_data['all_runs']
            best_run = min(runs, key=lambda x: x.get('best_fitness', float('inf')))
            if 'solution' in best_run:
                best_solutions[algo_name] = {
                    'params': best_run['solution'],
                    'fitness': best_run['best_fitness']
                }
                print(f"  {algo_name}: Kp={best_run['solution'][0]:.4f}, "
                      f"Ki={best_run['solution'][1]:.4f}, Kd={best_run['solution'][2]:.4f}, "
                      f"фитнес={best_run['best_fitness']:.6e}")
    
    print("-" * 60)
    return best_solutions


def plot_step_responses(best_solutions):
    """Строит графики переходных процессов для всех алгоритмов"""
    if not best_solutions:
        print("❌ Нет данных для построения графиков")
        return None
    
    t = np.linspace(0, 5, 1000)  # теперь 5 секунд, согласовано с objective
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Сравнение переходных процессов двигателя постоянного тока\nс оптимальными параметрами ПИД-регуляторов', 
                fontsize=16, fontweight='bold')
    
    ax1.set_title('Переходная характеристика (0-5 с)')
    ax1.set_xlabel('Время, с')
    ax1.set_ylabel('Угловая скорость (нормированная)')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Установившееся значение')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 1.2)
    
    ax2.set_title('Начальный участок (0-1 с)')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Угловая скорость (нормированная)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.2)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    all_metrics = {}
    
    for algo_name, data in best_solutions.items():
        Kp, Ki, Kd = data['params']
        fitness = data['fitness']
        
        t_out, y_out = simulate_dc_motor_pid([Kp, Ki, Kd], t_end=5, n_points=1000)
        metrics = compute_step_metrics(t_out, y_out)
        all_metrics[algo_name] = {**metrics, 'fitness': fitness, 'Kp': Kp, 'Ki': Ki, 'Kd': Kd}
        
        color = colors.get(algo_name, 'gray')
        label = f"{algo_name} (фитнес={fitness:.2e})"
        ax1.plot(t_out, y_out, label=label, color=color, linewidth=2)
        ax2.plot(t_out, y_out, color=color, linewidth=2)
    
    ax1.legend(loc='lower right', fontsize=9)
    
    # Таблица метрик
    metrics_text = "Метрики переходных процессов:\n"
    metrics_text += "-" * 80 + "\n"
    metrics_text += f"{'Алгоритм':<8} {'Перерег. %':<12} {'Время нараст. (с)':<16} {'Время устан. (с)':<16} {'Уст. ошибка':<12}\n"
    metrics_text += "-" * 80 + "\n"
    
    for algo_name, m in all_metrics.items():
        metrics_text += f"{algo_name:<8} {m['overshoot']:<12.2f} {m['rise_time']:<16.4f} {m['settling_time']:<16.4f} {m['steady_state_error']:<12.4e}\n"
    
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    output_file = "step_response_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ График переходных процессов сохранен: {output_file}")
    plt.show()
    return all_metrics


def create_metrics_table(metrics):
    """Создает CSV-файл с метриками переходных процессов"""
    if not metrics:
        return
    
    data = []
    for algo_name, m in metrics.items():
        data.append({
            'Algorithm': algo_name,
            'Kp': m['Kp'],
            'Ki': m['Ki'],
            'Kd': m['Kd'],
            'Fitness': m['fitness'],
            'Overshoot_%': m['overshoot'],
            'Rise_Time_s': m['rise_time'],
            'Settling_Time_s': m['settling_time'],
            'Steady_State_Error': m['steady_state_error']
        })
    
    df = pd.DataFrame(data).sort_values('Fitness')
    output_file = "step_response_metrics.csv"
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"✅ Метрики сохранены: {output_file}")
    print("\n" + "="*80)
    print("📊 МЕТРИКИ ПЕРЕХОДНЫХ ПРОЦЕССОВ")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    return df


def main():
    print("=" * 80)
    print("🔄 ПОСТРОЕНИЕ ГРАФИКОВ ПЕРЕХОДНЫХ ПРОЦЕССОВ")
    print("=" * 80)
    
    best_solutions = load_best_solutions()
    if not best_solutions:
        print("\n❌ Не удалось загрузить решения. Убедитесь, что эксперименты выполнены.")
        print("   Сначала запустите: python experiment.py")
        return
    
    print("\n📈 Построение графиков...")
    metrics = plot_step_responses(best_solutions)
    if metrics:
        create_metrics_table(metrics)
    
    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)

if __name__ == "__main__":
    main()