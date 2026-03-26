"""
Модуль для визуализации результатов экспериментов.
Теперь строит графики сходимости с медианой и межквартильным интервалом по всем запускам.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

problem_titles = {
    'dc_motor_pid': 'ПИД-регулятор двигателя',
    'inverted_pendulum': 'Балансировка маятника',
    'liquid_level': 'Уровень жидкости'
}

colors = {
    'PSO': '#FF6B6B',
    'GWO': '#4ECDC4',
    'WOA': '#45B7D1',
    'HHO': '#96CEB4',
    'SMA': '#FFEAA7'
}

def load_results():
    results = {}
    summary_file = "experiment_results/summary_results.csv"
    if os.path.exists(summary_file):
        results['summary'] = pd.read_csv(summary_file)
        print(f"Загружены сводные результаты из: {summary_file}")
    else:
        print(f"Файл {summary_file} не найден")
    
    convergence_file = "experiment_results/convergence.json"
    if os.path.exists(convergence_file):
        with open(convergence_file, 'r') as f:
            results['convergence'] = json.load(f)
        print(f"Загружены данные сходимости из: {convergence_file}")
    else:
        print(f"Файл {convergence_file} не найден")
    
    return results

def plot_convergence(results, save_dir="plots"):
    """Построение графиков сходимости с медианой и IQR по всем запускам"""
    if 'convergence' not in results:
        print("Данные сходимости не загружены")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    for problem_name, algorithms_data in results['convergence'].items():
        plt.figure(figsize=(10, 6))
        
        for algo_name, histories in algorithms_data.items():
            if not histories or len(histories) == 0:
                continue
            # Преобразуем в массив (запуски x итерации)
            # Учитываем разную длину историй – обрезаем до минимальной длины
            min_len = min(len(h) for h in histories if len(h) > 0)
            if min_len == 0:
                continue
            truncated = np.array([h[:min_len] for h in histories if len(h) >= min_len])
            
            median = np.median(truncated, axis=0)
            q25 = np.percentile(truncated, 25, axis=0)
            q75 = np.percentile(truncated, 75, axis=0)
            
            iterations = range(1, min_len+1)
            plt.fill_between(iterations, q25, q75, alpha=0.3, color=colors.get(algo_name, 'gray'))
            plt.plot(iterations, median, label=algo_name, color=colors.get(algo_name, 'gray'), linewidth=2)
        
        plt.title(f'Сходимость алгоритмов: {problem_titles.get(problem_name, problem_name)}', fontsize=14)
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('Значение целевой функции', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        filename = f"convergence_{problem_name}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"График сходимости сохранен: {filepath}")
        plt.close()

def plot_quality_speed_comparison(save_dir="plots"):
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Нормализуем фитнес и время
    df_normalized = df.copy()
    for problem in df['Problem'].unique():
        mask = df['Problem'] == problem
        # Нормализация фитнеса (меньше лучше)
        fitness_col = 'Best_Fitness_Median'  # используем медиану
        f_min = df.loc[mask, fitness_col].min()
        f_max = df.loc[mask, fitness_col].max()
        if f_max > f_min:
            df_normalized.loc[mask, 'Fitness_Norm'] = 1 - (df.loc[mask, fitness_col] - f_min) / (f_max - f_min)
        else:
            df_normalized.loc[mask, 'Fitness_Norm'] = 0.5
        
        time_col = 'Execution_Time_Mean'
        t_min = df.loc[mask, time_col].min()
        t_max = df.loc[mask, time_col].max()
        if t_max > t_min:
            df_normalized.loc[mask, 'Time_Norm'] = 1 - (df.loc[mask, time_col] - t_min) / (t_max - t_min)
        else:
            df_normalized.loc[mask, 'Time_Norm'] = 0.5
    
    problems = df['Problem'].unique()
    n_problems = len(problems)
    fig, axes = plt.subplots(1, n_problems, figsize=(6*n_problems, 6))
    if n_problems == 1:
        axes = [axes]
    
    fig.suptitle('Сравнение качества и скорости алгоритмов\n(нормированные значения: 1 - лучше, 0 - хуже)', 
                fontsize=14, fontweight='bold', y=1.05)
    
    for idx, problem in enumerate(problems):
        ax = axes[idx]
        problem_data = df_normalized[df_normalized['Problem'] == problem]
        problem_data_orig = df[df['Problem'] == problem]
        algorithms = problem_data['Algorithm'].values
        x = np.arange(len(algorithms))
        width = 0.35
        
        fitness_bars = ax.bar(x - width/2, problem_data['Fitness_Norm'].values, width, 
                              label='Качество (фитнес)', color='skyblue', edgecolor='navy', alpha=0.7)
        time_bars = ax.bar(x + width/2, problem_data['Time_Norm'].values, width,
                           label='Скорость (время)', color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        for i, (fb, tb) in enumerate(zip(fitness_bars, time_bars)):
            fitness_val = problem_data_orig['Best_Fitness_Median'].values[i]
            ax.text(fb.get_x() + fb.get_width()/2, fb.get_height() + 0.02,
                    f'{fitness_val:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
            time_val = problem_data_orig['Execution_Time_Mean'].values[i]
            ax.text(tb.get_x() + tb.get_width()/2, tb.get_height() + 0.02,
                    f'{time_val:.2f}с', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_title(problem_titles.get(problem, problem), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel('Нормированное значение')
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(loc='upper right')
        for i, algo in enumerate(algorithms):
            ax.get_xticklabels()[i].set_color(colors.get(algo, 'black'))
    
    plt.tight_layout()
    output_file = os.path.join(save_dir, "quality_speed_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сравнения качества и скорости сохранен: {output_file}")
    plt.close()

def create_detailed_ranking_table(save_dir="plots"):
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    ax.axis('tight')
    
    table_data = []
    headers = ['Задача', 'Алгоритм', 'Фитнес (медиана)', 'Ранг\nкачества', 
               'Время (с)', 'Ранг\nскорости', 'Ср. ранг', 'Лучший?']
    
    all_best = []
    
    for problem in df['Problem'].unique():
        problem_df = df[df['Problem'] == problem].copy()
        # Ранги: меньше фитнес – лучше
        problem_df.loc[:, 'Quality_Rank'] = problem_df['Best_Fitness_Median'].rank(method='min')
        problem_df.loc[:, 'Speed_Rank'] = problem_df['Execution_Time_Mean'].rank(method='min')
        problem_df.loc[:, 'Avg_Rank'] = (problem_df['Quality_Rank'] + problem_df['Speed_Rank']) / 2
        best_avg_rank = problem_df['Avg_Rank'].min()
        problem_df.loc[:, 'Is_Best'] = problem_df['Avg_Rank'] == best_avg_rank
        problem_df = problem_df.sort_values('Avg_Rank')
        
        for _, row in problem_df.iterrows():
            is_best = row['Is_Best']
            best_marker = '✓' if is_best else ''
            if is_best:
                all_best.append(row['Algorithm'])
            
            fitness_val = row['Best_Fitness_Median']
            fitness_str = f'{fitness_val:.4e}' if fitness_val < 1e5 else f'{fitness_val:.0e}'
            table_data.append([
                problem_titles.get(row['Problem'], row['Problem']),
                row['Algorithm'],
                fitness_str,
                f"{int(row['Quality_Rank'])}",
                f"{row['Execution_Time_Mean']:.3f}",
                f"{int(row['Speed_Rank'])}",
                f"{row['Avg_Rank']:.1f}",
                best_marker
            ])
        table_data.append(['---', '---', '---', '---', '---', '---', '---', '---'])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.1, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i, header in enumerate(headers):
        table[0, i].set_facecolor('#4472C4')
        table[0, i].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        row_data = table_data[i-1]
        if row_data[0] == '---':
            for j in range(len(headers)):
                table[i, j].set_facecolor('#E0E0E0')
            continue
        is_best = row_data[7] == '✓'
        for j in range(len(headers)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
            if is_best:
                cell.set_text_props(weight='bold')
                if j == 1:
                    algo = row_data[1]
                    if algo in colors:
                        rgba = plt.matplotlib.colors.to_rgba(colors[algo], alpha=0.3)
                        cell.set_facecolor(rgba)
    
    ax.set_title('Детальный рейтинг алгоритмов по качеству и скорости\n(меньший ранг = лучше)', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_file = os.path.join(save_dir, "quality_speed_ranking.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Таблица рейтингов сохранена: {output_file}")
    plt.close()
    
    from collections import Counter
    best_counter = Counter(all_best)
    print("\n" + "="*60)
    print("СТАТИСТИКА ЛУЧШИХ АЛГОРИТМОВ")
    print("="*60)
    for algo, count in best_counter.most_common():
        print(f"{algo}: лучший в {count} задаче(ах)")
    print("="*60)

def plot_radar_chart(save_dir="plots"):
    summary_file = os.path.join("experiment_results", "summary_results.csv")
    if not os.path.exists(summary_file):
        print(f"Файл {summary_file} не найден")
        return
    
    df = pd.read_csv(summary_file)
    os.makedirs(save_dir, exist_ok=True)
    
    algorithms = df['Algorithm'].unique()
    metrics = {}
    for algo in algorithms:
        algo_df = df[df['Algorithm'] == algo]
        avg_fitness = algo_df['Best_Fitness_Median'].mean()
        avg_time = algo_df['Execution_Time_Mean'].mean()
        avg_feasible = algo_df['Feasible_Rate_%'].mean()
        avg_acceptable = algo_df['Acceptable_Rate_%'].mean()
        metrics[algo] = {'fitness': avg_fitness, 'time': avg_time, 
                         'feasible': avg_feasible, 'acceptable': avg_acceptable}
    
    # Нормализация
    normalized = {}
    for metric in ['fitness', 'time', 'feasible', 'acceptable']:
        values = [metrics[algo][metric] for algo in algorithms]
        min_val, max_val = min(values), max(values)
        for algo in algorithms:
            if algo not in normalized:
                normalized[algo] = {}
            if metric in ['fitness', 'time']:
                if max_val > min_val:
                    norm_val = 1 - (metrics[algo][metric] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
            else:
                if max_val > min_val:
                    norm_val = (metrics[algo][metric] - min_val) / (max_val - min_val)
                else:
                    norm_val = 0.5
            normalized[algo][metric] = norm_val
    
    categories = ['Качество\n(фитнес)', 'Скорость\n(время)', 'Допустимые\n(%)', 'Приемлемые\n(%)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    for algo in algorithms:
        values = [normalized[algo]['fitness'], normalized[algo]['time'],
                  normalized[algo]['feasible'], normalized[algo]['acceptable']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors.get(algo, 'gray'))
        ax.fill(angles, values, alpha=0.1, color=colors.get(algo, 'gray'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('Сравнение алгоритмов по комплексным метрикам', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    output_file = os.path.join(save_dir, "radar_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Радарная диаграмма сохранена: {output_file}")
    plt.close()

def main():
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    results = load_results()
    if not results:
        print("Нет данных для визуализации")
        return
    
    print("\n1. Создание графиков сходимости (медиана + IQR)...")
    plot_convergence(results, plots_dir)
    
    print("\n2. Создание базовых сравнительных графиков...")
    plot_quality_speed_comparison(plots_dir)
    
    print("\n3. Создание детальной таблицы рейтингов...")
    create_detailed_ranking_table(plots_dir)
    
    print("\n4. Создание радарной диаграммы...")
    plot_radar_chart(plots_dir)
    
    print("\n" + "=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"Все графики сохранены в директории: {plots_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()