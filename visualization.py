"""
visualization.py
Модуль для визуализации сходимости алгоритмов.
Строит графики с линейной шкалой, средним значением по 20 запускам.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300
})

# Названия задач (ключи должны совпадать с именами в convergence.json)
problem_titles = {
    'dc_motor_pid': 'ПИД-регулятор двигателя',
    'inverted_pendulum': 'Балансировка маятника',
    'liquid_level': 'Уровень жидкости'
}

# Цвета для алгоритмов
colors = {
    'PSO': '#FF6B6B',
    'GWO': '#4ECDC4',
    'WOA': '#45B7D1',
    'HHO': '#96CEB4',
    'SMA': '#FFEAA7'
}


def load_convergence_data(results_dir="results"):
    """Загружает данные сходимости из convergence.json"""
    convergence_file = os.path.join(results_dir, "convergence.json")
    if not os.path.exists(convergence_file):
        print(f"Ошибка: файл {convergence_file} не найден")
        return None
    with open(convergence_file, 'r') as f:
        convergence_data = json.load(f)
    print(f"Загружены данные сходимости из: {convergence_file}")
    return convergence_data

def plot_convergence(convergence_data, save_dir="plots"):
    """
    Строит графики сходимости для каждой задачи.
    - Линейная шкала по оси Y.
    - Отображается среднее значение целевой функции по 20 запускам.
    - Для каждого алгоритма – одна линия (без доверительных интервалов).
    - Ось Y форматируется в научном виде: число ×10^степень.
    """
    os.makedirs(save_dir, exist_ok=True)

    for problem_name, algorithms_data in convergence_data.items():
        plt.figure()

        for algo_name, histories in algorithms_data.items():
            if not histories:
                continue

            # Определяем минимальную длину истории (чтобы все линии были одинаковой длины)
            min_len = min(len(h) for h in histories if len(h) > 0)
            if min_len == 0:
                continue

            # Обрезаем каждую историю до min_len и преобразуем в массив
            truncated = np.array([h[:min_len] for h in histories if len(h) >= min_len])

            # Вычисляем среднее по запускам
            mean_values = np.mean(truncated, axis=0)

            # Строим линию
            iterations = range(1, min_len + 1)
            plt.plot(iterations, mean_values,
                     label=algo_name,
                     color=colors.get(algo_name, 'gray'),
                     linewidth=2)

        # Настройки графика
        plt.title(f'Сходимость алгоритмов: {problem_titles.get(problem_name, problem_name)}')
        plt.xlabel('Итерация')
        plt.ylabel('Среднее значение целевой функции по 20 запускам')
        
        # Форматирование оси Y в научном виде с множителем ×10^степень
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Сохраняем график
        filepath = os.path.join(save_dir, f"convergence_{problem_name}.png")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Сохранён график: {filepath}")



def main():
    parser = argparse.ArgumentParser(description='Построение графиков сходимости')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Папка с результатами (должна содержать convergence.json)')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Папка для сохранения графиков')
    args = parser.parse_args()

    convergence_data = load_convergence_data(args.results_dir)
    if convergence_data is None:
        return

    plot_convergence(convergence_data, args.plots_dir)
    print("Все графики успешно построены.")


if __name__ == "__main__":
    main()