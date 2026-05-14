import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
from simulation import simulate_dc_motor_pid
import scipy.integrate

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.figsize': (12, 8), 'figure.dpi': 100,
    'savefig.dpi': 300
})

colors = {'PSO': '#FF6B6B', 'GWO': '#4ECDC4', 'WOA': '#45B7D1', 'HHO': '#96CEB4', 'SMA': '#FFEAA7'}
linestyles = {'PSO': '-', 'GWO': '--', 'WOA': '-.', 'HHO': ':', 'SMA': '-'}

def load_best_solutions(results_dir="results"):
    results_file = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"❌ Файл {results_file} не найден")
        return None
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best = {}
    for problem_name, problem_data in data.items():
        best[problem_name] = {}
        for algo_name, algo_data in problem_data.items():
            if 'error' not in algo_data and 'all_runs' in algo_data:
                runs = algo_data['all_runs']
                valid = [r for r in runs if 'solution' in r and r.get('feasible', False)]
                if valid:
                    best_run = min(valid, key=lambda x: x.get('best_fitness', float('inf')))
                    best[problem_name][algo_name] = {
                        'params': best_run['solution'],
                        'fitness': best_run['best_fitness']
                    }
                    print(f"  {problem_name}/{algo_name}: fitness={best_run['best_fitness']:.4e}")
    return best

def plot_dc_motor(best_solutions, save_dir):
    data = best_solutions.get('dc_motor_pid', {})
    if not data:
        print("Нет данных для dc_motor_pid")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    t = np.linspace(0, 5, 1000)
    for algo, d in data.items():
        t_out, y_out = simulate_dc_motor_pid(d['params'], 5, 1000)
        label = f"{algo} (fit={d['fitness']:.2e})"
        ax1.plot(t_out, y_out, label=label, color=colors.get(algo, 'gray'),
                 linestyle=linestyles.get(algo, '-'), linewidth=2)
        ax2.plot(t_out, y_out, color=colors.get(algo, 'gray'),
                 linestyle=linestyles.get(algo, '-'), linewidth=2)
    for ax in (ax1, ax2):
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax1.set_title('Полный переходный процесс (0-5 с)')
    ax1.set_xlabel('Время, с'); ax1.set_ylabel('Угловая скорость')
    ax1.set_xlim(0,5); ax1.set_ylim(0,1.2)
    ax2.set_title('Начальный участок (0-1 с)')
    ax2.set_xlabel('Время, с'); ax2.set_ylabel('Угловая скорость')
    ax2.set_xlim(0,1); ax2.set_ylim(0.9,1.1)
    ax1.legend(loc='lower right', fontsize=9)
    fig.suptitle('Переходные процессы ПИД-регулятора двигателя', fontsize=14)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "step_response_dc_motor.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График двигателя сохранён: {filepath}")

def plot_inverted_pendulum(best_solutions, save_dir):
    data = best_solutions.get('inverted_pendulum', {})
    if not data:
        print("Нет данных для inverted_pendulum")
        return
    plt.figure(figsize=(10,6))
    t = np.linspace(0,5,500)
    for algo, d in data.items():
        K = np.array(d['params'])
        M, m, b, l, g = 1.0, 0.1, 0.1, 0.5, 9.81
        A = np.array([[0,1,0,0],[0,-b/M,-m*g/M,0],[0,0,0,1],[0,-b/(M*l),(M+m)*g/(M*l),0]])
        B = np.array([[0],[1/M],[0],[1/(M*l)]])
        A_closed = A - B @ K.reshape(1,-1)
        def dyn(x, t): return A_closed.dot(x)
        x0 = [0,0,0.1,0]
        x = scipy.integrate.odeint(dyn, x0, t)
        plt.plot(t, x[:,2], label=f"{algo} (fit={d['fitness']:.2e})",
                 color=colors.get(algo, 'gray'), linestyle=linestyles.get(algo, '-'), linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Время, с'); plt.ylabel('Угол отклонения, рад')
    plt.title('Переходные процессы балансировки маятника')
    plt.legend(loc='best'); plt.grid(True, alpha=0.3); plt.xlim(0,5)
    filepath = os.path.join(save_dir, "step_response_inverted_pendulum.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График маятника сохранён: {filepath}")

def plot_liquid_level(best_solutions, save_dir):
    data = best_solutions.get('liquid_level', {})
    if not data:
        print("Нет данных для liquid_level")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12,10), sharex=True)
    fig.suptitle('Переходные процессы управления уровнем жидкости', fontsize=14)
    t_span = np.linspace(0,20,200)
    for algo, d in data.items():
        Kp1, Ki1, Kp2, Ki2 = d['params']
        A1, A2 = 2.0, 1.5; R1_base, R2_base = 0.5, 0.7; dt = 0.1; steps = 200
        h1, h2 = 0.5, 0.3; I1, I2 = 0.0, 0.0
        h1_traj, h2_traj = [h1], [h2]; times = [0.0]
        for k in range(steps):
            t = k*dt
            h1_ref, h2_ref = (1.0,0.8) if t<10 else (0.9,0.7)
            e1, e2 = h1_ref - h1, h2_ref - h2
            I1 += e1*dt; I2 += e2*dt
            u1 = np.clip(Kp1*e1 + Ki1*I1, 0, 2)
            u2 = np.clip(Kp2*e2 + Ki2*I2, 0, 2)
            R1 = R1_base*(1+0.5*h1); R2 = R2_base*(1+0.5*h2)
            q12 = max(0, (h1-h2)/R1)*np.sqrt(abs(h1-h2)+1e-6)
            q2out = (h2/R2)*np.sqrt(h2+1e-6)
            h1 += (u1 - q12)/A1*dt; h2 += (q12 - q2out + u2)/A2*dt
            h1, h2 = max(0,h1), max(0,h2)
            h1_traj.append(h1); h2_traj.append(h2); times.append(t+dt)
        color = colors.get(algo, 'gray'); ls = linestyles.get(algo, '-')
        axes[0].plot(times, h1_traj, label=f"{algo} (fit={d['fitness']:.2e})",
                     color=color, linestyle=ls, linewidth=1.5)
        axes[1].plot(times, h2_traj, color=color, linestyle=ls, linewidth=1.5)
    times_arr = np.array(times)
    ref1 = [1.0 if t<10 else 0.9 for t in times_arr]
    ref2 = [0.8 if t<10 else 0.7 for t in times_arr]
    axes[0].plot(times_arr, ref1, 'k--', linewidth=1, alpha=0.7, label='Уставка h1')
    axes[1].plot(times_arr, ref2, 'k--', linewidth=1, alpha=0.7, label='Уставка h2')
    axes[0].set_ylabel('Уровень h1, м'); axes[1].set_ylabel('Уровень h2, м')
    axes[1].set_xlabel('Время, с')
    axes[0].legend(loc='best'); axes[0].grid(True, alpha=0.3); axes[1].grid(True, alpha=0.3)
    axes[0].set_ylim(0,1.2); axes[1].set_ylim(0,1.0)
    filepath = os.path.join(save_dir, "step_response_liquid_level.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ График уровня жидкости сохранён: {filepath}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--plots_dir', type=str, default='plots')
    args = parser.parse_args()
    print("="*80)
    print("🔄 ПОСТРОЕНИЕ ПЕРЕХОДНЫХ ПРОЦЕССОВ ДЛЯ ВСЕХ ЗАДАЧ")
    print("="*80)
    print(f"Чтение результатов из: {args.results_dir}")
    best_solutions = load_best_solutions(args.results_dir)
    if not best_solutions:
        return
    print("\n📈 Построение графиков...")
    plot_dc_motor(best_solutions, args.plots_dir)
    plot_inverted_pendulum(best_solutions, args.plots_dir)
    plot_liquid_level(best_solutions, args.plots_dir)
    print("\n✅ ВСЕ ГРАФИКИ ПОСТРОЕНЫ")

if __name__ == "__main__":
    main()