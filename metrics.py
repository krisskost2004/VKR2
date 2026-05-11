# metrics.py
import numpy as np
from motor_model import simulate_motor_pid, compute_step_metrics
from pendulum_model import simulate_pendulum_lqr, check_stability
from tank_model import simulate_tanks_pi, compute_tank_metrics

def evaluate_motor_solution(params, t_end=5.0, dt=0.01):
    """Возвращает (feasible, acceptable, metrics_dict) для задачи двигателя."""
    Kp, Ki, Kd = params
    # Допустимость: параметры в границах и моделирование успешно
    if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
        return False, False, {}
    t, y = simulate_motor_pid(params, t_end, dt)
    if y is None:
        return False, False, {}
    feasible = True

    # Вычисление метрик
    overshoot, rise_time, settling_time, steady_state_error = compute_step_metrics(t, y)
    metrics = {
        'overshoot': overshoot,
        'rise_time': rise_time,
        'settling_time': settling_time,
        'steady_state_error': steady_state_error
    }
    # Приемлемость: перерегулирование <= 15%, время установления <= 1.5 с, остаточная ошибка < 0.05
    acceptable = (overshoot <= 15.0) and (settling_time <= 1.5) and (steady_state_error < 0.05)
    return feasible, acceptable, metrics

def evaluate_pendulum_solution(K, t_end=5.0, dt=0.01):
    """Возвращает (feasible, acceptable, metrics_dict) для задачи маятника."""
    K = np.array(K).flatten()
    if np.any(np.abs(K) > 50):
        return False, False, {}
    if not check_stability(K):
        return False, False, {}
    t, X = simulate_pendulum_lqr(K, t_end, dt)
    if X is None:
        return False, False, {}
    feasible = True

    max_angle = np.max(np.abs(X[:, 2]))
    final_angle = X[-1, 2]
    metrics = {'max_angle': max_angle, 'final_angle': final_angle}
    # Приемлемость: угол не превышает 0.2 рад и финальный угол близок к нулю
    acceptable = (max_angle < 0.2) and (abs(final_angle) < 0.05)
    return feasible, acceptable, metrics

def evaluate_tank_solution(params, t_end=10.0, dt=0.1):
    """Возвращает (feasible, acceptable, metrics_dict) для задачи уровней."""
    Kp1, Ki1, Kp2, Ki2 = params
    if not (0 <= Kp1 <= 5 and 0 <= Ki1 <= 2 and 0 <= Kp2 <= 5 and 0 <= Ki2 <= 2):
        return False, False, {}
    t, h1, h2 = simulate_tanks_pi(params, t_end, dt)
    if h1 is None:
        return False, False, {}
    feasible = True

    metrics = compute_tank_metrics(t, h1, h2)
    # Приемлемость: остаточные ошибки < 0.05
    acceptable = (metrics['final_error_1'] < 0.05) and (metrics['final_error_2'] < 0.05)
    return feasible, acceptable, metrics