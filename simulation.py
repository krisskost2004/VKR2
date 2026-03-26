"""
Модуль с единой функцией моделирования двигателя постоянного тока
и расчета метрик переходного процесса.
Используется в problems.py и plot_step_responses.py для согласованности.
"""

import numpy as np
import control as ctrl

def simulate_dc_motor_pid(params, t_end=5, n_points=500):
    """
    Симуляция двигателя постоянного тока с ПИД-регулятором.
    Параметры двигателя соответствуют problems.py.
    
    Args:
        params: [Kp, Ki, Kd]
        t_end: конечное время моделирования (с)
        n_points: количество точек временной сетки
    
    Returns:
        t: массив времени
        y: отклик (скорость, нормированная)
    """
    Kp, Ki, Kd = params
    # Параметры двигателя
    R = 1.0      # Сопротивление (Ом)
    L = 0.5      # Индуктивность (Гн)
    Kb = 0.01    # Коэффициент противо-ЭДС (В/рад/с)
    Kt = 0.01    # Коэффициент момента (Нм/А)
    J = 0.01     # Момент инерции (кг·м²)
    B = 0.1      # Коэффициент вязкого трения (Нм·с)

    # Передаточная функция двигателя
    num_motor = [Kt]
    den_motor = [J*L, J*R + B*L, B*R + Kt*Kb]
    G = ctrl.TransferFunction(num_motor, den_motor)

    # ПИД-регулятор
    num_pid = [Kd, Kp, Ki]
    den_pid = [1, 1e-10]  # малая константа для избежания деления на ноль
    C = ctrl.TransferFunction(num_pid, den_pid)

    # Замкнутая система с единичной обратной связью
    T = ctrl.feedback(C * G, 1)

    # Временная сетка
    t = np.linspace(0, t_end, n_points)
    t, y = ctrl.step_response(T, T=t)
    return t, y


def compute_step_metrics(t, y):
    """
    Вычисляет метрики переходного процесса по отклику y(t).
    
    Returns:
        dict: overshoot (%), rise_time (с), settling_time (с), steady_state_error (абс.)
    """
    if len(y) == 0:
        return {'overshoot': np.inf, 'rise_time': np.inf, 
                'settling_time': np.inf, 'steady_state_error': np.inf}

    # Установившееся значение – среднее последних 10% точек
    n = len(y)
    y_ss = np.mean(y[int(0.9*n):])
    if y_ss == 0:
        y_ss = 1e-8

    # Перерегулирование
    y_max = np.max(y)
    overshoot = max(0, (y_max - y_ss) / y_ss * 100)

    # Время нарастания (10% -> 90%)
    y_10 = 0.1 * y_ss
    y_90 = 0.9 * y_ss
    idx_10 = np.where(y >= y_10)[0]
    idx_90 = np.where(y >= y_90)[0]
    rise_time = t[idx_90[0]] - t[idx_10[0]] if len(idx_10) and len(idx_90) else np.inf

    # Время установления (2% коридор)
    settled = np.abs(y - y_ss) <= 0.02 * y_ss
    if np.all(settled):
        settling_time = 0.0
    else:
        unsettled = np.where(~settled)[0]
        settling_time = t[unsettled[-1]]

    # Установившаяся ошибка
    steady_state_error = abs(1 - y_ss)

    return {
        'overshoot': overshoot,
        'rise_time': rise_time,
        'settling_time': settling_time,
        'steady_state_error': steady_state_error
    }
