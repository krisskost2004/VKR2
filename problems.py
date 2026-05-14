"""
Модуль с тестовыми задачами из теории управления.
Каждая задача формализована как функция для оптимизации.
"""

import numpy as np
from scipy.integrate import odeint
import control as ctrl
import warnings
from simulation import simulate_dc_motor_pid, compute_step_metrics

warnings.filterwarnings('ignore')

# Глобальная переменная для фиксации seed шума в задаче уровня жидкости
_rng_seed = None

# ============================================================================
# 1. ОПТИМИЗАЦИЯ ПИД-РЕГУЛЯТОРА ДЛЯ ДВИГАТЕЛЯ ПОСТОЯННОГО ТОКА
# ============================================================================

def dc_motor_pid_objective(params):
    """
    Целевая функция для оптимизации ПИД-регулятора двигателя постоянного тока.
    Минимизируется ITAE + штраф за перерегулирование + штраф за ошибку на хвосте.
    """
    Kp, Ki, Kd = params
    
    # Проверка границ
    if not (0.1 <= Kp <= 50 and 0.01 <= Ki <= 30 and 0 <= Kd <= 10):
        return 1e6
    
    try:
        t, y = simulate_dc_motor_pid(params, t_end=5, n_points=500)
        e = 1 - y
        dt = t[1] - t[0]
        
        # ITAE
        J = np.sum(t * np.abs(e)) * dt
        
        # Штраф за перерегулирование
        overshoot = max(0, (np.max(y) - 1) / 1 * 100)
        if overshoot > 5:
            J += 100 * (overshoot - 5)   # штраф только за превышение 5%
        
        
        # Штраф за среднюю ошибку на хвосте (t > 3)
        tail_mask = t > 3
        if np.any(tail_mask):
            tail_error = np.mean(np.abs(e[tail_mask]))
            J += 50 * tail_error
        
        return float(J)
        
    except Exception:
        return 1e6


# ============================================================================
# 2. БАЛАНСИРОВКА ПЕРЕВЕРНУТОГО МАЯТНИКА
# ============================================================================

def inverted_pendulum_objective(params):
    """
    Целевая функция для балансировки перевернутого маятника.
    """
    K1, K2, K3, K4 = params
    
    if any(np.abs(p) > 50 for p in params):
        return 1e6
    
    try:
        M = 1.0; m = 0.1; b = 0.1; l = 0.5; g = 9.81
        
        A = np.array([
            [0, 1, 0, 0],
            [0, -b/M, -m*g/M, 0],
            [0, 0, 0, 1],
            [0, -b/(M*l), (M+m)*g/(M*l), 0]
        ])
        B = np.array([[0], [1/M], [0], [1/(M*l)]])
        K = np.array([[K1, K2, K3, K4]])
        A_closed = A - B @ K
        
        if np.any(np.real(np.linalg.eigvals(A_closed)) >= 0):
            return 1e6
        
        def system_dynamics(x, t):
            return A_closed.dot(x)
        
        x0 = np.array([0, 0, 0.1, 0])
        t = np.linspace(0, 5, 500)
        x = odeint(system_dynamics, x0, t)
        
        Q = np.diag([10, 0.1, 100, 0.1])
        cost = np.mean([x[i].T @ Q @ x[i] for i in range(len(t))])
        return float(cost)
        
    except Exception:
        return 1e6


# ============================================================================
# 3. УПРАВЛЕНИЕ УРОВНЕМ ЖИДКОСТИ В РЕЗЕРВУАРАХ
# ============================================================================

def liquid_level_control_objective(params):
    """
    Целевая функция для управления уровнем жидкости в двух связанных резервуарах.
    """
    Kp1, Ki1, Kp2, Ki2 = params
    
    if not (0 <= Kp1 <= 5 and 0 <= Ki1 <= 2 and 0 <= Kp2 <= 5 and 0 <= Ki2 <= 2):
        return 1e6
    
    try:
        A1, A2 = 2.0, 1.5
        R1_base, R2_base = 0.5, 0.7
        dt = 0.1
        steps = 200
        
        h1, h2 = 0.5, 0.3
        I1, I2 = 0.0, 0.0
        u1_prev, u2_prev = 0.0, 0.0
        
        total_error = 0.0
        control_smoothness_penalty = 0.0
        oscillation_penalty = 0.0
        prev_h1, prev_h2 = h1, h2
        
        rng = np.random.default_rng(_rng_seed) if _rng_seed is not None else np.random.default_rng()
        
        for k in range(steps):
            t = k * dt
            if t < 10:
                h1_ref, h2_ref = 1.0, 0.8
            else:
                h1_ref, h2_ref = 0.9, 0.7
            
            noise1 = rng.normal(0, 0.01)
            noise2 = rng.normal(0, 0.01)
            h1_meas = h1 + noise1
            h2_meas = h2 + noise2
            
            e1 = h1_ref - h1_meas
            e2 = h2_ref - h2_meas
            I1 += e1 * dt
            I2 += e2 * dt
            
            u1 = np.clip(Kp1 * e1 + Ki1 * I1, 0, 2)
            u2 = np.clip(Kp2 * e2 + Ki2 * I2, 0, 2)
            
            control_smoothness_penalty += (u1 - u1_prev)**2 + (u2 - u2_prev)**2
            u1_prev, u2_prev = u1, u2
            
            R1 = R1_base * (1 + 0.5 * h1)
            R2 = R2_base * (1 + 0.5 * h2)
            q12 = max(0, (h1 - h2) / R1) * np.sqrt(abs(h1 - h2) + 1e-6)
            q2out = (h2 / R2) * np.sqrt(h2 + 1e-6)
            
            h1 += (u1 - q12) / A1 * dt
            h2 += (q12 - q2out + u2) / A2 * dt
            h1 = max(0, h1)
            h2 = max(0, h2)
            
            total_error += (abs(h1_ref - h1) + abs(h2_ref - h2)) * dt
            oscillation_penalty += (h1 - prev_h1)**2 + (h2 - prev_h2)**2
            prev_h1, prev_h2 = h1, h2
        
        final_error = abs(h1_ref - h1) + abs(h2_ref - h2)
        
        J = (total_error 
             + 0.1 * control_smoothness_penalty
             + 0.01 * oscillation_penalty
             + 10 * final_error)
        
        return float(J)
        
    except Exception as e:
        return 1e6


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_problem_info(problem_name):
    """
    Возвращает информацию о задаче: размерность, границы параметров.
    """
    problems = {
        'dc_motor_pid': {
            'dim': 3,
            'bounds': (np.array([0.1, 0.01, 0]), np.array([50, 30, 10])),
            'description': 'Оптимизация ПИД-регулятора для двигателя постоянного тока',
            'objective_func': dc_motor_pid_objective
        },
        'inverted_pendulum': {
            'dim': 4,
            'bounds': (np.array([-50, -50, -50, -50]), np.array([50, 50, 50, 50])),
            'description': 'Балансировка перевернутого маятника',
            'objective_func': inverted_pendulum_objective
        },
        'liquid_level': {
            'dim': 4,
            'bounds': (np.array([0, 0, 0, 0]), np.array([5, 2, 5, 2])),
            'description': 'Управление уровнем жидкости в резервуарах',
            'objective_func': liquid_level_control_objective
        }
    }
    
    return problems.get(problem_name)