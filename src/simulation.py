import numpy as np
from scipy.integrate import odeint

# --- 1. Deterministic Solver (The "Truth") ---
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def deterministic_sir(S0, I0, R0, beta, gamma, t_max, steps=100):
    """Generates the smooth 'Mean' curve using ODEs."""
    N = S0 + I0 + R0
    t = np.linspace(0, t_max, steps)
    y0 = S0, I0, R0
    
    # Solve ODE
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R

# --- 2. Stochastic Solver (The "Noise") ---
def gillespie_sir(S0, I0, R0, beta, gamma, t_max):
    """Generates the noisy curve using Gillespie algorithm."""
    t = 0
    S, I, R = S0, I0, R0
    N = S0 + I0 + R0
    
    t_list = [t]
    S_list = [S]; I_list = [I]; R_list = [R]
    
    while t < t_max and I > 0:
        rate_infection = beta * S * I / N
        rate_recovery = gamma * I
        total_rate = rate_infection + rate_recovery
        
        if total_rate == 0: break
        
        dt = -np.log(np.random.random()) / total_rate
        t += dt
        
        if np.random.random() < (rate_infection / total_rate):
            S -= 1; I += 1
        else:
            I -= 1; R += 1
            
        t_list.append(t)
        S_list.append(S); I_list.append(I); R_list.append(R)
        
    return np.array(t_list), np.array(S_list), np.array(I_list), np.array(R_list)