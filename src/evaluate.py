import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Fix OpenMP error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models import SIR_LSTM
from src.simulation import gillespie_sir

# Config
MODEL_PATH = "saved_models/lstm_sir_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 1000 # Population

# --- Helper: ODE Solver for Plotting ---
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def solve_ode(beta, gamma, t_max=50, steps=100):
    t = np.linspace(0, t_max, steps)
    y0 = 990, 10, 0 # S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    return t, ret.T # returns S, I, R

# --- Main Evaluation Logic ---
def evaluate():
    # 1. Load the Trained Model
    print("Loading Model...")
    model = SIR_LSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    
    # 2. Generate a FRESH Test Case (Model ne ye kabhi nahi dekha)
    true_beta = np.random.uniform(0.3, 0.8)
    true_gamma = np.random.uniform(0.1, 0.3)
    print(f"TRUE Parameters -> Beta: {true_beta:.4f}, Gamma: {true_gamma:.4f}")
    
    # Run Stochastic Simulation
    t_sim, S_sim, I_sim, R_sim = gillespie_sir(990, 10, 0, true_beta, true_gamma, 50)
    
    # 3. Preprocess Input (Interpolate & Normalize) like we did in training
    # Hum seedha plotting ke liye simplify kar rahe hain:
    # Model ko fixed 100 steps chahiye
    from scipy.interpolate import interp1d
    t_fixed = np.linspace(0, 50, 100)
    
    f_S = interp1d(t_sim, S_sim, kind='linear', fill_value="extrapolate")
    f_I = interp1d(t_sim, I_sim, kind='linear', fill_value="extrapolate")
    f_R = interp1d(t_sim, R_sim, kind='linear', fill_value="extrapolate")
    
    # Input Shape: (1, 100, 3) -> Batch size 1
    input_tensor = np.stack([f_S(t_fixed)/N, f_I(t_fixed)/N, f_R(t_fixed)/N], axis=1)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 4. Predict with Model
    with torch.no_grad():
        prediction = model(input_tensor) # Output: [beta, gamma]
        
    pred_beta = prediction[0][0].item()
    pred_gamma = prediction[0][1].item()
    
    print(f"PREDICTED -> Beta: {pred_beta:.4f}, Gamma: {pred_gamma:.4f}")
    
    # Calculate Error
    err_beta = abs(true_beta - pred_beta) / true_beta * 100
    err_gamma = abs(true_gamma - pred_gamma) / true_gamma * 100
    print(f"Error -> Beta: {err_beta:.2f}%, Gamma: {err_gamma:.2f}%")
    
    # 5. Visual Comparison
    # True Curve (ODEs with True parameters)
    t_true, (S_true, I_true, R_true) = solve_ode(true_beta, true_gamma)
    
    # Predicted Curve (ODEs with Predicted parameters)
    t_pred, (S_pred, I_pred, R_pred) = solve_ode(pred_beta, pred_gamma)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot Noisy Data (Input)
    plt.scatter(t_sim, I_sim, s=10, alpha=0.3, color='gray', label='Noisy Input (Stochastic)')
    
    # Plot True Smooth Curve
    plt.plot(t_true, I_true, 'g--', linewidth=2, label=f'True ODE (Beta={true_beta:.2f})')
    
    # Plot Predicted Curve
    plt.plot(t_pred, I_pred, 'r-', linewidth=2, label=f'Model Prediction (Beta={pred_beta:.2f})')
    
    plt.title(f"Model Evaluation: Recovering Parameters from Noise\nError: {err_beta:.1f}%")
    plt.xlabel("Time")
    plt.ylabel("Infected Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "plots/final_result.png"
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()