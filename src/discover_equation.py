import torch
import numpy as np
import pysindy as ps
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import os

# --- Fix OpenMP Error ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Script started... (Imports done)")

from src.models import SIR_LSTM
from src.simulation import gillespie_sir

# --- Config ---
MODEL_PATH = "saved_models/lstm_sir_v1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 1000 

def discover_dynamics():
    # 1. Load Model
    print("Loading Neural Network...")
    model = SIR_LSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Ground Truth
    true_beta = 0.50
    true_gamma = 0.20
    print(f"Ground Truth -> Beta: {true_beta}, Gamma: {true_gamma}")
    
    # Generate Noisy Data
    t_sim, S_sim, I_sim, R_sim = gillespie_sir(990, 10, 0, true_beta, true_gamma, 50)
    
    # 3. Predict with LSTM
    t_fixed = np.linspace(0, 50, 100)
    f_S = interp1d(t_sim, S_sim, kind='linear', fill_value="extrapolate")
    f_I = interp1d(t_sim, I_sim, kind='linear', fill_value="extrapolate")
    f_R = interp1d(t_sim, R_sim, kind='linear', fill_value="extrapolate")
    
    input_tensor = np.stack([f_S(t_fixed)/N, f_I(t_fixed)/N, f_R(t_fixed)/N], axis=1)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_params = model(input_tensor)
    
    pred_beta = pred_params[0][0].item()
    pred_gamma = pred_params[0][1].item()
    print(f"LSTM Prediction -> Beta: {pred_beta:.4f}, Gamma: {pred_gamma:.4f}")

    # 4. Generate Clean Trajectory (Normalized)
    def deriv_normalized(y, t, beta, gamma):
        s, i, r = y
        dsdt = -beta * s * i
        didt = beta * s * i - gamma * i
        drdt = gamma * i
        return dsdt, didt, drdt

    y0 = 0.99, 0.01, 0.0
    X_clean = odeint(deriv_normalized, y0, t_fixed, args=(pred_beta, pred_gamma))

    # --- Use only S (col 0) and I (col 1) ---
    X_sindy = X_clean[:, :2] 

    # 5. Run SINDy
    print("\n--- Running Symbolic Regression (S & I Only) ---")
    
    # Library: Degree 2, No Bias
    library = ps.PolynomialLibrary(degree=2, include_bias=False)
    
    # Optimizer: Threshold 0.01
    optimizer = ps.STLSQ(threshold=0.01, normalize_columns=True)
    
    # NO FEATURE NAMES HERE
    sindy_model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer
    )
    
    sindy_model.fit(X_sindy, t=t_fixed)
    
    # 6. Output
    print("\nDiscovered Equations:")
    print("Legend: x0 = S (Susceptible), x1 = I (Infected)")
    sindy_model.print()
    
    print("\n------------------------------------------------")
    print("Interpretation:")
    print(f"Expected (x1)' =  ({pred_beta:.3f}) x0 x1  - ({pred_gamma:.3f}) x1")
    print("------------------------------------------------")
    
    score = sindy_model.score(X_sindy, t=t_fixed)
    print(f"Model Score (R^2): {score:.4f}")

if __name__ == "__main__":
    discover_dynamics()