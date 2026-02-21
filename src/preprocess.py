import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import os

# Configs
RAW_DATA_PATH = "data/raw/sir_dataset.csv"
OUTPUT_PATH = "data/processed"
N = 1000  # Total Population (For Normalization)
T_MAX = 50
STEPS = 100 # We need exact 100 points for each simulation

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading Raw Data... (Thoda time lagega)")
df = pd.read_csv(RAW_DATA_PATH)

# Unique simulation IDs extraction
sim_ids = df['sim_id'].unique()
num_sims = len(sim_ids)

# Arrays to store processed data
# X: Input Features (S, I, R curves) -> Shape: (num_sims, steps, 3)
# Y: Targets (Beta, Gamma) -> Shape: (num_sims, 2)
X_data = []
Y_data = []

print(f"Processing {num_sims} simulations...")

for sim_id in tqdm(sim_ids):
    # 1. Filter data for this specific simulation
    sim_data = df[df['sim_id'] == sim_id]
    
    # Sort by time (Important for interpolation)
    sim_data = sim_data.sort_values('time')
    
    # 2. Extract Time and States
    t_raw = sim_data['time'].values
    S_raw = sim_data['S'].values
    I_raw = sim_data['I'].values
    R_raw = sim_data['R'].values
    
    # 3. Handle duplicates in time (If Gillespie had recorded 2 events at the same time)
    _, unique_indices = np.unique(t_raw, return_index=True)
    t_raw = t_raw[unique_indices]
    S_raw = S_raw[unique_indices]
    I_raw = I_raw[unique_indices]
    R_raw = R_raw[unique_indices]
    
    # 4. Interpolation (Resample to fixed time grid: 0, 0.5, 1.0, ...)
    # Create smooth functions for S, I, R
    f_S = interp1d(t_raw, S_raw, kind='linear', fill_value="extrapolate")
    f_I = interp1d(t_raw, I_raw, kind='linear', fill_value="extrapolate")
    f_R = interp1d(t_raw, R_raw, kind='linear', fill_value="extrapolate")
    
    # Generate fixed time points
    t_fixed = np.linspace(0, T_MAX, STEPS)
    
    # Get interpolated values & Normalize (Divide by N)
    S_fixed = f_S(t_fixed) / N
    I_fixed = f_I(t_fixed) / N
    R_fixed = f_R(t_fixed) / N
    
    # Stack into a single matrix (steps, 3)
    # Shape: (100, 3) -> Columns are S, I, R
    simulation_matrix = np.stack([S_fixed, I_fixed, R_fixed], axis=1)
    
    # 5. Store Input (X)
    X_data.append(simulation_matrix)
    
    # 6. Store Target (Y) -> Beta, Gamma (This is  constant for whole sim )
    # take the first row only
    beta = sim_data['beta'].iloc[0]
    gamma = sim_data['gamma'].iloc[0]
    Y_data.append([beta, gamma])

# Convert to NumPy Arrays
X_final = np.array(X_data, dtype=np.float32)
Y_final = np.array(Y_data, dtype=np.float32)

print(f"Saving Processed Data...")
print(f"X Shape: {X_final.shape} (Sims, TimeSteps, Features)")
print(f"Y Shape: {Y_final.shape} (Sims, Targets)")

# Save as .npy files (Fast loading for PyTorch)
np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_final)
np.save(os.path.join(OUTPUT_PATH, "Y_train.npy"), Y_final)

print("✅ Data Processing Complete! Ready for Training.")