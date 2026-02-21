import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from simulation import gillespie_sir, deterministic_sir

# Configuration
NUM_SIMULATIONS = 5000   # how many epidemics need to be generate
OUTPUT_DIR = "data/raw"  # where to save them

# Make folder if doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_list = []

print(f"Generating {NUM_SIMULATIONS} synthetic epidemics...")

for sim_id in tqdm(range(NUM_SIMULATIONS)):
    # 1. Random Parameters choose
    # Beta: 0.2 se 1.0 ke beech (Infection rate)
    # Gamma: 0.1 se 0.5 ke beech (Recovery rate)
    beta = np.random.uniform(0.2, 1.0)
    gamma = np.random.uniform(0.1, 0.5)
    
    # R0 (Reproduction Number) check - For Epidemic doesnot fail 
    if beta / gamma < 1.1: 
        continue # Skip if disease does not spread 
        
    # 2. Run Simulations
    # Stochastic (Noisy Input)
    t_stoch, S_stoch, I_stoch, R_stoch = gillespie_sir(990, 10, 0, beta, gamma, t_max=50)
    
    # Deterministic (Smooth Target) - for checking "True" values later
    # Note: We are saving Stochastic for training
    # But parameters (beta, gamma) needed to be saved
    #     
    # 3. Save Data Points
    # Stochastic data's time is uneven so, each step needed to be saved
    for i in range(len(t_stoch)):
        data_list.append({
            "sim_id": sim_id,
            "time": t_stoch[i],
            "S": S_stoch[i],
            "I": I_stoch[i],
            "R": R_stoch[i],
            "beta": beta,      # Target for Inverse Problem
            "gamma": gamma     # Target for Inverse Problem
        })

# 4. Save to CSV
print("Saving to CSV... (This might take a minute)")
df = pd.DataFrame(data_list)
csv_path = os.path.join(OUTPUT_DIR, "sir_dataset.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Done! Dataset saved at: {csv_path}")
print(f"Total rows: {len(df)}")