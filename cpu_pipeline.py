import pandas as pd
import numpy as np
import time
import os
import glob
import pyarrow.parquet as pq

# Gradient Descent for Linear Regression on CPU
def train_linear_regression_cpu(X, y, learning_rate=0.01, epochs=200):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features, dtype=np.float32)
    bias = 0.0
    
    for epoch in range(epochs):
        # Forward pass
        predictions = np.dot(X, weights) + bias
        
        # Compute loss (MSE)
        errors = predictions - y
        
        # Compute gradients
        dw = (2 / n_samples) * np.dot(X.T, errors)
        db = (2 / n_samples) * np.sum(errors)
        
        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
    return weights, bias

def run_cpu_pipeline(data_dir, num_files=2):
    print("--- Starting CPU Pipeline ---")
    start_time = time.time()
    
    # 1. Loading data
    print(f"Loading first {num_files} Parquet datasets...")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    parquet_files = parquet_files[:num_files]
    
    dfs = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    load_time = time.time()
    print(f"Loaded {len(df)} rows in {load_time - start_time:.2f} seconds.")
    
    # 2. Preprocessing
    prep_start = time.time()
    
    # Needs pickup and dropoff datetime for duration
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'passenger_count', 'fare_amount'])
    
    # Calculate duration in minutes securely on CPU
    duration_secs = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds().values
    df['duration_min'] = duration_secs / 60.0
    
    # Filter valid trips
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]
    df = df[(df['duration_min'] > 0) & (df['duration_min'] < 200)]
    
    prep_time = time.time()
    print(f"Preprocessing CPU filtering & duration calculated in {prep_time - prep_start:.2f} seconds.")
    
    # Feature matrix X and target y
    X = df[['trip_distance', 'passenger_count', 'duration_min']].values.astype(np.float32)
    # Standardize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    y = df['fare_amount'].values.astype(np.float32)
    
    # 3. Training
    print("Starting Gradient Descent on CPU...")
    train_start = time.time()
    w, b = train_linear_regression_cpu(X, y, learning_rate=0.05, epochs=200)
    train_time = time.time()
    print(f"Training CPU finished in {train_time - train_start:.2f} seconds.")
    print(f"Model converged with Weights: {w}, Bias: {b:.4f}")
    
    total_time = time.time() - start_time
    print(f"Total CPU Pipeline Time: {total_time:.2f} seconds.")
    
    return {
        'total_time': total_time,
        'load_time': load_time - start_time,
        'prep_time': prep_time - prep_start,
        'train_time': train_time - train_start
    }

if __name__ == "__main__":
    run_cpu_pipeline(r"c:\Users\RUTIKA\Desktop\uber\Data", num_files=2)
