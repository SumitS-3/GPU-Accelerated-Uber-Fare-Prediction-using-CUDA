import os
import sys
import pandas as pd
import numpy as np
import time
import glob
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────────────────────
# Prepend pip-installed NVIDIA DLL directories so ctypes can find them
# ─────────────────────────────────────────────────────────────────────────────
try:
    site_pkgs = next(p for p in sys.path if 'site-packages' in p)
    for _sub in ['cuda_nvrtc', 'cuda_runtime', 'cuda_nvcc', 'cublas']:
        _bin = os.path.join(site_pkgs, 'nvidia', _sub, 'bin')
        if os.path.isdir(_bin) and _bin not in os.environ['PATH']:
            os.environ['PATH'] = _bin + os.pathsep + os.environ['PATH']
    _nvvm = os.path.join(site_pkgs, 'nvidia', 'cuda_nvcc', 'nvvm', 'bin')
    if os.path.isdir(_nvvm) and _nvvm not in os.environ['PATH']:
        os.environ['PATH'] = _nvvm + os.pathsep + os.environ['PATH']
except StopIteration:
    pass

import cupy as cp

# ─────────────────────────────────────────────────────────────────────────────
# Custom CUDA kernels written as raw CUDA C – compiled at runtime by CuPy
# ─────────────────────────────────────────────────────────────────────────────

# Kernel 1: element-wise dot product (X[i] @ w + b → pred[i]) + error
_forward_kernel_code = r"""
extern "C" __global__
void forward_kernel(const float* X, const float* w, const float b,
                    float* pred, const int n_samples, const int n_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples) {
        float val = b;
        for (int j = 0; j < n_features; ++j) {
            val += X[i * n_features + j] * w[j];
        }
        pred[i] = val;
    }
}
"""

# Kernel 2: element-wise error: error[i] = pred[i] - y[i]
_error_kernel_code = r"""
extern "C" __global__
void error_kernel(const float* pred, const float* y,
                  float* err, const int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        err[i] = pred[i] - y[i];
    }
}
"""

# Compile kernels once at module load time
_forward_kernel = cp.RawKernel(_forward_kernel_code, 'forward_kernel')
_error_kernel   = cp.RawKernel(_error_kernel_code,   'error_kernel')


def train_linear_regression_gpu(d_X, d_y, learning_rate=0.05, epochs=200):
    """
    Gradient Descent using raw CUDA kernels (compiled via CuPy RawKernel).
    Forward pass & error computation run as custom CUDA C kernels.
    Gradient reduction uses CuPy's massively-parallel BLAS routines.
    """
    n_samples, n_features = d_X.shape
    d_w    = cp.zeros(n_features, dtype=cp.float32)
    d_b    = cp.float32(0.0)
    d_pred = cp.zeros(n_samples,  dtype=cp.float32)
    d_err  = cp.zeros(n_samples,  dtype=cp.float32)

    threads = 256
    blocks  = int(cp.ceil(cp.float32(n_samples) / threads))

    # flatten X for the kernel (row-major, already the case for C-contiguous)
    d_X_flat = cp.ascontiguousarray(d_X)

    # Scalar wrapper arrays so we can pass by pointer
    d_b_arr = cp.array([0.0], dtype=cp.float32)

    for _ in range(epochs):
        # ── CUDA Kernel: forward pass (predict) ──────────────────────────────
        _forward_kernel(
            (blocks,), (threads,),
            (d_X_flat, d_w, d_b_arr[0], d_pred, cp.int32(n_samples), cp.int32(n_features))
        )

        # ── CUDA Kernel: compute element-wise errors ──────────────────────────
        _error_kernel(
            (blocks,), (threads,),
            (d_pred, d_y, d_err, cp.int32(n_samples))
        )

        # ── CuPy BLAS: gradient accumulation (highly parallelised on GPU) ────
        d_dw = (2.0 / n_samples) * cp.dot(d_X_flat.T, d_err)
        d_db = (2.0 / n_samples) * cp.sum(d_err)

        d_w    -= learning_rate * d_dw
        d_b_arr[0] -= learning_rate * d_db

    cp.cuda.Stream.null.synchronize()
    return d_w.get(), float(d_b_arr[0].get())


def run_gpu_pipeline(data_dir, num_files=2):
    print("--- Starting GPU Pipeline ---")
    start_time = time.time()

    # 1. Load data (identical to CPU pipeline – fair comparison)
    print(f"Loading first {num_files} Parquet datasets...")
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))[:num_files]

    dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    load_time = time.time()
    print(f"Loaded {len(df):,} rows in {load_time - start_time:.2f}s.")

    # 2. Preprocessing
    prep_start = time.time()
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                            'trip_distance', 'passenger_count', 'fare_amount'])
    df['duration_min'] = (
        (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])
        .dt.total_seconds() / 60.0
    )
    df = df[(df['fare_amount'] > 0)    & (df['fare_amount'] < 500)]
    df = df[(df['trip_distance'] > 0)  & (df['trip_distance'] < 100)]
    df = df[(df['duration_min'] > 0)   & (df['duration_min'] < 200)]

    # ── Transfer to GPU memory ────────────────────────────────────────────────
    X_cpu = df[['trip_distance', 'passenger_count', 'duration_min']].values.astype(np.float32)
    d_X   = cp.asarray(X_cpu)
    # Standardise on the GPU
    d_X   = (d_X - cp.mean(d_X, axis=0)) / cp.std(d_X, axis=0)
    d_y   = cp.asarray(df['fare_amount'].values.astype(np.float32))
    prep_time = time.time()
    print(f"Preprocessing + GPU mem-transfer finished in {prep_time - prep_start:.2f}s.")

    # 3. Training (warm-up compile + timed run)
    print("Compiling CUDA kernels & warming up GPU...")
    train_linear_regression_gpu(d_X[:512], d_y[:512], epochs=1)   # JIT warm-up

    print("Starting Gradient Descent on GPU with custom CUDA kernels…")
    train_start = time.time()
    w, b = train_linear_regression_gpu(d_X, d_y, learning_rate=0.05, epochs=200)
    train_time = time.time()
    print(f"Training GPU finished in {train_time - train_start:.2f}s.")
    print(f"Model  →  Weights: {w}, Bias: {b:.4f}")

    total_time = time.time() - start_time
    print(f"Total GPU Pipeline Time: {total_time:.2f}s.")

    return {
        'total_time': total_time,
        'load_time':  load_time - start_time,
        'prep_time':  prep_time - prep_start,
        'train_time': train_time - train_start,
    }


if __name__ == "__main__":
    run_gpu_pipeline(r"c:\Users\RUTIKA\Desktop\uber\Data", num_files=2)
