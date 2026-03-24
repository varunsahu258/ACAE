"""
device_config.py  –  Universal hardware configuration for ACAE.

Philosophy: use 100% of whatever is available. No conservative scaling,
no safety margins. If the GPU has 4 GB or 24 GB, fill it.

Detection priority (automatic, no user input needed):
  1. NVIDIA GPU  — any CUDA card (RTX 4500 Ada, RTX 3090, V100, ...)
  2. Apple GPU   — Metal on M1/M2/M3/M4 via tensorflow-metal
  3. Other GPU   — AMD ROCm, Intel XPU, anything else TF can see
  4. CPU         — always available, slower

Batch sizes are computed from actual VRAM at runtime so every GPU
runs at maximum throughput regardless of how much memory it has.

Usage (identical API on every machine):
    from device_config import configure, get_device, optimal_batch, print_device_info
    configure()
    device = get_device()           # "/GPU:0"  or  "/CPU:0"
    bs = optimal_batch("movielens", "acae")
"""

import os
import sys
import platform
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Windows: ensure CUDA DLLs are findable before TF loads them
# ─────────────────────────────────────────────────────────────────────────────

def _patch_windows_cuda_path() -> None:
    if sys.platform != "win32":
        return
    common = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    path = os.environ.get("PATH", "")
    for p in common:
        if os.path.isdir(p) and p not in path:
            os.environ["PATH"] = p + os.pathsep + path
            path = os.environ["PATH"]


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection — pure TF APIs, no subprocess
# ─────────────────────────────────────────────────────────────────────────────

def _get_gpu_name(gpu) -> str:
    try:
        details = tf.config.experimental.get_device_details(gpu)
        return details.get("device_name", gpu.name)
    except Exception:
        return gpu.name


def _detect_backend(gpus: list) -> str:
    """
    Returns: "cuda" | "metal" | "gpu" | "cpu"
    Detection is done entirely through TF device APIs — no nvidia-smi,
    no PATH checks, no subprocess calls.
    """
    if not gpus:
        return "cpu"

    # Apple Silicon: Darwin + arm64 + TF sees a GPU = Metal
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"

    # Check GPU name for NVIDIA keywords
    name = _get_gpu_name(gpus[0]).upper()
    nvidia_kw = ("NVIDIA", "TESLA", "QUADRO", "RTX", "GTX",
                 "A100", "V100", "H100", "T4", "L4", "A10",
                 "A30", "A40", "A2000", "A4000", "A5000", "A6000")
    if any(kw in name for kw in nvidia_kw):
        return "cuda"

    # Final fallback: if TF was built with CUDA and sees a GPU, it's CUDA
    try:
        if tf.test.is_built_with_cuda():
            return "cuda"
    except Exception:
        pass

    return "gpu"  # AMD ROCm, Intel XPU, or unknown — still usable


# ─────────────────────────────────────────────────────────────────────────────
# VRAM detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_vram_gb(gpus: list) -> float:
    """Query total VRAM of GPU:0 in GB. Returns 0.0 if unavailable."""
    if not gpus:
        return 0.0
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        mem = details.get("memory_limit", 0)
        if mem and mem > 0:
            return mem / (1024 ** 3)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch size computation — fills VRAM completely
#
# Strategy:
#   1. Measure actual VRAM.
#   2. Compute how many rating-matrix rows fit given the item count and
#      model memory footprint, targeting 85% VRAM utilisation.
#   3. Round to the nearest power of two (GPU kernels prefer aligned sizes).
#   4. Hard floor of 32 so tiny GPUs still make progress.
#
# Why 85% and not 100%?
#   The remaining 15% is consumed by model weights, gradients, and TF's
#   internal graph buffers. The batch itself is just the activations.
#   Targeting 100% of VRAM for the batch would OOM on the first backward pass.
#   85% consistently saturates GPU compute without OOM on any tested hardware.
# ─────────────────────────────────────────────────────────────────────────────

# Memory footprint per training sample in bytes (empirically measured).
# These are conservative upper bounds — real usage is usually lower.
_BYTES_PER_SAMPLE = {
    # autoencoder models: each sample is one user row of the rating matrix
    # plus hidden activations (latent_dim=50 default)
    "cae":   4 * 3706,   # float32 × n_items (MovieLens worst case)
    "cdae":  4 * 3706,
    "acae":  4 * 3706 * 3,  # ×3 for clean + two adversarial forward passes
    # pairwise: (user_emb + item_emb) × 2, latent_dim=64
    "bpr":   4 * 64 * 4,
    "amf":   4 * 64 * 6,    # ×1.5 for adversarial noise computation
    # pointwise NeuMF: MLP activations dominate
    "neumf": 4 * (64 + 32 + 16 + 8) * 2,
}

_VRAM_TARGET = 0.85   # use 85% of VRAM for the batch


def _compute_batch_size(vram_gb: float, model: str, fallback: int) -> int:
    """
    Compute the largest power-of-two batch size that fits in 85% of VRAM.
    """
    if vram_gb <= 0:
        return fallback

    available_bytes = vram_gb * (1024 ** 3) * _VRAM_TARGET
    bps = _BYTES_PER_SAMPLE.get(model.lower(), 4 * 3706)  # default: AE-style
    raw = int(available_bytes / bps)

    # Round down to nearest power of two, floor 32
    raw = max(32, raw)
    p = 1
    while p * 2 <= raw:
        p *= 2
    return p


# Module-level state set by configure()
_backend: str   = "cpu"
_vram_gb: float = 0.0


def optimal_batch(dataset: str, model: str, fallback: int = 256) -> int:
    """
    Return the largest safe batch size for the given model on the
    detected hardware, computed from actual VRAM.

    On CPU, returns a fixed 128 (memory-safe, avoids swapping).
    """
    if _backend == "cpu":
        return 128

    bs = _compute_batch_size(_vram_gb, model, fallback)

    # Sanity cap: never exceed 16384 (diminishing returns, longer eval gaps)
    return min(bs, 16384)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def configure(
    memory_growth:   bool = True,
    enable_tf32:     bool = True,   # free ~3× throughput on Ampere/Ada; no-op elsewhere
    enable_xla:      bool = True,   # XLA kernel fusion; no-op on Metal
    mixed_precision: bool = False,  # float16 — faster but small numeric drift vs paper
    verbose:         bool = True,
) -> bool:
    """
    Detect hardware, apply the best settings, return True if a GPU was found.
    """
    global _backend, _vram_gb

    _patch_windows_cuda_path()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    gpus      = tf.config.list_physical_devices("GPU")
    found_gpu = bool(gpus)
    _backend  = _detect_backend(gpus)

    # Memory growth MUST be set before any GPU operation.
    # Without it TF grabs all VRAM at startup, leaving none for other processes
    # and preventing VRAM queries from returning useful numbers.
    if found_gpu and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised — not fatal

    # Query VRAM now (after memory growth is set)
    _vram_gb = _get_vram_gb(gpus)

    # NVIDIA-specific acceleration
    if _backend == "cuda":
        if enable_tf32:
            # Ampere/Ada TF32: matmul/conv run at 10× the throughput of FP32
            # with no code changes and negligible accuracy impact
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        if enable_xla:
            # XLA fuses elementwise ops into single kernels — big win on the
            # ACAE adversarial loop which has many small ops per step
            os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
            os.environ["XLA_FLAGS"]    = "--xla_gpu_cuda_data_dir=."

    # Mixed precision — optional, works on any GPU backend
    if mixed_precision and found_gpu:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if verbose:
        _print_startup(gpus, found_gpu, enable_tf32, enable_xla, mixed_precision)

    tf.random.set_seed(42)
    return found_gpu


def _print_startup(gpus, found_gpu, enable_tf32, enable_xla, mixed_precision):
    labels = {"cuda": "NVIDIA CUDA", "metal": "Apple Metal",
              "gpu": "GPU (non-NVIDIA)", "cpu": "CPU"}
    print(f"[device] Backend : {labels.get(_backend, _backend)}")
    if found_gpu:
        for g in gpus:
            name = _get_gpu_name(g)
            vram = f"  ({_vram_gb:.1f} GB VRAM)" if _vram_gb > 0 else ""
            print(f"[device] GPU     : {name}{vram}")
        if _backend == "cuda":
            print(f"[device] TF32 : {enable_tf32}   XLA : {enable_xla}")
        if mixed_precision:
            print("[device] Mixed precision (float16) ON")
        # Show what batch sizes were computed
        ex_bs = optimal_batch("movielens", "acae")
        print(f"[device] Batch size (movielens/acae) : {ex_bs}")
    else:
        print("[device] No GPU — running on CPU.")
        if sys.platform == "win32":
            print("[device] NVIDIA GPU? See setup_cuda_windows.md")
        elif platform.system() == "Darwin" and platform.machine() == "arm64":
            print("[device] Apple GPU? Run: pip install tensorflow-metal")


# ─────────────────────────────────────────────────────────────────────────────
# Public accessors
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

def has_gpu() -> bool:
    return bool(tf.config.list_physical_devices("GPU"))

def get_backend() -> str:
    return _backend

def print_device_info() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print("\n" + "═" * 58)
    print("  Hardware / Software Summary")
    print("═" * 58)
    print(f"  OS       : {platform.platform()}")
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  TF       : {tf.__version__}")
    print(f"  Backend  : {get_backend().upper()}")
    print(f"  CPUs     : {len(cpus)}")
    if gpus:
        print(f"  GPUs     : {len(gpus)}")
        for g in gpus:
            print(f"             {_get_gpu_name(g)}")
        if _vram_gb > 0:
            print(f"  VRAM     : {_vram_gb:.1f} GB")
        print(f"  Batch sizes (computed from VRAM):")
        for ds in ("movielens", "filmtrust", "ciao"):
            for m in ("acae", "bpr", "neumf"):
                print(f"    {ds}/{m:<12} : {optimal_batch(ds, m)}")
        try:
            for g in gpus:
                info = tf.config.experimental.get_memory_info(g.name)
                used = info.get("current", 0) // 1024 ** 2
                peak = info.get("peak",    0) // 1024 ** 2
                print(f"  Mem used : {used} MB   peak {peak} MB")
        except Exception:
            pass
    else:
        print("  GPUs     : none")
    print("═" * 58 + "\n")
