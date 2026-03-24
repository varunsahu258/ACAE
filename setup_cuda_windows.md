# Setup Guide: RTX 4500 Ada + CUDA on Windows 11
## For ACAE Paper Replication

---

## Step 1 — NVIDIA Driver

1. Open **Device Manager** → Display Adapters → confirm RTX 4500 Ada is listed
2. Download the latest **Studio Driver** (more stable than Game Ready for ML):
   https://www.nvidia.com/Download/index.aspx
   - Product Type: RTX / Ada Lovelace
   - Recommended: 546.xx or newer
3. Install and **reboot**

Verify in PowerShell:
```powershell
nvidia-smi
```
You should see the GPU, driver version, and CUDA version listed.

---

## Step 2 — CUDA Toolkit 12.3

TensorFlow 2.15 requires CUDA 12.x + cuDNN 8.9.

1. Download CUDA Toolkit 12.3:
   https://developer.nvidia.com/cuda-12-3-0-download-archive
   - OS: Windows → x86_64 → 11 → exe (local)
2. Run the installer → choose **Custom** → install:
   - CUDA Toolkit
   - CUDA Documentation (optional)
   - CUDA Samples (optional)
   - **Uncheck** the Display Driver (you installed it in Step 1)

Default install path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3`

Verify:
```powershell
nvcc --version
# Expected: Cuda compilation tools, release 12.3
```

---

## Step 3 — cuDNN 8.9

1. Download cuDNN 8.9 for CUDA 12.x:
   https://developer.nvidia.com/rdp/cudnn-download
   (requires free NVIDIA Developer account)
   - Select: cuDNN v8.9.x for CUDA 12.x → Windows → zip

2. Extract the zip. Inside you'll find three folders:
   ```
   bin\
   include\
   lib\
   ```

3. Copy each folder's contents into your CUDA install:
   ```
   bin\      →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\
   include\  →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include\
   lib\      →  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64\
   ```

---

## Step 4 — Environment Variables

Open **System Properties** → **Advanced** → **Environment Variables**

Add to **System Path**:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp
```

Add new **System Variable**:
```
Variable name:  CUDA_PATH
Variable value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
```

**Reboot** after changing environment variables.

---

## Step 5 — Python Environment

Use Python 3.10 (most compatible with TF 2.15 on Windows).

Download Python 3.10: https://www.python.org/downloads/release/python-31011/
- Use the Windows installer (64-bit)
- Check "Add Python to PATH"

Create a dedicated virtual environment:
```powershell
cd C:\Users\<you>\projects\acae
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 6 — Install TensorFlow + Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# TensorFlow 2.15 — the last version with native Windows CUDA support
pip install tensorflow[and-cuda]==2.15.0

# Other dependencies
pip install numpy pandas scipy matplotlib tqdm
```

> **Note on TF versions and Windows:**
> TensorFlow 2.16+ dropped native Windows GPU support in the official package.
> TF 2.15 is the recommended version for Windows 11 + CUDA 12.x as of 2024.

---

## Step 7 — Verify CUDA is Detected

```powershell
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
if gpus:
    print('SUCCESS: RTX 4500 Ada detected')
else:
    print('FAIL: No GPU found - check CUDA/cuDNN install')
"
```

Expected output:
```
TF version: 2.15.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
SUCCESS: RTX 4500 Ada detected
```

---

## Step 8 — Run the Experiments

```powershell
# Activate venv if not already active
.\venv\Scripts\Activate.ps1

# MovieLens-1M downloads automatically
python run_experiments.py --dataset movielens

# For FilmTrust: place ratings.txt in data\filmtrust\
python run_experiments.py --dataset filmtrust

# For CiaoDVD: place movie-ratings.txt in data\ciao\
python run_experiments.py --dataset ciao

# All three at once
python run_experiments.py --all
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Could not load dynamic library 'cudart64_12.dll'` | CUDA bin not on PATH; re-check Step 4 and reboot |
| `Could not load dynamic library 'cudnn64_8.dll'` | cuDNN not copied correctly; re-check Step 3 |
| GPU detected but 0% utilisation | `nvidia-smi dmon` to confirm compute activity; check `TF_XLA_FLAGS` |
| OOM (out of memory) error | Reduce batch size in `device_config.py` BATCH dict |
| `XLA compilation failed` | Set `enable_xla=False` in `run_experiments.py` configure() call |
| TF 2.15 not found | `pip install tensorflow==2.15.0` (without `[and-cuda]`) then install CUDA manually |

---

## Expected Runtime on RTX 4500 Ada (Windows 11)

| Dataset    | All 6 models | ACAE only |
|------------|-------------|-----------|
| MovieLens  | ~25 min     | ~10 min   |
| FilmTrust  | ~8 min      | ~3 min    |
| CiaoDVD    | ~12 min     | ~5 min    |
| **Total**  | **~45 min** | **~18 min** |

First run is ~5 min slower due to XLA compilation (one-time cost per session).
