# Cattle muzzle identification

Local application for enrolling individual cows and finding their IDs from muzzle
photographs. Full cow photos are cropped with a trained YOLO11n detector. The active
embedding system combines ResNet-18 and DINOv2-small and compares the image with
the enrolled gallery using calibrated cosine similarity thresholds.

Features: multiple enrollment photos, galleries, cow details, unknown-cow rejection,
and quality/duplicate review. Each photo can be up to 15 MB and 24 megapixels.
Rejected queries do not reveal the nearest cow's details.

## Clone and open in VS Code

```powershell
git clone https://github.com/BalaXZZ/cattleditnew.git
cd cattleditnew
code .
```

Alternatively open the cloned folder using VS Code's File > Open Folder.
Run the commands below in its terminal, from the repository root.

## Windows setup (CPU)

Install Git, VS Code and Python **3.12**. Python 3.14 is not the tested ML environment.
Activation is optional when using the explicit interpreter path:

```powershell
py -3.12 -m venv .detector-venv
.\.detector-venv\Scripts\python.exe -m pip install --upgrade pip
.\.detector-venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.detector-venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use **Python: Select Interpreter** in VS Code to select this environment.
CPU inference does not require an NVIDIA GPU. For supported NVIDIA systems, install
the matching PyTorch/CUDA build instead of CPU wheels. The original machine uses
torch 2.11.0+cu128 and torchvision 0.26.0+cu128; CPU is the portable starting point.

## Install trained models and run

**A clone contains code, not trained weights.** Obtain `cattle-runtime-v1.zip` and
its SHA256 checksum from the project owner through the team's private file sharing.
The package contains both active encoders, calibration, capture policy, the active
model pointer and the YOLO detector. It contains no cow photos or enrolled database.
Install only checkpoints supplied by your trusted team.

```powershell
.\.detector-venv\Scripts\python.exe scripts/runtime_bundle.py install --bundle "C:\path\to\cattle-runtime-v1.zip" --sha256 CHECKSUM_FROM_OWNER
.\.detector-venv\Scripts\python.exe scripts/check_runtime.py
.\.detector-venv\Scripts\python.exe scripts/serve_identity_ui.py --device cpu --port 8766
```

Open **http://127.0.0.1:8766/**. Keep the terminal running; stop with Ctrl+C.
`Start-CattleUI.ps1` is an alternative Windows launcher and selects CUDA when available.
If PowerShell restricts scripts, use the Python command above. If the port is busy,
choose another `--port` and open that port in your browser.

### Linux / macOS

Create the environment with `python3.12 -m venv .detector-venv`. Replace the Windows
interpreter path in the commands above with `.detector-venv/bin/python`.
Linux CPU wheels use the same PyTorch index. On macOS install torch/torchvision
without `--index-url`. Windows is the tested platform; run the smoke check on others.

## Enroll and identify

Each fresh clone begins with an empty registry. In **Enroll**, supply a unique cow
ID, details and one or more distinct photos of the same cow. Several clear frontal
and side views are recommended. Select whether the uploads are muzzle crops or full
photos. Complete any requested quality/duplicate review. In **Identify**, upload a
photo and select its type. Accepted matches show the enrolled ID and details;
unknown or uncertain cows are directed to enrollment/review.

Enrollment stores image templates; it does not retrain the networks. Separate
teammates' local registries do not automatically synchronize. Partial, blurred or
occluded captures can fail. A failed match alone does not prove a cow is unregistered.

## Storage

| Location | Purpose | In Git? |
|---|---|---|
| `src/`, `scripts/`, `ui/` | Models, tools, interface | Yes |
| `requirements.txt`, `.vscode/`, `docs/` | Setup and documentation | Yes |
| `outputs/embeddings/`, `outputs/yolo/` | Installed model artifacts | No |
| `data/registry/cows.sqlite3` | Cow IDs, details and embedding templates | No |
| `data/registry/captures/` | Saved enrolled photos | No |
| `data/raw/`, `data/muzzle_crops/` | Private training images | No |
| `work/`, `.detector-venv/` | Temporary files and environment | No |

Back up the registry and its capture directory together. Model replacement requires
registry embedding migration and compatible calibration. The bundle installer
refuses to replace different existing weights. The app binds only to localhost and
has no authentication or production deployment security layer.

## Checks and experiments

```powershell
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_runtime_bundle.py
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_biometrics.py
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_cnn_vit.py
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_mobilenet_experiment.py
```

These checks use synthetic/temporary inputs. `test_identity_ui.py` also requires the
original private evaluation crops and runtime models; it is not a clean-clone test.
Training scripts require separately supplied datasets/annotations. Keep training,
validation and test cow identities separate, and augment after splitting.

The active CNN + ViT model reached **78.1% held-out full-muzzle rank-1 retrieval**.
A MobileNetV3-L + CosFace candidate reached **79.2%**, but correctly accepted fewer
known cows at its fixed threshold, so it was not activated. Development validation
scores above 90% are not held-out application accuracy. The prototype does not
establish statewide reliability or guaranteed biometric uniqueness.
See `docs/development-history.md` for chronological experiments and tools.

The owner can regenerate the runtime package without private data:

```powershell
.\.detector-venv\Scripts\python.exe scripts/runtime_bundle.py export --bundle outputs/cattle-runtime-v1.zip
```

The command prints a checksum for teammates. Share model artifacts through private
team storage, not ordinary Git commits. Dependencies and pretrained weights retain
their own licenses; review their terms before commercial distribution (including
Ultralytics and DINOv2). No project software license is assigned here.
