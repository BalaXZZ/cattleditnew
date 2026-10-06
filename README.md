# 🐄 Cattle Muzzle Identification

A local application for **enrolling individual cows and identifying them from muzzle photographs**.

The system uses a trained YOLO detector to extract the cow's muzzle from full-cow photographs, followed by a CNN + Vision Transformer embedding system to compare the muzzle against enrolled cows.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [1. Clone the Repository](#1-clone-the-repository)
- [2. Create a Virtual Environment](#2-create-a-virtual-environment)
- [3. Install Dependencies](#3-install-dependencies)
- [4. Select Python Interpreter in VS Code](#4-select-python-interpreter-in-vs-code)
- [5. Obtain the Trained Models](#5-obtain-the-trained-models)
- [6. Install the Model Bundle](#6-install-the-model-bundle)
- [7. Verify the Installation](#7-verify-the-installation)
- [8. Start the Application](#8-start-the-application)
- [9. Open the Application](#9-open-the-application)
- [10. Enroll a Cow](#10-enroll-a-cow)
- [11. Identify a Cow](#11-identify-a-cow)
- [Stopping the Application](#stopping-the-application)
- [Troubleshooting](#-troubleshooting)
- [Quick Setup Summary](#-quick-setup-summary)

---

# 🛠 Prerequisites

Before starting, make sure the following are installed on your Windows laptop:

### Required

- **Git**
- **Visual Studio Code**
- **Python 3.12**

> ⚠️ **Important:** Python 3.12 is the tested Python version for this project.  
> Do not use Python 3.14 for this environment.

### Check your installed Python versions

Open **PowerShell** or the VS Code terminal and run:

```powershell
py -0
```

You should see Python 3.12 in the list.

You can also check directly:

```powershell
py -3.12 --version
```

Expected output:

```text
Python 3.12.x
```

---

# 1. 📥 Clone the Repository

Open **PowerShell** or the **VS Code terminal**.

Run:

```powershell
git clone https://github.com/BalaXZZ/cattleditnew.git
```

Move into the project directory:

```powershell
cd cattleditnew
```

Open the project in VS Code:

```powershell
code .
```

Your project should now be open in VS Code.

---

# 2. 🧠 Create a Virtual Environment

A virtual environment keeps the project's Python packages separate from other Python projects on your laptop.

From the project root directory, run:

```powershell
py -3.12 -m venv .detector-venv
```

This creates:

```text
cattleditnew/
└── .detector-venv/
```

You do **not** need to manually activate the environment because the commands below explicitly use the Python executable inside `.detector-venv`.

---

# 3. 📦 Install Dependencies

## Step 1 — Upgrade pip

Run:

```powershell
.\.detector-venv\Scripts\python.exe -m pip install --upgrade pip
```

---

## Step 2 — Install PyTorch

For a **CPU-only setup**, run:

```powershell
.\.detector-venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

This installs:

- PyTorch `2.11.0`
- Torchvision `0.26.0`

### GPU users

If your laptop has a supported NVIDIA GPU, you can install the appropriate CUDA version instead.

The original development machine used:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
```

For the first setup, however, **CPU is the simplest and most portable option**.

---

## Step 3 — Install Project Dependencies

Run:

```powershell
.\.detector-venv\Scripts\python.exe -m pip install -r requirements.txt
```

Wait for the installation to finish.

---

# 4. 🧭 Select Python Interpreter in VS Code

VS Code needs to know which Python environment should be used for this project.

### Step 1

Install the **Python extension by Microsoft** in VS Code if you haven't already.

Open:

```text
Extensions
```

or press:

```text
Ctrl + Shift + X
```

Search for:

```text
Python
```

Install the Python extension published by **Microsoft**.

---

### Step 2

Press:

```text
Ctrl + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Select the interpreter inside your project:

```text
.detector-venv\Scripts\python.exe
```

It may also appear as:

```text
Python 3.12.x ('.detector-venv': venv)
```

### If "Python: Select Interpreter" does not appear

Make sure the **Microsoft Python extension** is installed and restart VS Code.

You can also verify the environment directly from the terminal:

```powershell
.\.detector-venv\Scripts\python.exe --version
```

Expected output:

```text
Python 3.12.x
```

---

# 5. ⚠️ Obtain the Trained Models

> **Important:** Cloning this repository only downloads the project code.  
> The trained model weights are **not included in Git**.

You need to obtain the following file from the project owner or your team's private file-sharing system:

```text
cattle-runtime-v1.zip
```

You also need the corresponding:

```text
SHA256 checksum
```

The runtime bundle contains the trained components required by the application, including:

- YOLO muzzle detector
- ResNet-18 encoder
- DINOv2-small encoder
- Calibration data
- Capture policy
- Active model configuration

### 🔐 Important

Do not download model files from untrusted sources.

The runtime package should come from your trusted project owner/team.

---

# 6. 📥 Install the Model Bundle

Once you have:

```text
cattle-runtime-v1.zip
```

and its SHA256 checksum, install it using:

```powershell
.\.detector-venv\Scripts\python.exe scripts/runtime_bundle.py install --bundle "C:\path\to\cattle-runtime-v1.zip" --sha256 YOUR_CHECKSUM
```

### Replace these values

Replace:

```text
C:\path\to\cattle-runtime-v1.zip
```

with the actual location of your ZIP file.

For example:

```powershell
.\.detector-venv\Scripts\python.exe scripts/runtime_bundle.py install --bundle "C:\Users\Bala\Downloads\cattle-runtime-v1.zip" --sha256 YOUR_CHECKSUM
```

Replace:

```text
YOUR_CHECKSUM
```

with the SHA256 checksum provided by the project owner.

---

# 7. ✅ Verify the Installation

After installing the runtime bundle, run:

```powershell
.\.detector-venv\Scripts\python.exe scripts/check_runtime.py
```

This checks whether the required runtime files and models are available.

If the check completes successfully, your model environment is ready.

---

# 8. ▶️ Start the Application

From the project root, run:

```powershell
.\.detector-venv\Scripts\python.exe scripts/serve_identity_ui.py --device cpu --port 8766
```

You should see the application start in the terminal.

> ⚠️ **Keep this terminal open while using the application.**

The application is running locally on your computer.

---

# 9. 🌐 Open the Application

Open your web browser and go to:

```text
http://127.0.0.1:8766/
```

You should now see the **Cattle Muzzle Identification** interface.

---

# 10. 🐄 Enroll a Cow

Before identifying a cow, you need to enroll it in the local registry.

### Go to:

```text
Enroll
```

### Provide:

- A unique **Cow ID**
- Cow details
- One or more photographs

You can upload:

- Muzzle crops
- Full-cow photographs

The application can use the trained YOLO detector to extract the muzzle from full-cow images.

### Recommended enrollment images

For better recognition, use several clear images of the same cow:

- Front muzzle view
- Left-side view
- Right-side view
- Different angles
- Good lighting
- Clear, unobstructed muzzle

Complete any requested **quality or duplicate review**.

> Enrollment stores image templates for the cow. It does **not** retrain the neural networks.

---

# 11. 🔍 Identify a Cow

Go to:

```text
Identify
```

Upload a cow photograph and select the appropriate image type.

The system will:

1. Process the uploaded image.
2. Detect/crop the muzzle if necessary.
3. Generate an embedding using the active CNN + ViT model.
4. Compare the embedding against the enrolled gallery.
5. Apply the calibrated similarity threshold.
6. Return the cow's ID if the match is sufficiently confident.

### If a match is accepted

The application displays the enrolled cow's:

```text
Cow ID
Cow details
```

### If the cow is unknown or uncertain

The application rejects the match and directs you toward enrollment/review.

> The system does **not** reveal the nearest enrolled cow's details when a query is rejected.

---

# 🛑 Stopping the Application

When you are finished, go back to the terminal where the application is running.

Press:

```text
Ctrl + C
```

The local server will stop.

---

# 📁 Important Project Storage

The project stores different types of files in different locations.

| Location | Purpose | In Git? |
|---|---|---|
| `src/` | Application/model code | ✅ Yes |
| `scripts/` | Setup and utility scripts | ✅ Yes |
| `ui/` | User interface | ✅ Yes |
| `requirements.txt` | Python dependencies | ✅ Yes |
| `.vscode/` | VS Code configuration | ✅ Yes |
| `docs/` | Documentation | ✅ Yes |
| `outputs/embeddings/` | Installed model artifacts | ❌ No |
| `outputs/yolo/` | YOLO model artifacts | ❌ No |
| `data/registry/cows.sqlite3` | Cow IDs and templates | ❌ No |
| `data/registry/captures/` | Enrolled photographs | ❌ No |
| `data/raw/` | Private training images | ❌ No |
| `data/muzzle_crops/` | Private muzzle images | ❌ No |
| `work/` | Temporary files | ❌ No |
| `.detector-venv/` | Python virtual environment | ❌ No |

### ⚠️ Backup

If you need to back up the enrolled cows, back up these together:

```text
data/registry/cows.sqlite3
data/registry/captures/
```

The database and captured images should be kept together.

---

# 🔧 Troubleshooting

## Python 3.12 is not found

Run:

```powershell
py -0
```

If Python 3.12 is not listed, install Python 3.12 and try again.

---

## `Python: Select Interpreter` is missing

Install the **Python extension by Microsoft** in VS Code.

Then restart VS Code and press:

```text
Ctrl + Shift + P
```

Search:

```text
Python: Select Interpreter
```

---

## Virtual environment does not exist

Run:

```powershell
py -3.12 -m venv .detector-venv
```

Then verify:

```powershell
.\.detector-venv\Scripts\python.exe --version
```

---

## Model/runtime errors

Make sure you have installed:

```text
cattle-runtime-v1.zip
```

using the runtime bundle installation command.

Then run:

```powershell
.\.detector-venv\Scripts\python.exe scripts/check_runtime.py
```

---

## Port 8766 is already in use

Start the application on another port.

For example:

```powershell
.\.detector-venv\Scripts\python.exe scripts/serve_identity_ui.py --device cpu --port 8767
```

Then open:

```text
http://127.0.0.1:8767/
```

---

## Application is slow

The CPU configuration works without an NVIDIA GPU, but model inference will generally be faster on a supported NVIDIA GPU.

For CPU:

```text
--device cpu
```

---

# 🧪 Running Project Tests

After completing the setup, you can run the available tests.

### Runtime bundle test

```powershell
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_runtime_bundle.py
```

### Biometrics test

```powershell
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_biometrics.py
```

### CNN + ViT test

```powershell
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_cnn_vit.py
```

### MobileNet experiment test

```powershell
.\.detector-venv\Scripts\python.exe -m unittest discover -s scripts -p test_mobilenet_experiment.py
```

> `test_identity_ui.py` requires the original private evaluation data and runtime models, so it is not a clean-clone test.

---

# 📊 Current Model Performance

The active **CNN + ViT** system achieved:

```text
78.1% held-out full-muzzle Rank-1 retrieval
```

A MobileNetV3-L + CosFace experiment achieved:

```text
79.2%
```

However, the MobileNet model correctly accepted fewer known cows at its fixed threshold, so it was **not selected as the active model**.

> Development validation scores above 90% should not be interpreted as held-out application accuracy.

The prototype does not establish statewide reliability or guaranteed biometric uniqueness.

---

# 🔄 Quick Setup Summary

If everything is already installed, the basic workflow is:

```text
1. Clone repository
       ↓
2. Create Python 3.12 virtual environment
       ↓
3. Install PyTorch
       ↓
4. Install requirements.txt
       ↓
5. Select .detector-venv in VS Code
       ↓
6. Obtain cattle-runtime-v1.zip
       ↓
7. Install runtime bundle
       ↓
8. Run check_runtime.py
       ↓
9. Start the application
       ↓
10. Open http://127.0.0.1:8766/
       ↓
11. Enroll cows
       ↓
12. Identify cows
```

## 🚀 The three commands you will eventually use most

### Start the application

```powershell
.\.detector-venv\Scripts\python.exe scripts/serve_identity_ui.py --device cpu --port 8766
```

### Open the application

```text
http://127.0.0.1:8766/
```

### Stop the application

```text
Ctrl + C
```

---

## 🔐 Important Notes

- Do **not** commit trained model weights to Git.
- Do **not** commit enrolled cow photographs or the local registry.
- Keep the runtime model bundle in trusted/private storage.
- Keep training, validation, and test cow identities separate.
- A failed identification does **not** necessarily mean that the cow is unregistered.
- Partial, blurred, or occluded muzzle photographs may produce poor results.
- The application currently binds to `localhost` and does not provide production authentication or deployment security.