# AI-SOC Adversarial Defense System - Technical Specification

## 1. Project Overview
This project is an End-to-End AI Security Operations Center (AI-SOC) pipeline. It demonstrates the ability to not only build and train Deep Learning models but also proactively defend them against Adversarial Machine Learning attacks (such as FGSM and PGD). 

The system operates on a **Hybrid Workflow**:
* **Heavy compute (Model Training & Adversarial Attack Generation):** Executed on Kaggle/Cloud GPUs.
* **Production & Inference (API Gateway, Threat Detection, Logging):** Deployed on a Local Machine (CPU or Local GPU).

## 2. Core Architecture & Modules
The system is divided into 5 independent but interconnected modules:

1.  **API Gateway & Data Ingestion (`src/api/`):** A REST API built with FastAPI. It receives incoming requests (images/data), validates the payload using Pydantic, and handles preprocessing.
2.  **Threat Detection & Sanitization (`src/defense/`):** The "AI Firewall". It inspects incoming tensors for adversarial perturbations. If anomalous distributions are detected, it either sanitizes the input (e.g., via image smoothing/compression) or drops the request entirely.
3.  **Robust Inference Engine (`src/models/`):** The core Victim Model (e.g., a CNN like ResNet). It loads pre-trained weights (`.pth`) that have undergone Adversarial Training, ensuring high robust accuracy.
4.  **Red Team Simulator (`src/attacks/`):** Mathematical implementations of adversarial attacks (FGSM, PGD). Used strictly during the Cloud Training phase to generate poisoned data.
5.  **SOC Auditing & Alerting (`src/api/logger.py`):** A comprehensive logging system. It records all traffic, flags clean vs. sanitized requests, and outputs `[CRITICAL]` alerts to the terminal/log files when an attack is dropped.

## 3. Technology Stack
* **Deep Learning:** PyTorch, Torchvision
* **API & Backend:** FastAPI, Uvicorn, Pydantic
* **Data Processing:** NumPy, OpenCV/PIL
* **Environment:** Python 3.10+, Standard `logging` module

## 4. Directory Structure
```text
ai-soc-defense/
├── data/
│   ├── raw/                # Original dataset (e.g., GTSRB, CIFAR)
│   └── processed/          # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── config.py           # Global hyperparameters, paths, and thresholds
│   ├── data/
│   │   ├── dataset.py      # PyTorch Dataset classes and DataLoaders
│   │   └── transforms.py   # Augmentation and normalization pipelines
│   ├── models/
│   │   ├── cnn_classifier.py # The core CNN architecture (Victim Model)
│   │   └── soc_detector.py   # Anomaly detection model/logic
│   ├── attacks/
│   │   ├── fgsm.py         # Fast Gradient Sign Method implementation
│   │   └── pgd.py          # Projected Gradient Descent implementation
│   ├── defense/
│   │   ├── train.py        # Standard training loop
│   │   ├── adv_train.py    # Adversarial training loop (using PGD/FGSM)
│   │   └── sanitizer.py    # Input filtering/smoothing mechanisms
│   └── api/
│       ├── main.py         # FastAPI application, routing, and system integration
│       ├── schemas.py      # Pydantic models for request/response validation
│       └── logger.py       # Custom SOC logging configuration
├── weights/
│   ├── baseline_cnn.pth    # Weights of the vulnerable model
│   └── robust_cnn.pth      # Weights of the adversarially trained model
└── logs/
    └── soc_alerts.log      # Production log outputs
```
## 5. Strict Coding Guidelines for AI Assistant: 
When instructed to generate code for this project, the AI MUST adhere to the following constraints:
1. **Device Agnosticism (CRITICAL):** All local inference code MUST automatically detect hardware and safely map weights.Always use: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
2. **Always load weights safely:** torch.load(path, map_location=device)
3. **Type Hinting & OOP:** All functions and classes must include strict Python type hints (e.g., def forward(self, x: torch.Tensor) -> torch.Tensor:). Use Object-Oriented Programming principles where applicable.
4. **Modular Imports:** Assume the code is running from the root ai-soc-defense/ directory. Use absolute imports (e.g., from src.models.cnn_classifier import CNNClassifier).
5. **Error Handling & Logging:** Do not use plain print() statements in production code (src/api/, src/defense/). Use the custom logger defined in logger.py. Handle exceptions gracefully in the FastAPI endpoints.
6. **No Placeholder Logic:** Provide fully functional code. For complex math (like PGD calculations involving $L_\infty$ norm and $\epsilon$ steps), write the actual PyTorch tensor operations, not pass or TODO comments.
## 6. Execution Phases
* **Phase 1 (Local):** Setup boilerplate, API gateway, schemas, logging, and model architectures. Ensure the API can start without errors.
* **Phase 2 (Cloud/Kaggle):** Implement dataset loaders, standard training, and Red Team attacks (PGD/FGSM). Train models and export .pth weights.
* **Phase 3 (Local):** Implement the Threat Detection module, load .pth weights into the API, and test the End-to-End inference pipeline with simulated attack payloads.