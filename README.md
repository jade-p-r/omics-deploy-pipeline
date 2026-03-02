# TCGA RNA Deployment Project


This repository runs a dev-focused Vertex AI workflow for training and deployment from a single pipeline entrypoint. The deployment process via Vertex AI Pipelines includes three steps: training, model upload and model deployment to an endpoint.

![Pipeline diagram](plots/pipeline.png)


## Project Architecture

```text
├── app/
│   ├── app.py                  # Flask serving app
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile
│
├── model/
│   ├── train.py                # Optional local training script
│   └── preprocessor.py         # Preprocessing logic
│
├── deployment/
│   └── run_pipeline.py         # Compile + submit Vertex AI pipeline job
│
├── notebooks/
│   └── exploration.ipynb
│
└── README.md
```

## Quick Start

### 1) Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

### 2) Train locally (optional)

```bash
python model/train.py
```

### 3) Build and push serving image

```bash
make build
```

### 4) Compile pipeline spec only (optional)

```bash
make pipeline-compile
```

### 5) Run full dev pipeline on Vertex AI

```bash
make run-pipeline
```

## Notes

- The root `Makefile` keeps only active dev workflow commands.
- Archived commands and scripts are stored in `tests/old_Makefile` and `tests/legacy/` (ignored by git).
