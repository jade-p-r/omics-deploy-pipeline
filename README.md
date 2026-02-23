# TCGA RNA Dataset ML Pipeline

This repository demonstrates a machine learning pipeline for classifying liver tissue samples using gene expression data. The model is trained on TCGA (The Cancer Genome Atlas) RNA-seq data and deployed as a REST API on Google Cloud's Vertex AI.

## Overview

This project showcases:
- **Data preprocessing**: PCA dimensionality reduction of gene expression data
- **Model training**: Random Forest classifier on reduced features
- **MLOps**: Containerized pipeline deployment on Vertex AI
- **API serving**: Flask application for online predictions via GCS artifacts

## Results

- **Balanced accuracy** on test set: **0.91**
- Feature reduction: ~20,000 genes → ~50 PCA components

## Quick Start

### Prerequisites

- Python 3.9+
- gcloud CLI installed and authenticated
- Docker
- Google Cloud Project with Vertex AI enabled

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd TCGA_RNA_Datasets

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Start the Flask application
python app.py
```

The API will be available at `http://localhost:8080`

## Project Structure

```
├── app.py                          # Flask REST API for predictions
├── scripts/
│   ├── train_model.py             # Model training script
│   ├── pipeline.py                # Vertex AI pipeline orchestration
│   ├── gene_preprocessor.py       # Gene expression preprocessing
│   └── explore_rna_data.py        # Exploratory data analysis
├── tests/                          # Test suite
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image definition
├── cloudbuild.yaml               # Cloud Build configuration
├── Makefile                       # Build and deployment targets
└── plots/
    └── pipeline.png              # Architecture diagram
```

## Pipeline

The ML pipeline consists of 3 stages:

![Pipeline Diagram](plots/pipeline.png)

1. **train_model** — PCA dimensionality reduction + Random Forest training on TCGA liver RNA data
2. **upload_model** — Registers the model in Vertex AI Model Registry
3. **deploy_model** — Deploys to a Vertex AI endpoint for online prediction

## API Usage

### Prediction Endpoint

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

Expected response:
```json
{
  "prediction": 1,
  "probabilities": [0.15, 0.85]
}
```

## Deployment

### Deploy to Vertex AI

```bash
make build          # Build Docker image
make deploy         # Deploy to Vertex AI endpoint
make run-pipeline   # Run the training pipeline
```

Or manually:

```bash
# Build and push container
gcloud builds submit --config cloudbuild.yaml

# Run pipeline
python scripts/pipeline.py
```

## Development

### Data Exploration

Run the exploratory analysis notebook:

```bash
python scripts/explore_rna_data.py
```

This generates:
- PCA/t-SNE visualizations
- Statistical analysis (ANOVA)
- Pathway enrichment results

### Testing

```bash
pytest
```

## Configuration

Set environment variables:

```bash
export GCP_PROJECT=<your-project-id>
export GCS_BUCKET=biology-predict-bucket
export REGION=us-central1
```

## Contributing

Contributions are welcome! Please:

1. Create a feature branch (`git checkout -b feature/my-feature`)
2. Commit your changes (`git commit -m 'Add feature'`)
3. Push to the branch (`git push origin feature/my-feature`)
4. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TCGA data courtesy of [The Cancer Genome Atlas](https://www.cancer.gov/tcga)
- Built with [scikit-learn](https://scikit-learn.org/), [Flask](https://flask.palletsprojects.com/), and [Google Cloud](https://cloud.google.com/)

## Citation

If you use this project in your research, please cite:

```
@software{tcga_rna_pipeline,
  title={TCGA RNA Dataset ML Pipeline},
  year={2026}
}
```