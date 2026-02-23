PROJECT=biology-multimodal-2026
REGION=us-central1
IMAGE=us-docker.pkg.dev/$(PROJECT)/docker/tcga_rna_model:0.2

build:
	gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT) .

run-pipeline:
	python scripts/pipeline.py