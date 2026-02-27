PROJECT ?= biology-multimodal-2026
REGION ?= us-central1
REPOSITORY ?= tcga-rna-repo
IMAGE_NAME ?= tcga_rna_model
TAG ?= dev-latest
IMAGE ?= us-docker.pkg.dev/$(PROJECT)/$(REPOSITORY)/$(IMAGE_NAME):$(TAG)
PIPELINE_SCRIPT ?= deployment/run_pipeline.py

.PHONY: help install build build-local push train-local pipeline-compile run-pipeline

help:
	@echo "Dev targets:"
	@echo "  make install           - install app dependencies"
	@echo "  make build             - build and push image with Cloud Build"
	@echo "  make build-local       - build image locally"
	@echo "  make push              - push local image to Artifact Registry"
	@echo "  make train-local       - run local training"
	@echo "  make pipeline-compile  - compile Vertex pipeline spec only"
	@echo "  make run-pipeline      - compile + submit Vertex pipeline job (dev)"

install:
	pip install -r app/requirements.txt

build:
	gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT) .

build-local:
	docker build -f app/Dockerfile -t $(IMAGE) .

push:
	docker push $(IMAGE)

train-local:
	python model/train.py

pipeline-compile:
	python $(PIPELINE_SCRIPT) \
		--project $(PROJECT) \
		--region $(REGION) \
		--image-uri $(IMAGE) \
		--compile-only

run-pipeline:
	python $(PIPELINE_SCRIPT) \
		--project $(PROJECT) \
		--region $(REGION) \
		--image-uri $(IMAGE)