import argparse
from pathlib import Path

import google.cloud.aiplatform as aip
from kfp import compiler, dsl
from kfp.dsl import component


DEFAULT_PROJECT = "biology-multimodal-2026"
DEFAULT_REGION = "us-central1"
DEFAULT_IMAGE = "us-docker.pkg.dev/biology-multimodal-2026/tcga-rna-repo/tcga_rna_model:dev-latest"
DEFAULT_DATA_PATH = "gs://biology-predict-bucket/Liver RNA Data.csv"
DEFAULT_MODEL_OUTPUT_PATH = "gs://biology-predict-bucket/pca_random_forest_model.joblib"
DEFAULT_MODEL_DISPLAY_NAME = "tcga_rna_model_dev"
DEFAULT_ENDPOINT_DISPLAY_NAME = "tcga_rna_endpoint_dev"
DEFAULT_PIPELINE_ROOT = "gs://biology-predict-bucket/pipeline-root/dev"
DEFAULT_PIPELINE_SPEC = Path(__file__).resolve().parents[1] / "pipeline.yaml"
DEFAULT_JOB_DISPLAY_NAME = "tcga-rna-pipeline-dev"


@component(
    packages_to_install=[
        "scikit-learn",
        "joblib",
        "pandas",
        "numpy",
        "fsspec",
        "gcsfs",
        "google-cloud-storage",
    ]
)
def train_model(
    data_path: str,
    model_output_path: str,
) -> float:
    import joblib
    import numpy as np
    import pandas as pd
    from google.cloud import storage
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    dataframe = pd.read_csv(data_path)
    transformed = np.log(dataframe.drop(columns=["sample_type_id"]) + 1e-5)

    pca = PCA(n_components=3)
    features = pca.fit_transform(transformed)
    target = dataframe["sample_type_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    score = balanced_accuracy_score(y_test, model.predict(X_test))

    local_model_path = "/tmp/pca_random_forest_model.joblib"
    local_preprocessor_path = "/tmp/pca_preprocessor.joblib"
    joblib.dump(model, local_model_path)
    joblib.dump(pca, local_preprocessor_path)

    bucket_name = model_output_path.replace("gs://", "").split("/")[0]
    object_path = "/".join(model_output_path.replace("gs://", "").split("/")[1:])

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(object_path).upload_from_filename(local_model_path)

    preprocessor_object_path = object_path.replace(
        "pca_random_forest_model.joblib",
        "pca_preprocessor.joblib",
    )
    bucket.blob(preprocessor_object_path).upload_from_filename(local_preprocessor_path)

    return score


@component(packages_to_install=["google-cloud-aiplatform"])
def upload_model(
    project: str,
    region: str,
    image_uri: str,
    model_display_name: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        serving_container_image_uri=image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
    )
    return model.resource_name


@component(packages_to_install=["google-cloud-aiplatform"])
def deploy_model(
    project: str,
    region: str,
    model_resource_name: str,
    endpoint_display_name: str,
) -> None:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project,
        location=region,
    )
    endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(display_name=endpoint_display_name)

    model = aiplatform.Model(model_resource_name)
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
    )


@dsl.pipeline(
    name="tcga-rna-pipeline",
    description="Train and deploy RNA classification model",
)
def tcga_pipeline(
    project: str,
    region: str,
    image_uri: str,
    data_path: str = DEFAULT_DATA_PATH,
    model_output_path: str = DEFAULT_MODEL_OUTPUT_PATH,
    model_display_name: str = DEFAULT_MODEL_DISPLAY_NAME,
    endpoint_display_name: str = DEFAULT_ENDPOINT_DISPLAY_NAME,
):
    train_task = train_model(
        data_path=data_path,
        model_output_path=model_output_path,
    )

    upload_task = upload_model(
        project=project,
        region=region,
        image_uri=image_uri,
        model_display_name=model_display_name,
    ).after(train_task)

    deploy_model(
        project=project,
        region=region,
        model_resource_name=upload_task.output,
        endpoint_display_name=endpoint_display_name,
    ).after(upload_task)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and run the Vertex AI pipeline for the dev environment")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP project id")
    parser.add_argument("--region", default=DEFAULT_REGION, help="GCP region")
    parser.add_argument("--image-uri", default=DEFAULT_IMAGE, help="Serving image URI")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="GCS CSV path for training data")
    parser.add_argument("--model-output-path", default=DEFAULT_MODEL_OUTPUT_PATH, help="GCS model output path")
    parser.add_argument("--model-display-name", default=DEFAULT_MODEL_DISPLAY_NAME, help="Vertex model name")
    parser.add_argument("--endpoint-display-name", default=DEFAULT_ENDPOINT_DISPLAY_NAME, help="Vertex endpoint name")
    parser.add_argument("--pipeline-root", default=DEFAULT_PIPELINE_ROOT, help="Vertex pipeline root path")
    parser.add_argument("--pipeline-spec", default=str(DEFAULT_PIPELINE_SPEC), help="Output pipeline YAML path")
    parser.add_argument("--job-display-name", default=DEFAULT_JOB_DISPLAY_NAME, help="Pipeline job display name")
    parser.add_argument("--compile-only", action="store_true", help="Compile the pipeline without submitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compiler.Compiler().compile(
        pipeline_func=tcga_pipeline,
        package_path=args.pipeline_spec,
    )

    if args.compile_only:
        print(f"Pipeline compiled at: {args.pipeline_spec}")
        return

    aip.init(project=args.project, location=args.region)
    job = aip.PipelineJob(
        display_name=args.job_display_name,
        template_path=args.pipeline_spec,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project": args.project,
            "region": args.region,
            "image_uri": args.image_uri,
            "data_path": args.data_path,
            "model_output_path": args.model_output_path,
            "model_display_name": args.model_display_name,
            "endpoint_display_name": args.endpoint_display_name,
        },
        enable_caching=True,
    )
    job.submit()
    print(f"Submitted pipeline job: {job.resource_name}")


if __name__ == "__main__":
    main()
