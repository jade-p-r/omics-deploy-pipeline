import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.dsl import component

PROJECT = "biology-multimodal-2026"
REGION = "us-central1"
IMAGE = "us-docker.pkg.dev/biology-multimodal-2026/docker/tcga_rna_model:0.2"


@component(packages_to_install=["scikit-learn", "joblib", "pandas", "numpy", "fsspec", "gcsfs"])
def train_model(
    data_path: str,
    model_output_path: str,
) -> float:
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    import joblib
    from google.cloud import storage

    df = pd.read_csv(data_path)
    df_log = np.log(df.drop(columns=["sample_type_id"]) + 1e-5)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(df_log)
    y = df["sample_type_id"]

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    score = balanced_accuracy_score(y_test, rf.predict(X_test))

    #first dump, then upload to GCS 
    local_model_path = "/tmp/pca_random_forest_model.joblib"
    joblib.dump(rf, local_model_path)
    local_preprocessor_path = "/tmp/pca_preprocessor.joblib"
    joblib.dump(pca, local_preprocessor_path)

    bucket_name = model_output_path.replace("gs://", "").split("/")[0]
    blob_name = "/".join(model_output_path.replace("gs://", "").split("/")[1:])
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_model_path)

    # Also upload the preprocessor
    preprocessor_blob_name = "/".join(model_output_path.replace("gs://", "").split("/")[1:]).replace("pca_random_forest_model.joblib", "pca_preprocessor.joblib")
    preprocessor_blob = bucket.blob(preprocessor_blob_name)
    preprocessor_blob.upload_from_filename(local_preprocessor_path)

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
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    # Create or reuse endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project,
        location=region,
    )
    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

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
    data_path: str = "gs://biology-predict-bucket/Liver RNA Data.csv",
    model_output_path: str = "gs://biology-predict-bucket/pca_random_forest_model.joblib",
    model_display_name: str = "tcga_rna_model",
    endpoint_display_name: str = "tcga_rna_endpoint",
):
    train_task = train_model(
        data_path=data_path,
        model_output_path=model_output_path,
    )

    upload_task = upload_model(
        project=PROJECT,
        region=REGION,
        image_uri=IMAGE,
        model_display_name=model_display_name,
    ).after(train_task)

    deploy_model(
        project=PROJECT,
        region=REGION,
        model_resource_name=upload_task.output,
        endpoint_display_name=endpoint_display_name,
    ).after(upload_task)



if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(tcga_pipeline, "pipeline.yaml")

    aip.init(project=PROJECT, location=REGION)
    job = aip.PipelineJob(
        display_name="tcga-rna-pipeline",
        template_path="pipeline.yaml",
        enable_caching=True,
    )
    job.submit()