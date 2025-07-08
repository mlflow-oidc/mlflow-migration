# MLflow Migration

The MLflow Migration package provides tools to copy MLflow objects (runs, experiments or registered models) from one MLflow tracking server (Databricks workspace) to another.
Using the MLflow REST API, the tools export MLflow objects to an intermediate directory and then import them into the target tracking server.

# Reference notes

This is a technical fork of the original [MLflow Export Import](https://github.com/mlflow/mlflow-export-import) released under a new name and intended for publication on PyPI.

# Documentation

Documentation published at [https://mlflow-oidc.github.io/mlflow-migration/]
Documentation source is available in the [docs](docs) directory.

# Installation

Install the package using pip:

```bash
pip install mlflow-migration
```


# License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
