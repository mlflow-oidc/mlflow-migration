import os

version = os.environ.get("MLFLOW_MIGRATION", "2.1.0.dev0")

__version__ = version
