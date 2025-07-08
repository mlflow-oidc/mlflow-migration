import os
version = os.environ.get("MLFLOW_MIGRATION", "2.0.0.dev0")

__version__ = version
