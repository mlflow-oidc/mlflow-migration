from importlib.metadata import version, PackageNotFoundError

pkg = "mlflow_migration"


def get_version():
    try:
        return version(pkg)
    except PackageNotFoundError:
        return ""
