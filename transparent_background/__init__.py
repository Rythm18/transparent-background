import importlib

__all__ = ["Remover", "console", "gui"]


def __getattr__(name):
    if name in {"Remover", "console"}:
        module = importlib.import_module(".Remover", __name__)
        return getattr(module, name)
    if name == "gui":
        module = importlib.import_module(".gui", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")