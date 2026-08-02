import os
import yaml
from types import SimpleNamespace


class AttrNS(SimpleNamespace):
    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, AttrNS):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


class NestedConfig(AttrNS):
    """Hydra-like nested config with back-compat attribute shims."""
    def __getattr__(self, name):
        # Back-compat shims for legacy flat access
        if name == "model_name":
            return getattr(self.model, "name", None)
        if name == "model_class":
            return getattr(self.model, "class_name", None)
        if name == "model_pt":
            return getattr(self.model, "pt", None)
        if name == "inner_params":
            return getattr(self.model, "inner_params", None)
        if name == "edit_lr":
            return getattr(self.editor, "edit_lr", None)
        if name == "task":
            return getattr(self.experiment, "task", None)
        if name == "dataset_name":
            return getattr(self.experiment, "dataset_name", None)
        return super().__getattr__(name)


def load_yaml(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def to_ns(obj):
    if isinstance(obj, dict):
        return AttrNS(**{k: to_ns(v) for k, v in obj.items()})
    return obj
