import yaml

class Dict2Obj:
    """Recursively turn dict into object with dot notation access."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = Dict2Obj(v)
            setattr(self, k, v)
    def __getitem__(self, key):
        return getattr(self, key)
    #def __getattr__(self, name):
    #    return getattr(self, name)

    def load_config(path="config.yaml"):
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return Dict2Obj(cfg_dict)
