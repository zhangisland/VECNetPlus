# losses/loss_registry.py

LOSS_REGISTRY = {}

def register_loss(name):
    def wrapper(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return wrapper

def build_loss(name, **kwargs):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"[LOSS_REGISTRY] Unknown loss: {name}")
    return LOSS_REGISTRY[name](**kwargs)
