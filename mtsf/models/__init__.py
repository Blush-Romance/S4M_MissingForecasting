from .baselines import DLinear, PatchTST, iTransformer
from .crib_wrap import CRIB
from .mars import MARS

REGISTRY = {
    "crib": CRIB,
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "itransformer": iTransformer,
    "mars": MARS,
}


def build_model(cfg):
    return REGISTRY[cfg.model](cfg)
