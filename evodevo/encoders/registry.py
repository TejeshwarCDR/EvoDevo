from evodevo.encoders.base import Encoder
from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import NCAEncoder

_REGISTRY = {
    "direct": DirectEncoder,
    "nca": NCAEncoder,
}


def get_encoder(name: str, **kwargs) -> Encoder:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown encoder '{name}'; available: {sorted(_REGISTRY)}")
    return cls(**kwargs)
