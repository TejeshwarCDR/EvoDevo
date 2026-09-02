from evodevo.encoders.base import Encoder
from evodevo.encoders.direct import DirectEncoder
from evodevo.encoders.nca import NCAEncoder
from evodevo.encoders.registry import get_encoder, list_encoders, register_encoder
from evodevo.encoders.rule_based import RuleBasedEncoder

# Explicit registration -- never dynamic filesystem scanning, so that a
# broken or misconfigured encoder package fails loudly here rather than
# being silently skipped.
register_encoder("direct", DirectEncoder)
register_encoder("nca", NCAEncoder)
register_encoder("rule_based", RuleBasedEncoder)

__all__ = [
    "Encoder",
    "DirectEncoder",
    "NCAEncoder",
    "RuleBasedEncoder",
    "get_encoder",
    "list_encoders",
    "register_encoder",
]
