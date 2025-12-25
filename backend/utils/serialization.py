import numpy as np
from dataclasses import is_dataclass, asdict

def sanitize_for_state(obj):
    """
    Convert objects into LangGraph + MsgPack safe types.
    """
    if obj is None:
        return None

    # NumPy scalars
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Dataclass
    if is_dataclass(obj):
        return sanitize_for_state(asdict(obj))

    # Dict
    if isinstance(obj, dict):
        return {k: sanitize_for_state(v) for k, v in obj.items()}

    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_state(v) for v in obj]

    # Primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback (never crash orchestration)
    return str(obj)
