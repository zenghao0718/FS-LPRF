"""Global learnable layer weights shared by training and all inference paths."""

METHOD_VERSION = "FS-LPRF-LW-v1"


def layer_weights(layer_logits):
    if layer_logits.shape != (25,):
        raise ValueError("FS-LPRF-LW requires exactly 25 layer logits")
    return layer_logits.softmax(dim=0)
