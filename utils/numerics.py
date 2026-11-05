"""Explicit FP16 diagnostics; the formal cache default remains FP32."""
import torch


def tensor_errors(reference, actual):
    reference, actual = reference.detach().double().flatten(), actual.detach().double().flatten()
    delta = actual - reference
    scale = reference.norm().clamp_min(torch.finfo(torch.float64).eps)
    denominator = (reference.norm() * actual.norm()).clamp_min(torch.finfo(torch.float64).eps)
    return dict(max_absolute_error=float(delta.abs().max()), mean_absolute_error=float(delta.abs().mean()),
                relative_l2_error=float(delta.norm() / scale),
                cosine_similarity=float(torch.dot(reference, actual) / denominator))
