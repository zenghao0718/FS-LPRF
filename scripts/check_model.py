"""Real XLS-R CUDA smoke test; no dataset download or training experiment."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import sys
import numpy as np
import soundfile as sf
import torch
import _bootstrap
from utils.config import common_parser, load_config
from utils.provenance import UPSTREAM_COMMIT
from utils.seed import seed_all
from utils.numerics import tensor_errors
from models.fs_lprf_model import FSLPRF
from models.frontend_compat import LOAD_REPORT
from engine.trainer import make_optimizer
from engine.checkpoint import save_checkpoint, load_checkpoint

from data.audio import FADTrim


class SmokeAudio(FADTrim):
    """Temporary WAV through official audio processing and sample_segment, unchanged."""
    def _load_both(self, args):
        self.uttid2path = {"smoke": args.wav_path}
        self.uttid2info = {"smoke": dict(label=0, origin_ds="synthetic", speaker="smoke", attacker="real")}


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--report", default="outputs/local_smoke/model_report.json")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    seed_all(42)
    device = torch.device(cfg["runtime"]["device"])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = FSLPRF(cfg)
    model.train()
    assert not model.frontend_model.training
    assert all(not p.requires_grad for p in model.frontend_model.parameters())
    assert set(name for name, p in model.named_parameters() if p.requires_grad) == {
        "time_attn_mlp1", "bottleneck_proj.weight", "bottleneck_proj.bias", "layer_logits"}
    assert model.time_attn_mlp1.shape == (25, 1024, 4)
    assert len({model.time_attn_mlp1[i].data_ptr() for i in range(25)}) == 25
    assert isinstance(model.bottleneck_proj, torch.nn.Linear)
    assert model.bottleneck_proj.weight.shape == (256, 1024)
    observed = {}

    def inspect_hidden(module, inputs, output):
        hidden = [output["post_features"]] + [item[0] for item in output["layer_results"]]
        assert len(hidden) == 25
        assert all(not value.requires_grad for value in hidden)
        observed["hidden_shapes"] = [list(value.shape) for value in hidden]

    def inspect_pool(module, inputs):
        observed["attention_shape"] = list(inputs[0].shape)
        assert inputs[0].shape == (1, 25, 1024)

    h1 = model.frontend_model.model.register_forward_hook(inspect_hidden)
    h2 = model.bottleneck_proj.register_forward_pre_hook(inspect_pool)
    with tempfile.TemporaryDirectory() as temp:
        wav_path = Path(temp) / "smoke.wav"
        time = np.arange(64000, dtype=np.float32) / 16000
        wave = 0.1 * np.sin(2 * np.pi * (160 * time + 70 * time**2))
        sf.write(wav_path, wave, 16000, subtype="FLOAT")
        dataset = SmokeAudio(Namespace(data_path=temp, wav_path=str(wav_path), max_samples=64000,
                                      fix_len=True, trim_silence=True, do_augment=False), "dev", False)
        batch = dataset.collate_fn([dataset._getitem_impl(0)])
        assert batch["feats"].shape == (1, 1, 64000)
        output = model({"feats": batch["feats"]})["feats"]
        assert output.shape == (1, 25, 256) and torch.isfinite(output).all()
        with torch.no_grad():
            cached_states = model.extract_frontend_features(batch["feats"], to_cpu=True)
            cached_output = model.forward_from_frontend_features(cached_states)["feats"]
            fp16_output = model.forward_from_frontend_features(cached_states.half())["feats"]
            torch.testing.assert_close(output, cached_output, atol=1e-5, rtol=1e-5)
            observed["cache_fp32_backend"] = tensor_errors(output, cached_output)
            observed["cache_fp16_hidden"] = tensor_errors(cached_states, cached_states.half().float())
            observed["cache_fp16_backend"] = tensor_errors(output, fp16_output)
            observed["cache_fp32_diagnostic_loss"] = float(cached_output.square().mean())
            observed["cache_fp16_diagnostic_loss"] = float(fp16_output.square().mean())
            observed["cache_scope"] = "single synthetic WAV; real frozen frontend; diagnostic squared loss, no task metrics"
        # Diagnostic only: validate backend gradients without running an optimizer update.
        (output.square().mean(dim=(0, 2)) * model.layer_weights()).sum().backward()
        grads = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all()
                assert param.grad.abs().sum() > 0
                grads[name] = float(param.grad.norm())
            else:
                assert param.grad is None
        assert all(model.time_attn_mlp1.grad[i].abs().sum() > 0 for i in range(25))
        optimizer, scheduler = make_optimizer(model, cfg)
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        cp = Path(temp) / "smoke.pt"
        save_checkpoint(cp, model, optimizer, scheduler, cfg, 0, None, float("inf"), "synthetic", "synthetic")
        with torch.no_grad():
            model.bottleneck_proj.weight.add_(1)
        load_checkpoint(cp, model, optimizer, scheduler, restore_random=True)
        assert all(torch.equal(model.state_dict()[key], value) for key, value in saved.items())
        assert cp.stat().st_size < 5_000_000
        checkpoint_bytes = cp.stat().st_size
    h1.remove()
    h2.remove()
    report = dict(status="passed", scope="synthetic 4-second WAV; real official frontend; no experiment or optimizer update",
                  python_executable=sys.executable, python_prefix=sys.prefix, python_version=sys.version,
                  torch=torch.__version__, torchaudio=__import__("torchaudio").__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(model.device),
                  capability=list(torch.cuda.get_device_capability(model.device)), upstream_commit=UPSTREAM_COMMIT,
                  frontend_load=dict(LOAD_REPORT), **observed, output_shape=list(output.shape),
                  frontend_frozen=True, frontend_eval_after_train=True, gradient_norms=grads,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(model.device), backend_checkpoint_bytes=checkpoint_bytes)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
