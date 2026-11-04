import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from engine.checkpoint import (checkpoint_sha256, load_checkpoint,
                                      resolve_checkpoint, save_checkpoint)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Backend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)


@pytest.fixture
def saved_node(config, tmp_path):
    model = Backend()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    run_dir = tmp_path / "run"
    path = run_dir / "checkpoints" / "step_0800.pt"
    args = dict(path=path, model=model, optimizer=optimizer, scheduler=scheduler,
                config=config, step=800, dev_metrics=None, best_eer=float("inf"),
                manifest_hash=digest("manifest"), train_fingerprint=digest("train"),
                run_identity=digest("run"))
    save_checkpoint(**args)
    index = dict(version=1, checkpoint=path.name, step=800, mean_eer=0.25,
                 checkpoint_sha256=checkpoint_sha256(path), run_identity=args["run_identity"],
                 dev_manifest_sha256=args["manifest_hash"], train_fingerprint=args["train_fingerprint"])
    (path.parent / "best_checkpoint.json").write_text(json.dumps(index), encoding="utf-8")
    return run_dir, path, args, index


def write_index(run_dir, index):
    (run_dir / "checkpoints" / "best_checkpoint.json").write_text(json.dumps(index), encoding="utf-8")


def test_node_save_safe_roundtrip_and_best_resolution(saved_node):
    run_dir, path, args, index = saved_node
    assert resolve_checkpoint("best", run_dir) == path.resolve()
    assert resolve_checkpoint(path, run_dir) == path.resolve()
    payload = load_checkpoint(path, args["model"], args["optimizer"], args["scheduler"],
                              restore_random=True, expected_run_identity=args["run_identity"])
    assert payload["step"] == 800 and payload["run_identity"] == index["run_identity"]
    assert not any(key.startswith("frontend_model.") for key in payload["model"])
    assert not list(path.parent.glob("*.tmp"))


def test_existing_node_is_never_overwritten(saved_node):
    _, path, args, _ = saved_node
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_checkpoint(**args)
    assert path.read_bytes() == original


def test_identity_mismatch_rejected_before_model_mutation(saved_node):
    _, path, args, _ = saved_node
    model = Backend()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, model, expected_run_identity=digest("other run"))
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


@pytest.mark.parametrize("reference", ["../step_0800.pt", "..\\step_0800.pt", "/tmp/step_0800.pt",
                                      "C:/elsewhere/step_0800.pt", "nested/step_0800.pt", "best.pt"])
def test_best_index_rejects_path_escape_and_non_node_names(saved_node, reference):
    run_dir, _, _, index = saved_node
    index["checkpoint"] = reference
    write_index(run_dir, index)
    with pytest.raises(ValueError, match="basename"):
        resolve_checkpoint("best", run_dir)


@pytest.mark.parametrize("field,value", [("step", 1600), ("step", True),
                                        ("mean_eer", float("nan")), ("mean_eer", -0.1),
                                        ("mean_eer", 1.1), ("version", 2)])
def test_best_index_rejects_step_and_metric_mismatch(saved_node, field, value):
    run_dir, _, _, index = saved_node
    index[field] = value
    write_index(run_dir, index)
    with pytest.raises(ValueError):
        resolve_checkpoint("best", run_dir)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "run_identity", "dev_manifest_sha256", "train_fingerprint"])
def test_best_index_checks_checkpoint_hash_and_bound_metadata(saved_node, field):
    run_dir, _, _, index = saved_node
    index[field] = digest("mismatch")
    write_index(run_dir, index)
    with pytest.raises(ValueError, match="SHA-256|match"):
        resolve_checkpoint("best", run_dir)


def test_best_index_rejects_checkpoint_payload_step_change(saved_node):
    run_dir, path, _, index = saved_node
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["step"] = 1600
    torch.save(payload, path)
    index["checkpoint_sha256"] = checkpoint_sha256(path)
    write_index(run_dir, index)
    with pytest.raises(ValueError, match="step"):
        resolve_checkpoint("best", run_dir)


def test_partial_save_does_not_expose_node_or_leave_temp(saved_node, tmp_path, monkeypatch):
    _, _, args, _ = saved_node
    path = tmp_path / "checkpoints" / "step_1600.pt"
    def fail_save(payload, handle):
        handle.write(b"interrupted")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk failure"):
        save_checkpoint(**dict(args, path=path, step=1600))
    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_publication_race_cannot_replace_existing_node(saved_node, tmp_path, monkeypatch):
    import os
    _, _, args, _ = saved_node
    path = tmp_path / "checkpoints" / "step_1600.pt"
    original_link = os.link
    def raced_link(source, destination):
        Path(destination).write_bytes(b"another completed node")
        return original_link(source, destination)
    monkeypatch.setattr(os, "link", raced_link)
    with pytest.raises(FileExistsError):
        save_checkpoint(**dict(args, path=path, step=1600))
    assert path.read_bytes() == b"another completed node"
    assert not list(path.parent.glob("*.tmp"))


def test_best_resolution_requires_index_and_real_node(saved_node):
    run_dir, path, _, index = saved_node
    path.unlink()
    with pytest.raises(FileNotFoundError, match="node does not exist"):
        resolve_checkpoint("best", run_dir)
    (run_dir / "checkpoints" / "best_checkpoint.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("best", run_dir)


def test_test_cli_accepts_best_without_changing_evaluation_route(saved_node, config, monkeypatch):
    import importlib.util
    import sys
    run_dir, path, args, _ = saved_node
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_cli_test", scripts / "evaluate.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = copy.deepcopy(config)
    cfg["runtime"]["output_dir"] = str(run_dir)
    (run_dir / "evaluation").mkdir()
    dataset, manifest = object(), {"synthetic": True}
    monkeypatch.setattr(cli, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(cli, "SpeechFakeDataset", Mock(return_value=dataset))
    monkeypatch.setattr(cli, "load_or_create_manifest", Mock(return_value=manifest))
    monkeypatch.setattr(cli, "FSLPRF", Mock(return_value=args["model"]))
    load = Mock(wraps=load_checkpoint)
    monkeypatch.setattr(cli, "load_checkpoint", load)
    evaluation = Mock(return_value={"mean_eer": 0.25})
    monkeypatch.setattr(cli, "evaluate", evaluation)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", "best", "--shot", "10"])
    cli.main()
    load.assert_called_once_with(path.resolve(), args["model"])
    evaluation.assert_called_once_with(args["model"], dataset, manifest, cfg, run_dir / "evaluation", "test")
    report = json.loads((run_dir / "evaluation" / "test_run.json").read_text(encoding="utf-8"))
    assert report["checkpoint"] == str(path.resolve()) and report["summary"] == {"mean_eer": 0.25}
