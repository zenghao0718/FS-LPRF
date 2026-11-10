import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from data.audio_benchmark import BenchmarkManifestDataset
from data.cache import make_cache_identity
from data.cache_pipeline import (build_dataset_cache, open_train_dataset, bind_evaluation_caches,
                                 cache_bindings, evaluation_view, view_tokens)
from data.deletion_audit import audit_source_deletion
from data.loaders import ReusableAudioLoader
from data.randomness import sample_rng
from data.speechfake_dataset import SpeechFakeDataset
from data.waveform_cache import open_waveform_cache
from test_cache_integration import _cli


def write_manifests(root, paths):
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = paths[split]
        (root/f"{split}.tsv").write_text("".join(f"{split}_{i}"+chr(9)+str(path)+chr(10) for i,path in enumerate(rows)),encoding="utf-8")
        (root/f"{split}.txt").write_text("".join(chr(9).join((f"{split}_{i}","fixture","speaker","real","bonafide"))+chr(10)
                                                         for i in range(len(rows))),encoding="utf-8")


@pytest.fixture
def wave_fixture(config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["data"].update(train_cache_mode="waveform",dev_cache_mode="waveform",test_cache_mode="none",
                       do_augment=False,data_path=str(tmp_path/"manifests"))
    cfg["cache"].update(root=str(tmp_path/"cache"),shard_size=2)
    cfg["runtime"].update(output_dir=str(tmp_path/"run"),pin_memory=False)
    cfg["train"]["audio_microbatch_size"]=2
    paths=[]
    for i in range(4):
        path=tmp_path/f"source_{i}.wav"
        time=np.arange(128000,dtype=np.float32)/16000
        sf.write(path,(.04+.01*time)*np.sin(2*np.pi*((130+i*13)*time+11*time*time)),16000,subtype="FLOAT")
        paths.append(path)
    write_manifests(Path(cfg["data"]["data_path"]),dict(train=paths[:3],dev=paths[1:3],test=[paths[2],paths[3]]))
    return cfg, paths


def datasets(cfg):
    return {split:BenchmarkManifestDataset(cfg["data"],split) for split in ("train","dev","test")}


def build(cfg, split):
    ds=BenchmarkManifestDataset(cfg["data"],split)
    identity=make_cache_identity(ds,cfg,"waveform")
    return ds,build_dataset_cache(ds,cfg,identity)


@pytest.mark.parametrize("split",["train","dev"])
def test_waveform_cache_matches_raw_full_and_seeded_crop(wave_fixture,split):
    cfg,_=wave_fixture
    ds,reader=build(cfg,split)
    cached=BenchmarkManifestDataset(cfg["data"],split).use_cache(reader,"waveform")
    for index,uid in enumerate(ds.uttids):
        expected=ds.load_preprocessed_waveform(uid)
        assert expected.shape==(1,128000) and expected.dtype==torch.float32
        assert torch.equal(reader.get(uid),expected)
        with sample_rng(42):
            raw=ds[index]["feats"]
        with sample_rng(42):
            actual=cached[index]["feats"]
        with sample_rng(71):
            changed=cached[index]["feats"]
        assert torch.equal(actual,raw) and not torch.equal(actual,changed)


@pytest.mark.parametrize("workers",[0,2])
def test_train_dev_reinitialize_without_sources_or_source_hash_io(wave_fixture,monkeypatch,workers):
    import data.cache_pipeline as pipeline
    import data.cache as storage
    cfg,paths=wave_fixture
    write_manifests(Path(cfg["data"]["data_path"]),
                    dict(train=paths[:3],dev=paths[1:3],test=[paths[3]]))
    for split in ("train","dev"):
        build(cfg,split)
    expected={}
    for split in ("train","dev"):
        ds=BenchmarkManifestDataset(cfg["data"],split)
        with ReusableAudioLoader(ds,cfg,training=False,seed=42) as loader:
            expected[split]=torch.cat([row["feats"] for row in loader.for_indices(view_tokens(ds,ds.uttids,42,"full"))])
    root=paths[0].parent.resolve()
    # Only move fixture-owned copies. The user's Real directory is never touched.
    for path in paths[:3]:
        destination=path.with_suffix(".hidden")
        assert path.resolve().is_relative_to(root) and destination.resolve().is_relative_to(root)
        path.rename(destination)
    def forbidden(*args,**kwargs):
        raise AssertionError("Cache-only runtime must not inspect source audio")
    monkeypatch.setattr(storage,"source_digest",forbidden)
    monkeypatch.setattr(pipeline,"source_digest",forbidden)
    monkeypatch.setattr(pipeline,"SpeechFakeDataset",BenchmarkManifestDataset)
    cfg["runtime"]["eval_num_workers"]=workers
    train=open_train_dataset(cfg)
    dev=BenchmarkManifestDataset(cfg["data"],"dev")
    readers=bind_evaluation_caches(dev,cfg,{})
    assert len(readers)==1 and not dev.do_augment
    bindings,metadata=cache_bindings(train,dev)
    assert {row["split"] for row in bindings["caches"]}=={"train","dev"}
    assert len(metadata)==2 and "dev_source_sha256" not in bindings
    for split,ds in (("train",train),("dev",dev)):
        view=evaluation_view(ds,cfg,"full",42)
        with ReusableAudioLoader(view,cfg,training=False,seed=42) as loader:
            actual=torch.cat([row["feats"] for row in loader.for_indices(view_tokens(view,view.uttids,42,"full"))])
        assert torch.equal(actual,expected[split])
    assert BenchmarkManifestDataset(cfg["data"],"test")[0]["feats"].shape==(1,64000)


def test_test_raw_is_independent_of_all_cache_storage(wave_fixture,monkeypatch):
    import data.cache_pipeline as pipeline
    cfg,_=wave_fixture
    ds=BenchmarkManifestDataset(cfg["data"],"test")
    def forbidden(*args,**kwargs):
        raise AssertionError("Raw Test must not access waveform cache storage")
    monkeypatch.setattr(pipeline,"open_waveform_cache",forbidden)
    assert bind_evaluation_caches(ds,cfg,{})==[]
    view=evaluation_view(ds,cfg,"full",42)
    assert view.cache_mode=="none"
    with ReusableAudioLoader(view,cfg,training=False,seed=42) as loader:
        rows=list(loader.for_indices(view_tokens(view,view.uttids,42,"full")))
    assert sum(len(row["uttids"]) for row in rows)==len(ds)
    assert all(row["feats"].shape[1:]==(1,64000) for row in rows)


@pytest.mark.parametrize("mode",["waveform","xlsr_feature"])
def test_test_cache_creation_and_binding_are_rejected(wave_fixture,mode):
    cfg,_=wave_fixture
    ds=BenchmarkManifestDataset(cfg["data"],"test")
    with pytest.raises(ValueError,match="Test caches"):
        make_cache_identity(ds,cfg,mode,seed=42)
    cfg["data"]["test_cache_mode"]=mode
    with pytest.raises(ValueError,match="original WAVs"):
        bind_evaluation_caches(ds,cfg,{})


@pytest.mark.parametrize("change",["manifest","labels","preprocessing","schema","rng"])
def test_runtime_rejects_changed_metadata_without_raw_fallback(wave_fixture,monkeypatch,change):
    import data.cache as storage
    cfg,_=wave_fixture
    build(cfg,"dev")
    root=Path(cfg["data"]["data_path"])
    if change=="manifest":
        path=root/"dev.tsv"
        path.write_text(path.read_text(encoding="utf-8").replace("source_1.wav","different.wav"),encoding="utf-8")
    elif change=="labels":
        path=root/"dev.txt"
        path.write_text(path.read_text(encoding="utf-8").replace("speaker","different-speaker"),encoding="utf-8")
    elif change=="preprocessing":
        monkeypatch.setattr(storage,"PREPROCESSING",dict(storage.PREPROCESSING,target_rms_db=-19))
    elif change=="schema":
        cfg["cache"]["schema_version"]=2
    else:
        monkeypatch.setattr(storage,"RNG_SCHEMA_VERSION",2)
    ds=BenchmarkManifestDataset(cfg["data"],"dev")
    monkeypatch.setattr(storage,"source_digest",lambda *a,**k:pytest.fail("Must not inspect raw audio"))
    with pytest.raises(ValueError,match="Required dev Waveform Cache"):
        open_waveform_cache(ds,cfg)


def test_missing_or_corrupt_cache_never_falls_back_to_existing_wavs(wave_fixture):
    cfg,_=wave_fixture
    ds=BenchmarkManifestDataset(cfg["data"],"train")
    with pytest.raises(ValueError,match="No WAV fallback"):
        open_waveform_cache(ds,cfg)
    _,reader=build(cfg,"train")
    shard=reader.path/reader.metadata["shards"][0]["file"]
    raw=bytearray(shard.read_bytes())
    shard.write_bytes(raw[:-4])
    with pytest.raises(ValueError,match="size/record"):
        open_waveform_cache(ds,cfg)


def test_runtime_rejects_invalid_historical_source_sha_and_unsafe_reference(wave_fixture):
    cfg,_=wave_fixture
    ds,reader=build(cfg,"dev")
    path=reader.path/"metadata.json"
    metadata=json.loads(path.read_text(encoding="utf-8"))
    metadata["identity"]["source_audio_sha256"]="not-a-digest"
    path.write_text(json.dumps(metadata),encoding="utf-8")
    with pytest.raises(ValueError,match="recorded source"):
        open_waveform_cache(ds,cfg)
    reference=next((reader.path.parent/"references").glob("*.json"))
    saved=json.loads(reference.read_text(encoding="utf-8"))
    saved["cache_fingerprint"]="../outside"
    reference.write_text(json.dumps(saved),encoding="utf-8")
    with pytest.raises(ValueError,match="reference"):
        open_waveform_cache(ds,cfg)


def test_old_unambiguous_train_cache_works_without_reference_or_wav(wave_fixture):
    cfg,paths=wave_fixture
    ds,reader=build(cfg,"train")
    reference=next((reader.path.parent/"references").glob("*.json"))
    assert reference.resolve().is_relative_to(Path(cfg["cache"]["root"]).resolve())
    reference.unlink()
    for path in paths[:3]:
        destination=path.with_suffix(".hidden")
        assert destination.resolve().is_relative_to(path.parent.resolve())
        path.rename(destination)
    assert open_waveform_cache(BenchmarkManifestDataset(cfg["data"],"train"),cfg).identity==reader.identity


def test_source_deletion_audit_protects_test_and_only_suggests_cached_sources(wave_fixture):
    cfg,paths=wave_fixture
    for split in ("train","dev"):
        build(cfg,split)
    result=audit_source_deletion(cfg,datasets(cfg))
    assert result["status"]=="passed" and not result["deletion_performed"]
    eligible={row["path"] for row in result["deletable_files"]}
    assert eligible=={str(path) for path in paths[:2]}
    assert all(path.is_file() for path in paths)
    protected={row["path"]:row for row in result["retained_files"]}
    assert "referenced_by_Test_manifest" in protected[str(paths[2])]["reasons"]
    assert str(paths[3]) in protected
    assert protected[str(paths[3])]["bytes"]==paths[3].stat().st_size
    assert len([row for row in result["retained_essentials"] if row["reason"]=="manifest_and_labels"])==6


@pytest.mark.parametrize("failure",["cache_missing","cache_corrupt","source_changed","test_missing","special_read"])
def test_deletion_audit_blocks_on_incomplete_or_changed_inputs(wave_fixture,failure):
    cfg,paths=wave_fixture
    _,train=build(cfg,"train")
    if failure!="cache_missing":
        build(cfg,"dev")
    if failure=="cache_corrupt":
        shard=train.path/train.metadata["shards"][0]["file"]
        shard.write_bytes(b"corrupt")
    elif failure=="source_changed":
        paths[0].write_bytes(paths[0].read_bytes()+b"changed source")
    elif failure=="test_missing":
        destination=paths[3].with_suffix(".hidden")
        assert destination.resolve().is_relative_to(paths[3].parent.resolve())
        paths[3].rename(destination)
    ds=datasets(cfg)
    if failure=="special_read":
        ds["train"].uttid2info[ds["train"].uttids[0]].update(st=0,dur=8)
    result=audit_source_deletion(cfg,ds)
    assert result["status"]=="blocked" and result["errors"] and result["deletable_files"]==[]
    assert not result["deletion_performed"]


def test_build_check_operates_after_fixture_wavs_are_moved(wave_fixture,monkeypatch,capsys):
    import data.cache as storage
    import sys
    cfg,paths=wave_fixture
    build(cfg,"dev")
    for path in paths[1:3]:
        target=path.with_suffix(".hidden")
        assert target.resolve().is_relative_to(path.parent.resolve())
        path.rename(target)
    cli=_cli("build_cache",monkeypatch)
    monkeypatch.setattr(cli,"load_config",lambda *a:cfg)
    monkeypatch.setattr(cli,"SpeechFakeDataset",BenchmarkManifestDataset)
    monkeypatch.setattr(storage,"source_digest",lambda *a,**k:pytest.fail("No source SHA I/O during --check"))
    capsys.readouterr()
    monkeypatch.setattr(sys,"argv",["build_cache.py","--split","dev","--mode","waveform","--check"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"]=="verified"


def test_benchmark_fixture_does_not_relax_formal_32_class_check(wave_fixture):
    cfg,_=wave_fixture
    assert BenchmarkManifestDataset(cfg["data"],"train").fake_classes==[]
    with pytest.raises(ValueError,match="32 distinct Fake"):
        SpeechFakeDataset(cfg["data"],"train")


def test_full_training_step_and_dev_route_after_synthetic_source_removal(dataset,config,tmp_path,monkeypatch):
    from data.episode_sampler import EpisodeSampler
    from data.evaluation_manifest import load_or_create_manifest,load_or_create_light_manifest
    from data.cache_pipeline import dev_manifests
    import data.cache_pipeline as pipeline
    import data.cache as storage
    from engine.evaluator import evaluate
    from engine.trainer import make_optimizer,optimize_step
    from test_cache_storage import cache_config
    from test_engine import SyntheticBackend
    # Reuse the existing synthetic 33-class unit fixture, never the user's Real data.
    cfg=cache_config(dataset,config,tmp_path)
    cfg["data"].update(train_cache_mode="waveform",dev_cache_mode="waveform")
    path=Path(dataset.uttid2path[dataset.uttids[0]])
    signal=np.sin(np.arange(1024,dtype=np.float32)*.137)*.1
    sf.write(path,signal,16000,subtype="FLOAT")
    for split in ("train","dev"):
        ds=SpeechFakeDataset(cfg["data"],split,train_mode=False)
        build_dataset_cache(ds,cfg,make_cache_identity(ds,cfg,"waveform"))
    moved=path.with_suffix(".hidden")
    assert path.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
    path.rename(moved)
    def forbidden(*args,**kwargs):
        raise AssertionError("Training/Dev must not inspect removed source WAVs")
    monkeypatch.setattr(storage,"source_digest",forbidden)
    monkeypatch.setattr(pipeline,"source_digest",forbidden)
    train=open_train_dataset(cfg)
    dev=SpeechFakeDataset(cfg["data"],"dev")
    manifests=dev_manifests(dev,cfg)
    bind_evaluation_caches(dev,cfg,manifests)
    bindings,_=cache_bindings(train,dev)
    assert len(bindings["caches"])==2
    class WaveBackend(SyntheticBackend):
        def forward(self,source):
            return super().forward(dict(feats=source["feats"][:,:,:9]))
    model=WaveBackend()
    optimizer,_=make_optimizer(model,cfg)
    episodes=EpisodeSampler(train.class_indices).sample_step()
    loss,norm,diagnostics=optimize_step(model,train,episodes,cfg,optimizer,1)
    assert np.isfinite(loss) and np.isfinite(norm) and len(diagnostics)==8
    result=evaluate(model,dev,manifests["light"][0],cfg,tmp_path/"reports","dev",200,mode="light")
    assert np.isfinite(result["mean_eer"]) and len(result["per_class_summary"])==32


def test_audit_retains_hardlinks_referenced_by_test(wave_fixture):
    import os
    cfg,paths=wave_fixture
    linked=paths[0].with_name("test-hardlink.wav")
    assert linked.resolve().is_relative_to(paths[0].parent.resolve())
    os.link(paths[0],linked)
    write_manifests(Path(cfg["data"]["data_path"]),
                    dict(train=paths[:3],dev=paths[1:3],test=[linked,paths[3]]))
    for split in ("train","dev"):
        build(cfg,split)
    report=audit_source_deletion(cfg,datasets(cfg))
    assert report["status"]=="passed"
    retained={row["path"]:row for row in report["retained_files"]}
    assert "same_physical_file_as_Test" in retained[str(paths[0])]["reasons"]
    assert str(paths[0]) not in {row["path"] for row in report["deletable_files"]}
    assert paths[0].is_file() and linked.is_file()


def test_pure_cache_smoke_cli_does_not_initialize_test(wave_fixture,monkeypatch,capsys):
    import sys
    cfg,paths=wave_fixture
    for split in ("train","dev"):
        build(cfg,split)
    for path in paths[:3]:
        moved=path.with_suffix(".hidden")
        assert moved.resolve().is_relative_to(path.parent.resolve())
        path.rename(moved)
    cli=_cli("check_cache_runtime",monkeypatch)
    monkeypatch.setattr(cli,"load_config",lambda *a:cfg)
    monkeypatch.setattr(cli,"open_train_dataset",lambda config:BenchmarkManifestDataset(config["data"],"train").use_cache(
        open_waveform_cache(BenchmarkManifestDataset(config["data"],"train"),config),"waveform"))
    def create(options,split,**kwargs):
        assert split=="dev"
        return BenchmarkManifestDataset(options,split)
    monkeypatch.setattr(cli,"SpeechFakeDataset",create)
    capsys.readouterr()
    monkeypatch.setattr(sys,"argv",["check_cache_runtime.py","--batches","1"])
    cli.main()
    result=json.loads(capsys.readouterr().out)
    assert result["status"]=="passed" and not result["test_initialized"]
    assert result["optimizer_updates"]==0 and not result["hidden_states_saved"]


def test_config_rejects_test_feature_cache():
    from utils.config import load_config
    with pytest.raises(ValueError,match="test_cache_mode"):
        load_config("configs/speechfake.yaml",["data.test_cache_mode=xlsr_feature"])


def test_generic_raw_preprocessing_has_no_speechfake_class_or_cache_dependency(wave_fixture,monkeypatch):
    from argparse import Namespace
    from data.audio import GeneralFAD
    import data.cache_pipeline as pipeline
    cfg,_=wave_fixture
    raw=GeneralFAD(Namespace(**cfg["data"]),"test",train_mode=False)
    assert not hasattr(raw,"fake_classes") and not hasattr(raw,"cache_mode")
    monkeypatch.setattr(pipeline,"open_waveform_cache",lambda *a,**k:pytest.fail("Raw adapter must not use caches"))
    view=evaluation_view(raw,cfg,"full",42)
    with ReusableAudioLoader(view,cfg,training=False,seed=42) as loader:
        rows=list(loader.for_indices(view_tokens(view,view.uttids,42,"full")))
    assert [uid for row in rows for uid in row["uttids"]]==view.uttids
    assert all(row["feats"].shape[1:]==(1,64000) for row in rows)


def test_cache_reader_close_releases_mapped_shards(wave_fixture):
    cfg,_=wave_fixture
    ds,reader=build(cfg,"dev")
    value=reader.get(ds.uttids[0])
    expected=value.clone()
    assert reader._maps
    reader.close()
    assert not reader._maps and torch.equal(value,expected)
    # Public tensors are copies: their contents survive explicit mapping release.
    with reader:
        assert torch.equal(reader.get(ds.uttids[0]),expected)
    assert not reader._maps


def test_removed_train_source_still_allows_dynamic_rir_and_rawboost(dataset,config,rir_root,tmp_path,monkeypatch):
    from test_cache_storage import cache_config
    cfg=cache_config(dataset,config,tmp_path)
    cfg["data"].update(do_augment=True,rir_root=str(rir_root))
    base=SpeechFakeDataset(cfg["data"],"train",train_mode=False)
    uid=base.uttids[0]
    reader=build_dataset_cache(base,cfg,make_cache_identity(base,cfg,"waveform",uttids=[uid],smoke=True))
    raw=SpeechFakeDataset(cfg["data"],"train",train_mode=True)
    cached=SpeechFakeDataset(cfg["data"],"train",train_mode=True).use_cache(reader,"waveform")
    # Force both existing augmentation branches in this test only.
    for ds in (raw,cached):
        ds.augmentor.rir_prob=1.
        ds.augmentor.noise_prob=1.
    with sample_rng(42):
        expected=raw[0]["feats"]
    source=Path(base.uttid2path[uid])
    moved=source.with_suffix(".hidden")
    assert source.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
    source.rename(moved)
    original=sf.SoundFile
    reads=[]
    def guarded(path,*args,**kwargs):
        target=Path(path).resolve()
        assert target.is_relative_to(rir_root.resolve())
        reads.append(target)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(sf,"SoundFile",guarded)
    with sample_rng(42):
        actual=cached[0]["feats"]
    assert reads and torch.equal(actual,expected)
    reader.close()


@pytest.mark.parametrize("key",["train_cache_mode","dev_cache_mode"])
def test_official_config_cannot_enable_hidden_state_caching(key):
    from utils.config import load_config
    with pytest.raises(ValueError,match=key):
        load_config("configs/speechfake.yaml",[f"data.{key}=xlsr_feature"])


def test_official_builder_cannot_generate_hidden_state_cache(monkeypatch):
    import sys
    cli=_cli("build_cache",monkeypatch)
    monkeypatch.setattr(sys,"argv",["build_cache.py","--split","dev","--mode","xlsr_feature"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code==2
