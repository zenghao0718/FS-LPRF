"""Real-only fixture benchmarks for Train/Dev waveforms and raw Test.

No fake classes, formal split generation, model loading or hidden-state caching.
All sources are read-only; removal checks move only copies in an owned temp root.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

import _bootstrap
import benchmark_audio as base
import torch

from data.audio_benchmark import BenchmarkManifestDataset, scan_real_audio, waveform_digest
from data.cache import atomic_json, make_cache_identity
from data.cache_pipeline import build_dataset_cache, bind_evaluation_caches, cache_bindings, view_tokens
from data.deletion_audit import audit_source_deletion
from data.loaders import ReusableAudioLoader
from data.randomness import sample_rng, sample_seed
from data.waveform_cache import open_waveform_cache
from utils.config import common_parser, load_config


def write_fixture_manifests(root, paths, records):
    root.mkdir(parents=True,exist_ok=True)
    for split in ("train","dev","test"):
        (root/f"{split}.tsv").write_text("".join(chr(9).join((f"fixture-{split}-{index:05d}",str(path)))+chr(10)
                                                for index,path in enumerate(paths[split])),encoding="utf-8")
        (root/f"{split}.txt").write_text("".join(chr(9).join((f"fixture-{split}-{index:05d}",records[index]["source"],
                                                            "benchmark-only","real","bonafide"))+chr(10)
                                                for index in range(len(paths[split]))),encoding="utf-8")


def fixture_config(cfg, root, workers=0):
    local=copy.deepcopy(cfg)
    local["data"].update(data_path=str(root/"manifests"),do_augment=False,
                         train_cache_mode="waveform",dev_cache_mode="waveform",test_cache_mode="none")
    local["cache"]["root"]=str(root/"cache")
    local["runtime"].pop("num_workers",None)
    local["runtime"].update(train_num_workers=workers,eval_num_workers=workers)
    return local


def reference(ds):
    crops,full={},{}
    for index,uid in enumerate(ds.uttids):
        whole=ds.load_preprocessed_waveform(uid)
        full[uid]=waveform_digest(whole)
        with sample_rng(sample_seed("view",ds.split,"full",42,uid)):
            crops[uid]=waveform_digest(ds.sample_segment(whole))
    return crops,full


def verify_views(raw,cached,ids):
    changed=0
    for mode in ("light","full"):
        for seed in (17,42,71,100,123):
            for uid in ids:
                token=view_tokens(raw,[uid],seed,mode)[0]
                with sample_rng(token[1]):
                    expected=raw[raw.uttid2idx[uid]]["feats"]
                with sample_rng(token[1]):
                    actual=cached[cached.uttid2idx[uid]]["feats"]
                if not torch.equal(actual,expected):
                    raise AssertionError("Raw/cache deterministic view mismatch")
    for position,uid in enumerate(ids):
        views=[]
        for step in (1,2):
            seed=sample_seed("train",42,step,position,uid)
            with sample_rng(seed):
                expected=raw[raw.uttid2idx[uid]]["feats"]
            with sample_rng(seed):
                actual=cached[cached.uttid2idx[uid]]["feats"]
            if not torch.equal(actual,expected):
                raise AssertionError("Dynamic train crop mismatch")
            views.append(actual)
        changed+=not torch.equal(*views)
    return dict(files=len(ids),modes=["light","full"],seeds=[17,42,71,100,123],
                raw_cache_views_exact=True,dynamic_step_crop_exact=True,
                dynamic_step_changed_files=changed)


def cache_only_copy_check(cfg,records,output):
    selected=records[:min(16,len(records))]
    with tempfile.TemporaryDirectory(prefix=".owned_source_copies_",dir=output) as temporary:
        owned=Path(temporary).resolve()
        sources=owned/"train_dev_copies"
        test_sources=owned/"test_copies"
        sources.mkdir();test_sources.mkdir()
        train_paths,test_paths=[],[]
        for i,row in enumerate(selected):
            left=sources/f"{i:04d}.wav";right=test_sources/f"{i:04d}.wav"
            shutil.copy2(row["path"],left);shutil.copy2(row["path"],right)
            train_paths.append(left);test_paths.append(right)
        local=fixture_config(cfg,owned)
        mapping=dict(train=train_paths,dev=train_paths,test=test_paths)
        write_fixture_manifests(owned/"manifests",mapping,selected)
        for split in ("train","dev"):
            ds=BenchmarkManifestDataset(local["data"],split)
            build_dataset_cache(ds,local,make_cache_identity(ds,local,"waveform"))
        make_ds=lambda:{split:BenchmarkManifestDataset(local["data"],split) for split in ("train","dev","test")}
        audit=audit_source_deletion(local,make_ds())
        if audit["status"]!="passed" or len(audit["deletable_files"])!=len(selected):
            raise AssertionError("Owned-copy deletion audit failed")
        shared=dict(mapping,test=[train_paths[0],*test_paths[1:]])
        write_fixture_manifests(owned/"manifests",shared,selected)
        shared_audit=audit_source_deletion(local,make_ds())
        if str(train_paths[0]) in {row["path"] for row in shared_audit["deletable_files"]}:
            raise AssertionError("Audit suggested removing Test-shared source")
        write_fixture_manifests(owned/"manifests",mapping,selected)
        hidden=owned/"hidden_train_dev_copies"
        assert sources.resolve().is_relative_to(owned) and hidden.resolve().is_relative_to(owned)
        sources.rename(hidden)
        import data.cache as storage
        import data.cache_pipeline as pipeline
        calls=[]
        def forbidden(*args,**kwargs):
            calls.append(args)
            raise AssertionError("Cache-only runtime inspected removed source audio")
        loaded=[]
        with patch.object(storage,"source_digest",forbidden),patch.object(pipeline,"source_digest",forbidden):
            train=BenchmarkManifestDataset(local["data"],"train")
            train.use_cache(open_waveform_cache(train,local),"waveform")
            dev=BenchmarkManifestDataset(local["data"],"dev")
            bind_evaluation_caches(dev,local,{})
            bindings,_=cache_bindings(train,dev)
            for ds in (train,dev):
                with ReusableAudioLoader(ds,local,training=False,seed=42) as loader:
                    loaded.append(sum(len(batch["uttids"]) for batch in loader.for_indices(view_tokens(ds,ds.uttids,42,"full"))))
        test=BenchmarkManifestDataset(local["data"],"test")
        with ReusableAudioLoader(test,local,training=False,seed=42) as loader:
            test_count=sum(len(batch["uttids"]) for batch in loader.for_indices(view_tokens(test,test.uttids,42,"full")))
        train.cache_reader.close()
        dev.cache_reader.close()
        return dict(scope="owned copies and temporary Real-only manifests; no formal Fake classes",
                    files=len(selected),train_dev_loaded_without_originals=loaded,
                    historical_source_hash_calls=len(calls),cache_bindings=len(bindings["caches"]),
                    raw_test_loaded=test_count,test_copies_preserved=all(path.is_file() for path in test_paths),
                    deletion_audit_status=audit["status"],audit_suggested_files=len(audit["deletable_files"]),
                    audit_estimated_logical_bytes=audit["estimated_reclaimable_logical_bytes"],
                    test_shared_source_retained=True,deletion_performed=False)


def main():
    parser=common_parser("Real-only Train/Dev cache and raw Test engineering benchmarks")
    parser.add_argument("--audio-root",required=True)
    parser.add_argument("--limit",type=int,default=512)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--workers",default="0,2")
    parser.add_argument("--build-workers",default="0,2")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    if args.rounds<3:
        parser.error("Use at least three rounds: one warmup and two measured")
    cfg=load_config(args.config,args.set)
    torch.set_num_threads(cfg["runtime"]["worker_torch_num_threads"])
    output=Path(args.output_dir).resolve()
    project=Path(_bootstrap.ROOT).resolve()
    if output.is_relative_to(project) and (
            output==project or output.relative_to(project).parts[0] not in ("outputs",".tmp",".cache")):
        parser.error("Benchmark reports must be outside source directories")
    output.mkdir(parents=True,exist_ok=True)
    target=output/"results.json"
    if target.exists():
        parser.error("Existing reports are preserved; choose a new output directory")
    records,inventory=scan_real_audio(args.audio_root,args.limit)
    if not records:
        raise ValueError("No readable Real files")
    report=dict(scope="Real-only temporary identity fixtures; not formal SpeechFake splits or classification results",
                inventory=inventory,builds=[],reads=[],view_correctness={},pure_cache_check=None,
                workers=[int(x) for x in args.workers.split(",")],rounds=args.rounds)
    atomic_json(target,report)
    for build_workers in [int(x) for x in args.build_workers.split(",")]:
        runs=[]
        for repeat in range(args.rounds):
            with tempfile.TemporaryDirectory(prefix=".waveform_fixture_",dir=output) as temporary:
                local=fixture_config(cfg,Path(temporary),build_workers)
                paths=[Path(row["path"]) for row in records]
                write_fixture_manifests(Path(local["data"]["data_path"]),dict(train=paths,dev=paths,test=paths),records)
                readers={}
                row=dict(repeat=repeat,warmup=repeat==0)
                for split in ("train","dev"):
                    ds=BenchmarkManifestDataset(local["data"],split)
                    tick=time.perf_counter()
                    identity=make_cache_identity(ds,local,"waveform")
                    preparation=time.perf_counter()-tick
                    reader=build_dataset_cache(ds,local,identity,preparation_seconds=preparation)
                    row[split]=dict(total_seconds=time.perf_counter()-tick,identity_seconds=preparation,
                                    bytes=reader.stats()["bytes"],samples=len(ds),fingerprint=reader.metadata["fingerprint"])
                    readers[split]=reader
                if build_workers==int(args.build_workers.split(",")[0]) and repeat==args.rounds-1:
                    for split in ("train","dev","test"):
                        raw=BenchmarkManifestDataset(local["data"],split)
                        expected,full=reference(raw)
                        if split=="test":
                            ds=raw
                        else:
                            ds=BenchmarkManifestDataset(local["data"],split).use_cache(readers[split],"waveform")
                            for uid in ds.uttids:
                                if waveform_digest(ds.load_preprocessed_waveform(uid))!=full[uid]:
                                    raise AssertionError("Full FP32 waveform mismatch")
                            report["view_correctness"][split]=verify_views(raw,ds,ds.uttids[:min(32,len(ds))])
                        for count in report["workers"]:
                            result=base.benchmark_case(ds,local,expected,workers=count,persistent=True,
                                                       prefetch=cfg["runtime"]["prefetch_factor"],
                                                       strategy="reuse",rounds=args.rounds)
                            result["split"]=split
                            result["input"]="raw" if split=="test" else "waveform"
                            report["reads"].append(result)
                            atomic_json(target,report)
                            print(f"{split} {result['input']} workers={count}: {result['samples_per_second']:.2f}/s",flush=True)
                for reader in readers.values():
                    reader._maps.clear()
                runs.append(row)
                print(f"build workers={build_workers} round={repeat}: train={row['train']['total_seconds']:.3f}s dev={row['dev']['total_seconds']:.3f}s",flush=True)
        report["builds"].append(dict(workers=build_workers,rounds=runs,
                                     mean_train_seconds=sum(row["train"]["total_seconds"] for row in runs[1:])/(len(runs)-1),
                                     mean_dev_seconds=sum(row["dev"]["total_seconds"] for row in runs[1:])/(len(runs)-1)))
        atomic_json(target,report)
    report["pure_cache_check"]=cache_only_copy_check(cfg,records,output)
    report["complete"]=True
    atomic_json(target,report)
    print(f"Complete: {target}",flush=True)


if __name__=="__main__":
    main()
