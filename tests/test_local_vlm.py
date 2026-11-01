import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src.todesanzeigen.cli import build_parser, main
from src.todesanzeigen.field_evaluation import evaluate_comparison, score_records
from src.todesanzeigen.inference_config import InferenceConfig
from src.todesanzeigen.llm import MODEL_RESPONSE_COLUMNS
from src.todesanzeigen.local_backend import FirstTokenObserver
from src.todesanzeigen.local_vlm import benchmark_report, run_local, variant_lookup
from src.todesanzeigen.methods import LOCAL_VISION_METHOD
from src.todesanzeigen.review import _select_review_candidates
from src.todesanzeigen.storage import (
    active_extraction_for_variant, apply_migrations, connect, create_dataset_split,
    export_variant_csv, insert_extraction_output, save_ground_truth_label, sha256_file, upsert_document,
)
from src.todesanzeigen.variants import ExtractionVariant, load_variant_config


@pytest.fixture
def setup(tmp_path):
    db = tmp_path / "test.sqlite3"
    apply_migrations(db)
    image = tmp_path / "notice.jpg"
    image.write_bytes(b"fake-image")
    with connect(db) as c:
        doc = upsert_document(c, source_name="paper", filename_stem="notice", image_path=image)
        save_ground_truth_label(c, document_id=doc, label_set="gt-v1", fields={"name": "Muster"})
    variant = load_variant_config().variant("local_qwen3vl_fp16")
    return db, image, doc, variant


class FakeBackend:
    calls = 0
    loads = 0
    response = json.dumps({**dict.fromkeys(MODEL_RESPONSE_COLUMNS, ""), "nachname": "Muster"})

    def __init__(self, model, config):
        self.config = config

    def load(self):
        type(self).loads += 1
        return 17.0, {"gpu": "fake"}

    def infer(self, prompt, image):
        type(self).calls += 1
        assert "Todesanzeige" in prompt
        return self.response, {"generation_ms": 10.0, "preprocess_ms": 2.0,
                               "first_token_ms": 4.0, "decode_tokens_per_second": 100.0,
                               "input_tokens": 20, "output_tokens": 3,
                               "peak_allocated_bytes": 100, "peak_reserved_bytes": 200,
                               "output_limit_reached": False}


def test_config_identity_and_validation():
    config = InferenceConfig("a" * 40)
    assert config.identity_hash == InferenceConfig.from_dict(config.as_dict()).identity_hash
    assert replace(config, use_cache=False).identity_hash != config.identity_hash
    assert replace(config, weight_format="nf4").identity_hash != config.identity_hash
    for data in ({"revision": "main"}, {"revision": "a" * 40, "extra": True},
                 {"revision": "a" * 40, "weight_format": "bf16"},
                 {"revision": "a" * 40, "max_pixels": 0}):
        with pytest.raises(ValueError):
            InferenceConfig.from_dict(data)


def test_identity_migration_and_model_required(setup):
    db, _, doc, variant = setup
    with connect(db) as c:
        for config in (variant.inference, replace(variant.inference, use_cache=False)):
            v = replace(variant, inference=config)
            insert_extraction_output(c, document_id=doc, run_id=None, status="vision_processed", fields={}, **variant_lookup(v))
        with pytest.raises(sqlite3.IntegrityError):
            insert_extraction_output(c, document_id=doc, run_id=None, status="vision_processed", fields={}, **variant_lookup(variant))
        with pytest.raises(ValueError, match="model is required"):
            insert_extraction_output(c, document_id=doc, run_id=None, status="vision_processed", fields={}, method=LOCAL_VISION_METHOD)
        with pytest.raises(sqlite3.IntegrityError, match="model is required"):
            c.execute("INSERT INTO extraction_outputs(document_id,method,model,status) VALUES (?,?,?,?)",
                      (doc, LOCAL_VISION_METHOD, "", "failed"))
    assert apply_migrations(db) == []


def test_benchmark_repetitions_cache_and_history(setup):
    db, image, doc, variant = setup
    FakeBackend.calls = FakeBackend.loads = 0
    first = run_local(db_path=db, variant=variant, benchmark=True, backend_factory=FakeBackend)
    report = first["benchmark"]
    assert FakeBackend.calls == 4 and FakeBackend.loads == 1
    assert report["aggregates"]["generation_ms"] == {"count": 3, "mean": 10., "median": 10., "p95": 10.}
    assert report["completion_rate"] == 1
    assert len(report["samples"]) == 4
    assert report["run"]["load_ms"] == 17
    with connect(db) as c:
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 1
        old_id = active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))["id"]
    cached = run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    assert cached["cached"] == 1 and FakeBackend.calls == 4
    run_local(db_path=db, variant=variant, benchmark=True, repetitions=1, warmups=0, backend_factory=FakeBackend)
    with connect(db) as c:
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))["id"] != old_id
    quality = evaluate_comparison(db_path=db, run_id=first["run_id"], label_set="gt-v1")
    score = quality["results"]["run:" + first["run_id"]]
    assert score["fields"]["name"]["f1"] == 1
    assert score["prediction_output_ids"][str(doc)] == old_id
    assert benchmark_report(db, first["run_id"])["samples"] == report["samples"]
    image.write_bytes(b"changed")
    result = run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    assert result["cached"] == 0


def test_cache_settings_do_not_supersede_each_other(setup):
    db, _, doc, variant = setup
    second = replace(variant, inference=replace(variant.inference, use_cache=False))
    for v in (variant, second):
        run_local(db_path=db, variant=v, backend_factory=FakeBackend)
    with connect(db) as c:
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(second))


@pytest.mark.parametrize("response,valid", [("not json", False), ('{"nachname":"Muster"}', True)])
def test_invalid_samples_keep_timings_without_predictions(setup, response, valid):
    db, _, _, variant = setup
    backend = type("BadBackend", (FakeBackend,), {"response": response})
    result = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=backend)
    report = result["benchmark"]
    assert report["successful_samples"] == 0
    assert report["aggregates"]["generation_ms"]["count"] == 3
    assert report["validity"]["json_valid"]["count"] == (3 if valid else 0)
    with connect(db) as c:
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 0


def test_oom_stops_and_retains_samples(setup):
    db, _, _, variant = setup
    class OutOfMemoryError(RuntimeError):
        pass
    class OomBackend(FakeBackend):
        def infer(self, prompt, image):
            raise OutOfMemoryError("CUDA out of memory")
    result = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=OomBackend)
    assert result["status"] == "failed"
    assert result["benchmark"]["attempted_samples"] == 1
    assert result["benchmark"]["completion_rate"] == 0


def test_load_failure_is_durable(setup):
    db, _, _, variant = setup
    class BadLoad(FakeBackend):
        def load(self):
            raise RuntimeError("GPU disabled")
    result = run_local(db_path=db, variant=variant, benchmark=True, backend_factory=BadLoad)
    assert result["status"] == "failed"
    assert result["benchmark"]["run"]["error"] == "GPU disabled"


def test_missing_image_remains_failure(setup):
    db, image, _, variant = setup
    image.unlink()
    result = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=FakeBackend)
    assert result["benchmark"]["attempted_samples"] == 3
    assert result["benchmark"]["aggregates"]["total_ms"]["count"] == 0
    assert result["benchmark"]["completion_rate"] == 0


def test_scoring_absent_values_normalization_and_metadata():
    result = score_records({1: {"name": "Muster", "geburtsdatum": "1930-01-02"}},
                           {1: {"name": "Other", "vorname": "Extra", "geburtsdatum": "02.01.1930", "quelle": "different"}})
    assert result["fields"]["name"]["fp"] == 1
    assert result["fields"]["name"]["fn"] == 1
    assert result["fields"]["vorname"]["fp"] == 1
    assert result["fields"]["geburtsdatum"]["accuracy"] == 1
    assert "quelle" not in result["fields"]
    assert result["fields"]["beruf"]["f1"] is None
    missing = score_records({1: {}}, {})
    assert missing["missing_predictions"] == 1
    assert missing["exact_record_accuracy"] == 0
    assert score_records({}, {})["status"] == "no evaluable documents"


def test_comparison_modes_coverage_and_subsets(setup):
    db, _, doc, variant = setup
    run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    with connect(db) as c:
        create_dataset_split(c, name="split", strategy="test", assignments={doc: "validation"})
    score = evaluate_comparison(db_path=db, variant_aliases=[variant.alias], label_set="gt-v1", split_name="split")
    assert score["results"][variant.alias]["documents"] == 0
    score = evaluate_comparison(db_path=db, variant_aliases=[variant.alias], label_set="gt-v1", split_name="split", subset="validation")
    assert score["results"][variant.alias]["fields"]["name"]["accuracy"] == 1
    agreement = evaluate_comparison(db_path=db, variant_aliases=[variant.alias], reference_variant="vlm_current")
    assert agreement["mode"] == "variant_agreement" and agreement["missing_references"] == 1
    score = evaluate_comparison(db_path=db, variant_aliases=["vlm_current", variant.alias], label_set="gt-v1")
    assert score["results"]["vlm_current"]["missing_predictions"] == 1
    assert score["results"][variant.alias]["documents"] == 1


def test_review_and_export_use_full_identity(setup, tmp_path):
    db, _, doc, variant = setup
    run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    with connect(db) as c:
        summary = export_variant_csv(c, output_csv=tmp_path / "out.csv", variant_alias=variant.alias)
    assert summary.rows == 1
    candidate = {"id": 1, "extraction_output_id": 1, "extraction_method": variant.method,
                 "extraction_provider": variant.provider, "extraction_model": variant.model,
                 "extraction_prompt_version": variant.prompt_version, "fields": {}}
    assert _select_review_candidates([candidate], (variant,)) == []
    candidate["extraction_inference_config_hash"] = variant.inference_config_hash
    assert len(_select_review_candidates([candidate], (variant,))) == 1


def test_token_observer_skips_prompt_and_syncs_once():
    calls = []
    observer = FirstTokenObserver(lambda: calls.append("sync"), clock=lambda: 123)
    observer.put("prompt")
    assert observer.first_token_at is None
    observer.put("token1")
    observer.put("token2")
    assert observer.first_token_at == 123 and calls == ["sync"]


def test_cli_eval_and_export_report(setup, tmp_path, capsys):
    db, _, _, variant = setup
    run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=FakeBackend)
    output = tmp_path / "report.json"
    assert main(["eval", "run", "--db", str(db), "--variant", variant.alias,
                 "--label-set", "gt-v1", "--output-file", str(output)]) == 0
    assert json.loads(output.read_text())["mode"] == "gt_accuracy"
    assert "GT accuracy" in capsys.readouterr().out
    args = build_parser().parse_args(["local-vlm", "benchmark", "--variant", variant.alias])
    assert args.warmups == 1 and args.repetitions == 3


def test_normal_extraction_failures_and_force(setup):
    db, image, doc, variant = setup
    result = run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    assert result["successful"] == 1
    class Broken(FakeBackend):
        response = "bad"
    failed = run_local(db_path=db, variant=variant, force=True, backend_factory=Broken)
    assert failed["failed"] == 1
    with connect(db) as c:
        # A failed replacement must leave the prior successful prediction active.
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))


def test_default_report_reference_excludes_unlabeled_documents(setup, tmp_path):
    db, _, _, variant = setup
    with connect(db) as c:
        upsert_document(c, source_name="paper", filename_stem="unlabeled", image_path=tmp_path / "other.jpg")
    result = evaluate_comparison(db_path=db, variant_aliases=[variant.alias])
    assert result["cohort_documents"] == 1


def test_benchmark_report_command_without_loading_gpu(setup, tmp_path, capsys):
    db, _, _, variant = setup
    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=FakeBackend)
    output = tmp_path / "export.json"
    assert main(["local-vlm", "report", "--db", str(db), "--run-id", run["run_id"],
                 "--label-set", "gt-v1", "--output-file", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["quality"]["mode"] == "gt_accuracy"
    assert report["run"]["id"] == run["run_id"]


def test_backend_instrumentation_with_fake_gpu(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from src.todesanzeigen.local_backend import TransformersBackend
    events = []
    class Tensor:
        shape = (1, 2)
        def __getitem__(self, item):
            return self
        def __int__(self):
            return 7
    class Inputs(dict):
        def to(self, device):
            events.append("transfer")
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["path"] == str((tmp_path / "image.jpg").resolve())
            return Inputs(input_ids=Tensor())
        def batch_decode(self, answer, **kwargs):
            return ["{}"]
    class Model:
        generation_config = SimpleNamespace(eos_token_id=7)
        def generate(self, **kwargs):
            assert kwargs["use_cache"] is False and kwargs["do_sample"] is False
            kwargs["streamer"].put("prompt")
            kwargs["streamer"].put("token")
            kwargs["streamer"].put("token")
            return Tensor()
    cuda = SimpleNamespace(synchronize=lambda: events.append("sync"), reset_peak_memory_stats=lambda: events.append("reset"),
                           memory_allocated=lambda: 100, max_memory_allocated=lambda: 150, max_memory_reserved=lambda: 200)
    backend = TransformersBackend("test", InferenceConfig("a" * 40, use_cache=False))
    backend.model, backend.processor = Model(), Processor()
    backend.torch = SimpleNamespace(cuda=cuda, manual_seed=lambda value: None, inference_mode=nullcontext)
    raw, metrics = backend.infer("prompt", tmp_path / "image.jpg")
    assert raw == "{}"
    assert events.count("reset") == 1
    assert events.count("sync") == 4  # before preprocessing, before generation, first token, generation end
    assert metrics["input_tokens"] == 2 and metrics["output_tokens"] == 2
    assert metrics["peak_allocated_bytes"] == 150 and metrics["resident_allocated_bytes"] == 100
    assert metrics["first_token_ms"] is not None
    assert metrics["decode_tokens_per_second"] > 0


def test_gpu_smoke_opt_in(tmp_path):
    import os
    if os.getenv("TODESANZEIGEN_GPU_SMOKE") != "1":
        pytest.skip("Set TODESANZEIGEN_GPU_SMOKE=1 and TODESANZEIGEN_SMOKE_IMAGE for real inference")
    image = Path(os.environ["TODESANZEIGEN_SMOKE_IMAGE"])
    from src.todesanzeigen.local_backend import TransformersBackend
    variant = load_variant_config().variant("local_qwen3vl_fp16")
    backend = TransformersBackend(variant.model, variant.inference)
    backend.load()
    raw, metrics = backend.infer("Lies den Namen in dieser Todesanzeige.", image)
    assert raw.strip() and metrics["output_tokens"] > 0 and metrics["peak_allocated_bytes"] > 0


@pytest.mark.parametrize("weight_format,dtype", [("fp16", "float16"), ("bf16", "bfloat16"),
                                                  ("int8", "float16"), ("nf4", "float16")])
def test_backend_load_maps_precision_and_revision(monkeypatch, weight_format, dtype):
    import sys
    from types import SimpleNamespace
    from src.todesanzeigen import local_backend
    calls = {}
    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["processor"] = kwargs
            return cls()
    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model"] = kwargs
            return cls()
        def eval(self):
            return self
        def named_modules(self):
            if weight_format in {"nf4", "int8"}:
                name = "Linear4bit" if weight_format == "nf4" else "Linear8bitLt"
                return [("decoder.layer", type(name, (), {})())]
            return []
    def quantization(**kwargs):
        calls["quantization"] = kwargs
        return kwargs
    def snapshot(model_id, **kwargs):
        calls["snapshot"] = kwargs
        return "/cached/model"
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                           synchronize=lambda: None, get_device_name=lambda _: "Fake GPU",
                           get_device_properties=lambda _: SimpleNamespace(total_memory=1024))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="12.8"),
                                                            float16="fp16", bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor,
                      BitsAndBytesConfig=quantization, Qwen3VLForConditionalGeneration=Model))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setattr(local_backend.importlib.metadata, "version", lambda _: "fake")
    monkeypatch.setattr(local_backend.subprocess, "check_output", lambda *a, **kw: "fake-driver")
    config = InferenceConfig("b" * 40, weight_format=weight_format, compute_dtype=dtype)
    backend = local_backend.TransformersBackend("Qwen/test", config)
    load_ms, env = backend.load()
    assert load_ms >= 0 and env["gpu"] == "Fake GPU"
    assert calls["snapshot"]["revision"] == "b" * 40
    assert calls["processor"]["size"]["longest_edge"] == config.max_pixels
    assert calls["model"]["device_map"] == {"": "cuda:0"}
    if weight_format in {"nf4", "int8"}:
        assert calls["quantization"]["load_in_4bit"] == (weight_format == "nf4")
        assert calls["quantization"]["load_in_8bit"] == (weight_format == "int8")
        assert calls["quantization"]["llm_int8_skip_modules"] == ["visual", "lm_head"]
    else:
        assert calls["model"]["quantization_config"] is None


def test_historical_quality_without_variant_file(setup, tmp_path):
    db, _, _, variant = setup
    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=FakeBackend)
    report = evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1",
                                 variants_config=tmp_path / "deleted.toml")
    assert report["results"]["run:" + run["run_id"]]["documents"] == 1
