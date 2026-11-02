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
from src.todesanzeigen.local_vlm import benchmark_report, run_local, select_images, variant_lookup
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
    variant = load_variant_config().variant("local_qwen3vl_bf16")
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
                 {"revision": "a" * 40, "weight_format": "fp16"},
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
    assert [s["extraction_output_id"] is not None for s in report["samples"]] == [False, True, False, False]
    with connect(db) as c:
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 1
        old_id = active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))["id"]
    cached = run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    assert cached["cached"] == 1 and FakeBackend.calls == 4
    run_local(db_path=db, variant=variant, benchmark=True, repetitions=1, warmups=0, backend_factory=FakeBackend)
    with connect(db) as c:
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))["id"] == old_id
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 1
    quality = evaluate_comparison(db_path=db, run_id=first["run_id"], label_set="gt-v1")
    score = quality["results"]["run:" + first["run_id"]]
    assert score["fields"]["name"]["f1"] == 1
    assert score["prediction_output_ids"][str(doc)] == old_id
    assert score["prediction_sample_ids"][str(doc)] == report["samples"][1]["id"]
    assert benchmark_report(db, first["run_id"])["samples"] == report["samples"]
    image.write_bytes(b"changed")
    result = run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    assert result["cached"] == 0
    assert evaluate_comparison(db_path=db, run_id=first["run_id"], label_set="gt-v1") == quality


@pytest.mark.parametrize("changed_image", [False, True])
def test_benchmark_preserves_existing_prediction_and_scores_own_answer(setup, changed_image):
    db, image, doc, variant = setup
    run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    with connect(db) as c:
        original = dict(active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant)))
        candidates = [dict(row) for row in c.execute("SELECT * FROM label_candidates")]
    if changed_image:
        image.write_bytes(b"new-image-content")

    class DifferentAnswers(FakeBackend):
        def __init__(self, *args):
            super().__init__(*args)
            self.answers = iter(["Warmup", "Different", "Muster"])

        def infer(self, prompt, image):
            _, metrics = super().infer(prompt, image)
            return json.dumps({**json.loads(self.response), "nachname": next(self.answers)}), metrics

    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=1, repetitions=2,
                    backend_factory=DifferentAnswers)
    with connect(db) as c:
        assert dict(active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))) == original
        assert [dict(row) for row in c.execute("SELECT * FROM label_candidates")] == candidates
        assert c.execute("SELECT count(*) FROM extraction_outputs").fetchone()[0] == 1
    samples = run["benchmark"]["samples"]
    assert all(s["extraction_output_id"] is None for s in samples)
    quality = evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1")
    score = quality["results"]["run:" + run["run_id"]]
    assert score["fields"]["name"]["f1"] == 0
    assert score["prediction_output_ids"] == {}
    assert score["prediction_sample_ids"] == {str(doc): samples[1]["id"]}
    # Subsequent explicit extraction replacement cannot change benchmark quality.
    run_local(db_path=db, variant=variant, force=True, backend_factory=FakeBackend)
    assert evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1") == quality


@pytest.mark.parametrize("history", ["failed", "superseded"])
def test_benchmark_fills_gap_after_failed_or_superseded_history(setup, history):
    db, _, doc, variant = setup
    with connect(db) as c:
        old_id = insert_extraction_output(
            c, document_id=doc, run_id=None, fields={},
            status="vision_failed" if history == "failed" else "vision_processed",
            superseded_at=None if history == "failed" else "2026-01-01",
            **variant_lookup(variant))
        original = dict(c.execute("SELECT * FROM extraction_outputs WHERE id=?", (old_id,)).fetchone())

    class InitiallyInvalid(FakeBackend):
        def __init__(self, *args):
            super().__init__(*args)
            self.answers = iter([self.response, "invalid", self.response, self.response])

        def infer(self, prompt, image):
            _, metrics = super().infer(prompt, image)
            return next(self.answers), metrics

    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=1, repetitions=3,
                    backend_factory=InitiallyInvalid)
    samples = run["benchmark"]["samples"]
    assert [s["extraction_output_id"] is not None for s in samples] == [False, False, True, False]
    with connect(db) as c:
        assert dict(c.execute("SELECT * FROM extraction_outputs WHERE id=?", (old_id,)).fetchone()) == original
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))["id"] == samples[2]["extraction_output_id"]
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 1
    score = evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1")["results"]["run:" + run["run_id"]]
    assert score["fields"]["name"]["f1"] == 1
    assert score["prediction_sample_ids"] == {str(doc): samples[2]["id"]}


def test_failed_benchmark_does_not_fall_back_to_existing_prediction(setup):
    db, _, doc, variant = setup
    run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    backend = type("InvalidBackend", (FakeBackend,), {"response": "invalid"})
    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, backend_factory=backend)
    score = evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1")["results"]["run:" + run["run_id"]]
    assert score["missing_predictions"] == 1
    assert score["prediction_sample_ids"] == score["prediction_output_ids"] == {}
    with connect(db) as c:
        assert active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant)) is not None
        assert c.execute("SELECT count(*) FROM extraction_outputs").fetchone()[0] == 1


def test_benchmark_preserves_prediction_published_during_inference(setup):
    db, _, doc, variant = setup

    class ConcurrentPublisher(FakeBackend):
        def infer(self, prompt, image):
            # Simulate a separate writer publishing after selection, before persistence.
            with connect(db) as c:
                self.output_id = insert_extraction_output(
                    c, document_id=doc, run_id=None, fields={"name": "Other writer"},
                    status="vision_processed", **variant_lookup(variant))
            return super().infer(prompt, image)

    run = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, repetitions=1,
                    backend_factory=ConcurrentPublisher)
    assert run["status"] == "completed"
    assert run["benchmark"]["samples"][0]["extraction_output_id"] is None
    with connect(db) as c:
        active = active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))
        assert json.loads(active["fields_json"])["name"] == "Other writer"
        assert c.execute("SELECT count(*) FROM extraction_outputs").fetchone()[0] == 1
        assert c.execute("SELECT count(*) FROM label_candidates").fetchone()[0] == 0


@pytest.mark.parametrize("benchmark", [False, True])
def test_cache_settings_do_not_supersede_each_other(setup, benchmark):
    db, _, doc, variant = setup
    second = replace(variant, inference=replace(variant.inference, use_cache=False))
    for v in (variant, second):
        run_local(db_path=db, variant=v, benchmark=benchmark, backend_factory=FakeBackend)
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


def test_missing_image_fails_before_backend_load(setup, capsys):
    db, image, _, variant = setup
    image.unlink()
    def unexpected_backend(*args):
        pytest.fail("Empty selection must not construct a backend")
    with pytest.raises(ValueError, match="No eligible images"):
        run_local(db_path=db, variant=variant, benchmark=True, backend_factory=unexpected_backend)
    assert '"missing_invalid_images":1' in capsys.readouterr().out
    with connect(db) as c:
        assert c.execute("SELECT count(*) FROM inference_benchmark_runs").fetchone()[0] == 0


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


def test_cli_benchmark_report_and_eval_score_samples_without_replacing(setup, tmp_path, monkeypatch):
    from src.todesanzeigen import local_backend
    db, _, doc, variant = setup
    run_local(db_path=db, variant=variant, backend_factory=FakeBackend)
    backend = type("DifferentBackend", (FakeBackend,), {
        "response": json.dumps({**json.loads(FakeBackend.response), "nachname": "Different"}),
    })
    monkeypatch.setattr(local_backend, "TransformersBackend", backend)
    output = tmp_path / "benchmark.json"
    assert main(["local-vlm", "benchmark", "--db", str(db), "--variant", variant.alias,
                 "--label-set", "gt-v1", "--warmups", "0", "--repetitions", "2",
                 "--output-file", str(output)]) == 0
    benchmark = json.loads(output.read_text())
    run_id = benchmark["run_id"]
    quality = benchmark["quality"]
    score = quality["results"]["run:" + run_id]
    assert score["fields"]["name"]["f1"] == 0
    assert score["prediction_output_ids"] == {}
    assert score["prediction_sample_ids"] == {str(doc): benchmark["benchmark"]["samples"][0]["id"]}
    assert main(["local-vlm", "report", "--db", str(db), "--run-id", run_id,
                 "--label-set", "gt-v1", "--output-file", str(output)]) == 0
    assert json.loads(output.read_text())["quality"] == quality
    assert main(["eval", "run", "--db", str(db), "--run-id", run_id,
                 "--label-set", "gt-v1", "--output-file", str(output)]) == 0
    assert json.loads(output.read_text()) == quality
    with connect(db) as c:
        active = active_extraction_for_variant(c, document_id=doc, **variant_lookup(variant))
        assert json.loads(active["fields_json"])["name"] == "Muster"


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
    variant = load_variant_config().variant("local_qwen3vl_bf16")
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


def register_image(db, path, *, source="paper", stored_hash=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())
    with connect(db) as c:
        return upsert_document(c, source_name=source, filename_stem=path.stem,
                               image_path=path, image_sha256=stored_hash)


def test_combined_selection_and_source_only(setup, tmp_path):
    db, _, original, _ = setup
    directory = tmp_path / "inputs"
    chosen = register_image(db, directory / "chosen.jpg")
    other_source = register_image(db, directory / "other.jpg", source="other")
    excluded = register_image(db, directory / "excluded.jpg")
    with connect(db) as c:
        save_ground_truth_label(c, document_id=chosen, label_set="gt-v1", fields={})
        create_dataset_split(c, name="selection", strategy="test", assignments={
            chosen: "test", original: "test", excluded: "validation", other_source: "test"})
        selected, summary = select_images(c, source="paper", input_dir=directory,
                                          split_name="selection", label_set="gt-v1",
                                          only=["chosen.jpg", "notice"], limit=1)
        assert [d["id"] for d, _ in selected] == [chosen]
        assert summary["matching_documents"] == 2
        assert summary["available_images"] == 1
        assert summary["missing_invalid_images"] == 1
        selected, _ = select_images(c, source="other")
        assert [d["id"] for d, _ in selected] == [other_source]
        selected, _ = select_images(c, split_name="selection", subset="validation")
        assert [d["id"] for d, _ in selected] == [excluded]


@pytest.mark.parametrize("benchmark", [False, True])
def test_missing_and_mismatched_images_do_not_consume_limit(setup, tmp_path, benchmark):
    db, _, original, variant = setup
    missing = register_image(db, tmp_path / "aaa-missing.jpg")
    (tmp_path / "aaa-missing.jpg").unlink()
    mismatch = register_image(db, tmp_path / "aab-mismatch.jpg", stored_hash="0" * 64)
    result = run_local(db_path=db, variant=variant, benchmark=benchmark, limit=1,
                       warmups=0, repetitions=1, backend_factory=FakeBackend)
    assert result["selected_document_ids"] == [original]
    assert result["selection"]["available_images"] == 1
    assert result["selection"]["missing_invalid_images"] == 2
    assert {i["document_id"] for i in result["selection"]["invalid_images"]} == {missing, mismatch}
    assert any("SHA-256 mismatch" in i["reason"] for i in result["selection"]["invalid_images"])


def test_benchmark_directory_manifest_and_no_record_updates(setup, tmp_path):
    db, image, doc, variant = setup
    with connect(db) as c:
        c.execute("UPDATE documents SET image_sha256=? WHERE id=?", (sha256_file(image), doc))
        before = dict(c.execute("SELECT * FROM documents WHERE id=?", (doc,)).fetchone())
    result = run_local(db_path=db, variant=variant, benchmark=True, source="paper", input_dir=tmp_path,
                       warmups=0, repetitions=1, backend_factory=FakeBackend)
    assert result["benchmark"]["run"]["manifest"] == [{
        "document_id": doc, "image_path": str(image.resolve()), "image_sha256": sha256_file(image)}]
    with connect(db) as c:
        assert dict(c.execute("SELECT * FROM documents WHERE id=?", (doc,)).fetchone()) == before


def test_relocated_copy_is_rejected_even_with_matching_hash(setup, tmp_path):
    db, image, doc, variant = setup
    digest = sha256_file(image)
    with connect(db) as c:
        c.execute("UPDATE documents SET image_sha256=? WHERE id=?", (digest, doc))
    directory = tmp_path / "restored"
    directory.mkdir()
    (directory / image.name).write_bytes(image.read_bytes())
    # A copy must not qualify even while the recorded original remains available.
    with pytest.raises(ValueError, match="No eligible images"):
        run_local(db_path=db, variant=variant, benchmark=True, input_dir=directory,
                  backend_factory=lambda *args: pytest.fail("Must not load"))
    image.unlink()
    with connect(db) as c:
        selected, summary = select_images(c, input_dir=directory)
        assert selected == [] and summary["missing_invalid_images"] == 1
        assert c.execute("SELECT image_path FROM documents WHERE id=?", (doc,)).fetchone()[0] == str(image)
        assert c.execute("SELECT count(*) FROM documents").fetchone()[0] == 1


def test_directory_aliases_are_ambiguous(setup, tmp_path):
    db, image, _, _ = setup
    (tmp_path / "alias.jpg").symlink_to(image)
    with connect(db) as c, pytest.raises(ValueError, match="Ambiguous image matches"):
        select_images(c, input_dir=tmp_path)


def test_reference_coverage_does_not_restrict_inference(setup, tmp_path, capsys):
    db, _, doc, variant = setup
    other = register_image(db, tmp_path / "other.jpg")
    reference = load_variant_config().variant("vlm_current")
    with connect(db) as c:
        insert_extraction_output(c, document_id=doc, run_id=None, status="vision_processed",
                                 fields={"name": "Muster"}, **variant_lookup(reference))
    class ObservedBackend(FakeBackend):
        def load(self):
            assert '"reference_coverage":{"available":1,"missing":1,"total":2}' in capsys.readouterr().out
            return super().load()
    result = run_local(db_path=db, variant=variant, benchmark=True, reference=reference,
                       warmups=0, repetitions=1, backend_factory=ObservedBackend)
    assert result["successful"] == 2
    assert set(result["selected_document_ids"]) == {doc, other}
    quality = evaluate_comparison(db_path=db, run_id=result["run_id"], reference_variant=reference.alias)
    assert quality["mode"] == "variant_agreement"
    assert quality["cohort_documents"] == 2 and quality["missing_references"] == 1
    assert quality["results"]["run:" + result["run_id"]]["documents"] == 1


@pytest.mark.parametrize("all_cached", [False, True])
def test_cli_extraction_evaluates_only_selected_including_cache(setup, tmp_path, monkeypatch, all_cached):
    from src.todesanzeigen import local_backend
    db, _, doc, variant = setup
    second = register_image(db, tmp_path / "second.jpg")
    unrelated = register_image(db, tmp_path / "unrelated.jpg", source="elsewhere")
    with connect(db) as c:
        for document_id in (second, unrelated):
            save_ground_truth_label(c, document_id=document_id, label_set="gt-v1", fields={"name": "Muster"})
    run_local(db_path=db, variant=variant, only=["notice"] if not all_cached else None, backend_factory=FakeBackend)
    monkeypatch.setattr(local_backend, "TransformersBackend", FakeBackend)
    output = tmp_path / "cohort.json"
    assert main(["local-vlm", "extract", "--db", str(db), "--variant", variant.alias,
                 "--source", "paper", "--limit", "2", "--label-set", "gt-v1", "--output-file", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["selected_document_ids"] == [doc, second]
    assert result["cached"] == (2 if all_cached else 1)
    assert result["quality"]["cohort_documents"] == 2
    assert {row["document_id"] for row in result["quality"]["reference_rows"]} == {doc, second}
    empty = evaluate_comparison(db_path=db, variant_aliases=[variant.alias], document_ids=[])
    assert empty["cohort_documents"] == 0


def test_extraction_ingestion_assigns_and_filters_source(setup, tmp_path):
    db, _, _, variant = setup
    directory = tmp_path / "new"
    directory.mkdir()
    (directory / "ingested.jpg").write_bytes(b"new image")
    result = run_local(db_path=db, variant=variant, input_dir=directory, source="new-paper", backend_factory=FakeBackend)
    assert result["selection"]["matching_documents"] == 1
    with connect(db) as c:
        row = c.execute("SELECT d.*, s.name AS source FROM documents d JOIN sources s ON s.id=d.source_id WHERE d.id=?",
                        (result["selected_document_ids"][0],)).fetchone()
        assert row["source"] == "new-paper"
        assert row["image_sha256"] == sha256_file(directory / "ingested.jpg")


def test_recorded_relative_paths_and_same_basename_do_not_relocate(setup, tmp_path, monkeypatch):
    db, image, doc, _ = setup
    directory = tmp_path / "different"
    register_image(db, directory / image.name, source="other")
    monkeypatch.chdir(tmp_path)
    with connect(db) as c:
        c.execute("UPDATE documents SET image_path=? WHERE id=?", (image.name, doc))
        selected, _ = select_images(c, source="paper", input_dir=tmp_path)
        assert selected[0][1]["image_path"] == str(image)
        selected, _ = select_images(c, source="paper", input_dir=directory)
        assert selected == []


def test_benchmark_quality_is_isolated_to_manifest(setup, tmp_path):
    db, _, doc, variant = setup
    other = register_image(db, tmp_path / "other.jpg")
    with connect(db) as c:
        save_ground_truth_label(c, document_id=other, label_set="gt-v1", fields={})
    run = run_local(db_path=db, variant=variant, benchmark=True, only=["notice"],
                    warmups=0, repetitions=1, backend_factory=FakeBackend)
    quality = evaluate_comparison(db_path=db, run_id=run["run_id"], label_set="gt-v1")
    assert quality["cohort_documents"] == 1
    assert [r["document_id"] for r in quality["reference_rows"]] == [doc]


def test_cli_benchmark_wires_source_directory_and_reference(setup, tmp_path, monkeypatch):
    from src.todesanzeigen import local_backend
    db, _, doc, variant = setup
    register_image(db, tmp_path / "other.jpg", source="other")
    monkeypatch.setattr(local_backend, "TransformersBackend", FakeBackend)
    output = tmp_path / "benchmark.json"
    assert main(["local-vlm", "benchmark", "--db", str(db), "--variant", variant.alias,
                 "--source", "paper", "--input-dir", str(tmp_path), "--reference-variant", "vlm_current",
                 "--warmups", "0", "--repetitions", "1", "--output-file", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["selected_document_ids"] == [doc]
    assert report["successful"] == 1
    assert report["selection"]["reference_coverage"] == {"available": 0, "missing": 1, "total": 1}
    assert report["quality"]["missing_references"] == 1
    assert report["quality"]["results"]["run:" + report["run_id"]]["documents"] == 0


def test_image_change_after_selection_remains_durable_failure(setup):
    db, image, _, variant = setup
    class ChangedImage(FakeBackend):
        def load(self):
            image.write_bytes(b"changed after selection")
            return super().load()
        def infer(self, *args):
            pytest.fail("Changed image must not reach inference")
    report = run_local(db_path=db, variant=variant, benchmark=True, warmups=0, repetitions=1,
                       backend_factory=ChangedImage)["benchmark"]
    assert report["attempted_samples"] == 1
    assert report["successful_samples"] == 0
    assert "Image changed" in report["samples"][0]["error"]
