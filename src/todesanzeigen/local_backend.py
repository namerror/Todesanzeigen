"""Qwen3-VL inference and GPU instrumentation, isolated from pipeline dependencies."""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .inference_config import InferenceConfig


class FirstTokenObserver:
    """generate() sends the prompt once, then generated token tensors."""

    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.prompt_seen = False
        self.first_token_at: float | None = None

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_at is None:
            self.synchronize()
            self.first_token_at = self.clock()

    def end(self):
        pass


class TransformersBackend:
    def __init__(self, model_id: str, config: InferenceConfig):
        self.model_id, self.config = model_id, config
        self.model = None

    def load(self) -> tuple[float, dict[str, Any]]:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the local-vlm optional dependencies to run local models") from exc
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Enable the NVIDIA GPU and check nvidia-smi/PyTorch.")
        if self.config.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 is unsupported by this GPU")
        dtype = getattr(torch, self.config.compute_dtype)
        quantization = None
        if self.config.weight_format in {"int8", "nf4"}:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("INT8/NF4 requires bitsandbytes from the local-vlm extra") from exc
            quantization = BitsAndBytesConfig(
                load_in_8bit=self.config.weight_format == "int8",
                load_in_4bit=self.config.weight_format == "nf4",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=False,
                # Preserve the visual encoder across language-weight experiments.
                llm_int8_skip_modules=["visual", "lm_head"],
            )
        # Populate the HF cache before starting the load timer (network excluded).
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(self.model_id, revision=self.config.revision)
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            snapshot, size={"shortest_edge": self.config.min_pixels, "longest_edge": self.config.max_pixels},
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot, dtype=dtype, device_map={"": "cuda:0"},
            attn_implementation=self.config.attention_backend,
            quantization_config=quantization,
        ).eval()
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - started) * 1000
        self.load_ms = load_ms
        self.environment = {
            "gpu": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "accelerate", "huggingface-hub")},
            "quantized_modules": [name for name, module in self.model.named_modules()
                                  if type(module).__name__ in {"Linear8bitLt", "Linear4bit"}],
        }
        try:
            self.environment["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            self.environment["driver"] = None
        try:
            self.environment["packages"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            pass
        return load_ms, self.environment

    def infer(self, prompt: str, image_path: Path) -> tuple[str, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Load the local model before inference")
        torch = self.torch
        torch.manual_seed(self.config.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        started = time.perf_counter()
        messages = [{"role": "user", "content": [
            {"type": "image", "path": str(image_path.resolve())},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to("cuda:0")
        torch.cuda.synchronize()
        generation_start = time.perf_counter()
        observer = FirstTokenObserver(torch.cuda.synchronize)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                do_sample=False, use_cache=self.config.use_cache, streamer=observer,
            )
        torch.cuda.synchronize()
        generation_end = time.perf_counter()
        answer = output[:, inputs["input_ids"].shape[1]:]
        count = int(answer.shape[1])  # Includes EOS when emitted.
        text = self.processor.batch_decode(answer, skip_special_tokens=True)[0]
        first = observer.first_token_at
        decode_seconds = generation_end - first if first is not None else 0
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = count >= self.config.max_new_tokens and (not count or int(answer[0, -1]) not in eos_ids)
        metrics = {
            "preprocess_ms": (generation_start - started) * 1000,
            "generation_ms": (generation_end - generation_start) * 1000,
            "first_token_ms": (first - generation_start) * 1000 if first is not None else None,
            "decode_tokens_per_second": (count - 1) / decode_seconds if count > 1 and decode_seconds > 0 else None,
            "inference_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": int(inputs["input_ids"].shape[1]), "output_tokens": count,
            "resident_allocated_bytes": baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "output_limit_reached": truncated,
        }
        return text, metrics
