from __future__ import annotations

from dataclasses import dataclass


TEXT_EXTRACTION_METHOD = "text_extraction"
VISION_REROUTE_METHOD = "vision_model_reroute"
VISION_IMAGE_ONLY_METHOD = "vision_model_image_only"

OCR_LLM_FAMILY = "ocr_llm"
VLM_FAMILY = "vlm"
OCR_LLM_SLOT = "ocr_llm"
VLM_SLOT = "vlm"

GT_EXPORT_SOURCE = "ground_truth"
DEFAULT_EXPORT_METHOD_PRIORITY = (
    VISION_IMAGE_ONLY_METHOD,
    VISION_REROUTE_METHOD,
    TEXT_EXTRACTION_METHOD,
)


@dataclass(frozen=True)
class ExtractionMethodDefinition:
    method: str
    method_family: str
    result_slot: str
    default_route_reason: str = ""
    description: str = ""


METHOD_DEFINITIONS = {
    TEXT_EXTRACTION_METHOD: ExtractionMethodDefinition(
        method=TEXT_EXTRACTION_METHOD,
        method_family=OCR_LLM_FAMILY,
        result_slot=OCR_LLM_SLOT,
        description="OCR text plus local name hint passed to a text LLM.",
    ),
    VISION_IMAGE_ONLY_METHOD: ExtractionMethodDefinition(
        method=VISION_IMAGE_ONLY_METHOD,
        method_family=VLM_FAMILY,
        result_slot=VLM_SLOT,
        default_route_reason="image_only",
        description="Direct image-only vision-language extraction.",
    ),
    VISION_REROUTE_METHOD: ExtractionMethodDefinition(
        method=VISION_REROUTE_METHOD,
        method_family=VLM_FAMILY,
        result_slot=VLM_SLOT,
        default_route_reason="low_confidence",
        description="Vision-language extraction used after low OCR/name confidence.",
    ),
}


def method_family(method: str) -> str:
    definition = METHOD_DEFINITIONS.get(method)
    return definition.method_family if definition else ""


def default_route_reason(method: str) -> str:
    definition = METHOD_DEFINITIONS.get(method)
    return definition.default_route_reason if definition else ""


def result_slot(method: str) -> str:
    definition = METHOD_DEFINITIONS.get(method)
    return definition.result_slot if definition else method
