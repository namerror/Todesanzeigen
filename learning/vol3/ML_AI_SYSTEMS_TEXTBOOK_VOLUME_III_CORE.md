# From OCR Pipeline to Owned Model Stack

## Volume III — Fine-Tuning and Rigorous Evaluation

Volume II produced a tokenizer, a decoder-only Transformer, a bounded domain-pretraining run, and a checkpoint you can explain end to end. That model learned next-token statistics. It was not taught the operational contract of the Family History Intelligence Platform:

- read OCR text and a permitted hint;
- extract only explicitly supported facts;
- emit a fixed schema;
- leave unknown values empty;
- preserve rare names and dates exactly;
- avoid inventing relationships, places, or occupations;
- remain auditable against a specific document and model version.

This volume adds that task behavior through supervised fine-tuning, but the adapter is only half the work. The other half is building an evaluation suite capable of rejecting a model that is fluent, cheap, and wrong.

The core task is deliberately narrow:

> **OCR text plus the same approved hints used by the text-extraction path → canonical structured JSON.**

That task maps naturally to the current repository and supports deterministic evaluation. Synthetic relationship/date reasoning and retrieval-grounded answering probes are added as regression suites, but they do not become production features in this volume.

The loop remains:

> **Predict → Implement → Run → Observe → Explain → Improve**

This volume stops before serving, KV caching, batching, and quantization.

---

# Volume contract

## What you should already own

You should be able to:

- explain causal language modeling and token-level cross-entropy;
- trace every major tensor in a decoder-only Transformer;
- state how causal masking prevents future leakage;
- train and resume a small model reproducibly;
- distinguish tokenizer identity from model identity;
- inspect gradients, residual-stream statistics, throughput, and memory;
- preserve grouped train/validation/test boundaries;
- explain why language-model loss is not task quality.

You do not need a high-quality Volume II checkpoint. For the useful fine-tuning run, you will usually adapt a suitable pretrained model. The model you built from scratch remains the conceptual oracle: it prevents LoRA, chat templates, and trainer abstractions from becoming magic.

## Core timebox: 10 hours

| Part | Core work | Time |
|---|---|---:|
| I. Task and benchmark | contract, reviewed data, frozen split, metrics, slices, baselines | 2.5 h |
| II. Fine-tuning mechanics | SFT loss/masking, LoRA from first principles, production PEFT/QLoRA | 3 h |
| III. Bounded adaptation and evaluation | model choice, adapter run, forgetting and robustness evaluation | 3 h |
| Capstone and mastery check | paired decision report and reconstruction | 1.5 h |
| **Total** |  | **10 h** |

Manual review time can exceed the core if the benchmark needs substantial new labels. Do not weaken review quality to preserve a stopwatch. Instead, distinguish:

- the **core mechanism run**, which can use a small frozen benchmark and makes limited claims;
- the **production extension**, which expands representative reviewed coverage before deployment.

## Repository-grounded starting point

As inspected for this volume, the project contains:

| Item | Count | Consequence |
|---|---:|---|
| Documents with OCR lineage | 1,168 | Enough unlabeled domain text and candidate review material |
| Extraction outputs | 2,326 | Existing text/VLM baselines are versioned and comparable |
| Reviewed ground-truth rows | 44 | Enough to verify evaluation plumbing; too few for a broad product claim |
| Feature snapshots | 0 | Router work is not the core of this volume |
| Named dataset splits | 0 | A split must be frozen before adaptation or comparison |

The configured operational baselines are exact variants, not generic labels:

- `text_current`: OCR plus text model, including provider/model/prompt identity;
- `vlm_current`: image-only vision-language extraction with its own exact identity.

The review interface can compare candidates and save human-reviewed ground truth. Teacher or pipeline candidates are review aids, not truth by themselves.

## Exit artifact

By the end, produce:

- a versioned prompt-completion dataset manifest;
- a named, grouped split and leakage report;
- a deterministic extraction evaluator with document-level paired comparisons;
- a hand-built LoRA linear layer with merge/unmerge and gradient tests;
- one reproducible adapter trained through a maintained PEFT/SFT stack;
- a compact evaluation suite covering extraction, hallucination, rare entities, dates, OCR noise, grounding, and forgetting;
- a decision report stating whether the adapter is rejected, retained for more research, or eligible for shadow mode.

An adapter that is rejected by the benchmark is still a successful educational artifact if the rejection is trustworthy.

## Suggested workspace

```text
learning/vol3/
  dataset.py
  format_example.py
  lora_linear.py
  train_adapter.py
  evaluate.py
  metrics.py
  slices.py
  judge_audit.py
  fixtures/
  tests/
  runs/
    <run-id>/
      config.json
      dataset_manifest.json
      adapter/
      metrics.json
      predictions.jsonl
      report.md
```

Private examples, raw prompts, predictions, and adapters trained on private text stay local unless explicitly reviewed. The production package must not import educational code.

## Claim levels

Every conclusion must name its level:

1. **Mechanism:** LoRA gradients, masking, serialization, and metric logic are correct.
2. **Benchmark:** the adapter changed outcomes on one frozen dataset with stated uncertainty.
3. **Product:** the adapter improves representative production traffic under quality, latency, privacy, and fallback constraints.

This volume can earn levels 1 and 2. Forty-four labels cannot establish level 3.

## Manual-versus-library boundary

| Component | Decision | Reason |
|---|---|---|
| SFT loss mask on one batch | Implement and inspect manually | Prompt/completion masking is central and easy to get silently wrong |
| LoRA linear layer | Implement from first principles | Rank, scaling, initialization, gradients, and merging are the mechanism |
| Full pretrained-model LoRA injection | Use a maintained PEFT library | Model-family module naming and checkpoint integration are maintenance work |
| QLoRA quantized loading | Use maintained libraries | Correct low-bit kernels are not a ten-hour textbook exercise |
| Dataset formatting and validation | Write explicitly | Most fine-tuning failures begin in data contracts |
| Trainer, optimizer, mixed precision | Use maintained libraries with pinned versions | Volume II already established the training-loop mechanics |
| Domain metrics and slices | Implement and test | They define whether the model is useful |
| Model-as-judge | Optional secondary signal only | A judge cannot replace deterministic truth or human review |

## Evidence hierarchy

Prefer evidence in this order:

1. schema and invariant tests;
2. deterministic field metrics against reviewed labels;
3. paired robustness and counterfactual tests;
4. blinded human review;
5. model-judge scores calibrated against humans;
6. anecdotal samples.

Samples are useful for discovering failure classes. They are weak evidence for frequency.

---

# Part I — Define the Task Before Training

## Chapter 1 — Freeze the extraction contract and data lineage

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** task contract, canonical formatter, and dataset manifest

### Why this comes first

Fine-tuning optimizes whatever byte/token sequence you put in the completion. If two equivalent records serialize differently, the model is punished for formatting rather than facts. If the prompt changes between train and evaluation, you no longer know whether the adapter or interface caused the result.

### Canonical input

Match the information available to the current text-extraction path:

- OCR text;
- local name hint when available;
- name confidence when available;
- task instructions and output schema version.

Do not include:

- image pixels in the text-model experiment;
- reviewed labels in the prompt;
- filename-derived names;
- teacher-model reasoning;
- source fields unavailable at production inference time.

A prompt-completion record should retain lineage outside the model-visible fields:

```json
{
  "example_id": "local-stable-id",
  "group_id": "document-or-duplicate-family",
  "subset": "train",
  "task_version": "death-notice-extraction-v1",
  "prompt": {
    "ocr_text": "...",
    "name_hint": "...",
    "name_confidence": 91.2
  },
  "completion": {
    "geschlecht": "",
    "nachname": "Mustermann",
    "vorname": "Erika",
    "geburtsdatum": "01.02.1930",
    "sterbedatum": "03.04.2020",
    "geburtsname": "",
    "titel": "",
    "genannt": "",
    "geburtsort": "",
    "sterbeort": "",
    "ort": "Aichach",
    "weitere_orte": "",
    "beruf": "",
    "zusaetzliche_hinweise": "",
    "confidence_score": ""
  },
  "lineage": {
    "document_id": 0,
    "label_set": "gt-v1",
    "source_candidate_id": null,
    "reviewed": true
  }
}
```

The values above illustrate schema, not real training data.

### Model schema versus storage schema

The current model-facing contract uses `nachname`; persistence retains legacy `name`. Keep the mapping at one tested boundary:

```text
model completion "nachname"
        │ parse + validate
        ▼
canonical evaluation field "name"
        │ production adapter later
        ▼
stored field "name"
```

Do not mix `name` and `nachname` inside metric code. A schema mismatch can look like a model failure or, worse, silently drop a field.

### Canonical completion

Choose and test:

- fixed key order;
- UTF-8 JSON with no Markdown fence;
- all required keys present;
- string values only, matching the current contract;
- `""` for unknown values;
- normalized date format only when the source supports it;
- no inferred facts;
- one trailing EOS according to the selected chat template.

For `confidence_score`, do not invent a supervision target from the model’s own opinion. In the core adapter, keep the compatibility key empty and exclude it from factual field metrics, or introduce a new explicitly versioned schema without that field. A meaningful confidence score should later come from measured calibration against held-out correctness, not from teaching the model arbitrary decimals.

Canonical JSON is a training target. The evaluator should still parse semantically equivalent whitespace and key order. Formatting validity and field correctness are separate metrics.

### Predict

1. What behavior results if the training completion sometimes omits unknown keys and sometimes emits empty strings?
2. Why can including the answer in a filename or hint produce excellent validation loss?
3. If the model sees a teacher’s unreviewed output as target, what has it actually learned?
4. Should `confidence_score` be supervised if no calibrated human target exists?
5. Why should the prompt version be part of run identity?

### Label policy

Use targets in this order:

1. reviewed ground truth;
2. deterministic synthetic records with known generation truth;
3. teacher/pipeline candidate only after human confirmation.

Unreviewed teacher outputs may support a separate distillation experiment, but label it as distillation and do not evaluate the student against its teacher as though that were ground truth.

### Dataset manifest

Record:

- task and schema version;
- prompt renderer and chat-template fingerprint;
- label-set identity;
- included document/group IDs by subset or privacy-safe hashes;
- data-source counts;
- reviewed/synthetic counts;
- field-population counts;
- slice counts;
- duplicate-grouping method;
- preprocessing/normalization version;
- content fingerprint;
- privacy class and retention policy.

### Tests

- every completion has exactly the schema keys;
- storage/model surname mapping round-trips;
- JSON serialization is deterministic;
- no target fields occur in hidden metadata passed to the model;
- no train group occurs in validation/test;
- no teacher-only candidate is marked reviewed;
- dates either validate or remain faithfully unresolved;
- empty fields survive serialization;
- the manifest fingerprint changes when an example, split, prompt, or target changes.

### Failure modes

- using filenames that contain the deceased’s name;
- treating a high-confidence provider output as truth;
- normalizing a label differently from a prediction;
- changing the system prompt between runs without versioning;
- supervising a confidence string that has no meaningful target;
- truncating the completion while leaving the prompt intact;
- putting document IDs into model-visible text.

### Platform connection

The existing review UI and `ground_truth_labels` table are the authoritative human-label boundary. Reuse that boundary. Do not create a parallel “ground truth” CSV whose provenance cannot be reconciled with the database.

## Chapter 2 — Build an evaluation suite that can say no

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** deterministic evaluator, frozen slices, and correctness fixtures

### The evaluation unit is the document

Fields within a document are correlated. Treating 13 fields as 13 independent samples produces overconfident uncertainty estimates. Compute field counts, but resample and compare at the document level.

### Parsing pipeline

Separate these outcomes:

1. request/model failure;
2. empty output;
3. invalid JSON;
4. valid JSON with wrong/missing keys;
5. valid schema with field errors;
6. exact record match.

Never drop invalid or missing predictions from the denominator. A model that fails to return output on hard documents must not appear more accurate.

### Core extraction metrics

For each field, after a frozen normalization policy:

- true positive: truth and prediction are non-empty and equal;
- false positive: prediction is non-empty when truth is empty, or disagrees with a non-empty truth;
- false negative: truth is non-empty and prediction is empty or wrong.

Aggregate:

$$
\text{precision}=\frac{TP}{TP+FP},
$$

$$
\text{recall}=\frac{TP}{TP+FN},
$$

$$
F_1=\frac{2PR}{P+R}.
$$

Also report:

- exact-record accuracy;
- schema-valid rate;
- prediction-missing rate;
- hallucinated-field rate on reviewed empty fields;
- per-field precision/recall/F1 and support;
- document count, not just field count.

### Important repository distinction

The current general variant evaluator penalizes a predicted non-empty value when reviewed truth is empty. The router-target scorer intentionally ignores blank truth fields because blank may mean unavailable supervision for routing. Therefore:

> Do not reuse router target metrics as the adapter’s hallucination metric.

They answer different questions.

### Normalization policy

Primary scoring should remain strict enough to catch entity corruption. Define explicit secondary normalization for analysis:

- trim/collapse whitespace;
- Unicode normalization if the project adopts it;
- case folding only for fields where case is non-semantic;
- parsed date equality after validating a source-supported date;
- controlled punctuation handling for titles/locations.

Report strict and normalized scores separately. Do not normalize away a rare-name spelling error.

### Required slices

Define slices before test evaluation:

- common versus rare surnames;
- umlaut/`ß`/non-ASCII names;
- clean versus noisy OCR;
- low versus high OCR/name confidence;
- complete versus sparse notices;
- dates with digits versus textual/ambiguous dates;
- multiple locations;
- titles/maiden names/nicknames;
- source/year or layout family where coverage exists;
- long inputs and truncation-risk examples.

Each slice report includes document count and populated-field support. Suppress or clearly mark conclusions from tiny slices.

### Paired OCR-noise robustness

For synthetic or approved examples, create clean/corrupted pairs from the same underlying truth. Score:

- clean performance;
- corrupted performance;
- per-document delta;
- error category introduced by corruption.

Paired evaluation isolates corruption effects better than comparing unrelated clean and noisy documents.

### Rare-entity metrics

Exact match is primary. Add diagnostic metrics:

- character edit distance;
- normalized edit distance;
- prefix/suffix preservation;
- copy fidelity for evidence-backed spans.

Do not substitute fuzzy matching for correctness. `Mayer` and `Mayr` may be different people.

### Date and relationship probes

Use deterministic synthetic records to test mechanisms under controlled truth:

- distinguish birth from death date;
- preserve day/month order;
- avoid inferring year from age unless task policy allows it;
- map relationship phrases without assigning relatives’ attributes to the deceased;
- abstain when a relationship or date is absent.

These probes diagnose reasoning/interface failures. They do not estimate natural production frequency.

### Grounding and provenance probes

For an optional answer-with-evidence format, require each extracted claim to cite a document ID and evidence span. Measure:

- citation presence;
- citation validity;
- evidence-span containment/overlap;
- whether the cited text supports the claim;
- unsupported-claim rate;
- correct abstention when support is absent.

Do not ask a model judge whether a citation “looks right” when deterministic span/source checks are possible.

### Bootstrap uncertainty

For a metric $m$, resample documents with replacement, recompute $m$ many times, and report percentile intervals. For comparing two models, resample the same document indices for both and compute paired differences:

$$
\Delta_b=m(A_b)-m(B_b).
$$

The interval is descriptive under the frozen sample. It does not repair biased or unrepresentative labels.

### Evaluator tests

Create hand-worked fixtures for:

- perfect prediction;
- all-empty prediction;
- hallucination on empty truth;
- wrong non-empty value counting as both FP and FN;
- invalid JSON;
- missing prediction;
- extra keys;
- date normalization;
- Unicode and whitespace differences;
- paired bootstrap preserving model pairing.

### Predict

1. Can field F1 rise while exact-record accuracy falls?
2. Can a model improve recall by hallucinating more fields?
3. Why can blank-ground-truth semantics change the apparent precision?
4. Why is one bootstrap sample per field invalid here?
5. Which metric would reveal a model that always emits valid JSON but invents occupations?

### Failure modes

- excluding invalid outputs;
- computing only micro-F1 and hiding rare fields;
- defining slices after inspecting test failures;
- changing normalization to improve a score;
- bootstrapping fields independently;
- using test repeatedly for prompt/rank/epoch selection;
- comparing models on different document subsets;
- rounding away meaningful paired differences.

## Chapter 3 — Freeze splits, baselines, and decision rules

**Classification:** Core — must do  
**Time:** 30 minutes  
**Artifact:** benchmark protocol and pre-registered decision table

### Split roles

- **Train:** may affect adapter weights.
- **Validation:** may affect model, prompt, rank, epoch, and checkpoint choices.
- **Test:** opened once after choices are frozen.
- **Challenge sets:** deterministic synthetic/counterfactual probes; report separately from natural test performance.

Keep duplicate families, OCR variants, templates, and related source/year groups together. A source-year hash split is a starting mechanism, not proof of representativeness.

### The 44-label constraint

Do not create three tiny subsets and report precise percentages. Choose one of two honest paths:

1. **Mechanism path:** use existing labels for development and evaluator validation; make no final product claim.
2. **Directional benchmark path:** freeze existing work as development, review a new held-out batch selected before seeing adapter outputs, and report wide uncertainty.

The production extension should grow the benchmark until critical slices have meaningful support. The correct number depends on error rates and decision risk; it is not “whatever is already labeled.”

### Baselines

Evaluate on identical inputs/settings:

- deterministic schema/rule baseline where applicable;
- selected pretrained model before adaptation;
- prompt-only few-shot baseline;
- current exact `text_current` variant;
- current exact `vlm_current` variant where image input is relevant;
- LoRA adapter;
- optionally full fine-tuning only for a small enough model and justified compute.

The VLM has different inputs and cost. Compare it as a system baseline, not an isolated language-model architecture contest.

### Pre-register a decision rule

Example structure:

```text
Primary metric:
  paired document-level field F1 difference versus base model

Hard gates:
  schema-valid rate >= ...
  hallucinated-empty-field rate <= ...
  no regression above ... on rare-name/date slices
  all privacy/provenance checks pass

Secondary metrics:
  exact-record accuracy, latency, adapter bytes, general-capability regression

Decision:
  reject / continue research / shadow-mode candidate
```

Choose actual thresholds from product risk and baseline behavior before opening test. Ellipses are not thresholds.

### Checkpoint

You are ready to fine-tune only when the dataset, evaluator, baselines, validation policy, and rejection rule exist. Training first and deciding success afterward is result shopping.

---

# Part II — Fine-Tuning Mechanics

## Chapter 4 — SFT is next-token learning with a selective loss

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** verified chat/prompt renderer and completion-only loss mask

### Pretraining versus SFT

The mathematical primitive is still causal next-token cross-entropy. The difference is the data distribution and which tokens contribute to loss.

Given prompt tokens $x$ and completion tokens $y$:

$$
\mathcal{L}_{SFT}
=
-\sum_{t\in\text{completion}}
\log p_\theta(y_t\mid x,y_{<t}).
$$

Prompt tokens provide context but usually receive ignored labels. Padding also receives ignored labels.

### Why completion-only loss

For extraction, the task is to generate the answer conditioned on the prompt. Training heavily on repeated system instructions wastes capacity and can dominate short completions. Completion-only loss directly optimizes output tokens.

Full-sequence loss is not inherently invalid, but it answers a different objective. Treat it as an ablation, not an accidental default.

### Visualize the exact token stream

For at least five examples, print a table:

| Position | Token ID | Decoded piece | Role | Label ID | Contributes to loss? |
|---:|---:|---|---|---:|---|

Inspect:

- system/user/assistant boundaries;
- BOS/EOS placement;
- completion start;
- JSON punctuation;
- padding;
- truncation;
- final EOS supervision.

Do not trust a trainer flag until this table proves the resulting labels.

### Chat template

Use the selected pretrained model’s documented template unless you are intentionally adapting a base model with a new template. Record:

- tokenizer revision;
- rendered template fingerprint;
- role markers;
- BOS/EOS behavior;
- generation prompt behavior;
- assistant/completion mask mechanism.

Training and inference must render the same contract. A correct adapter with the wrong chat template can appear broken.

### Prompt-completion versus conversational format

For this single-turn extraction task, prompt-completion format is easier to audit:

```text
prompt:
  task instruction + schema + OCR text + approved hint

completion:
  canonical JSON object
```

Conversational format is acceptable if the production interface is conversational and assistant-only masking is proven.

### Truncation policy

Never silently truncate the target. Measure token-length distributions before choosing `max_length`.

Options for overlength examples:

- reject and count them;
- truncate OCR input with an explicit marker while preserving the completion;
- use a longer supported context;
- build a document-selection/chunking policy outside this model.

Report coverage. A model evaluated only on short notices is not a solution for long notices.

### Packing

Packing can improve utilization by placing multiple examples in one sequence. Verify:

- example boundaries and EOS;
- labels remain completion-only;
- no target leaks into another example’s prompt through preprocessing;
- evaluation does not depend on training pack order.

Start without packing for the first correctness run.

### Predict

1. What happens if prompt tokens receive loss but evaluation changes the wording of the prompt?
2. What happens if the completion’s closing brace or EOS is truncated?
3. Why can mean training loss fall when most supervised tokens are repetitive JSON keys?
4. Should field values and schema punctuation be analyzed separately?
5. What happens if padding tokens are treated as ordinary targets?

### Required tests

- prompt labels are ignored under completion-only policy;
- every completion value token intended for supervision is active;
- padding is ignored;
- EOS is present and supervised according to policy;
- renderer is deterministic;
- inference prompt equals the training prompt prefix;
- overlength examples are counted and handled deterministically;
- one-example loss matches a manually masked cross-entropy calculation.

### Current maintained interface note

Maintained SFT libraries support standard/conversational and prompt-completion formats. Current TRL documentation describes completion-only loss for prompt-completion data and assistant-only loss for compatible chat templates. Pin library versions and test behavior; main-branch APIs can change.

## Chapter 5 — Implement LoRA before configuring it

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** LoRA linear layer with merge, parameter, and gradient audits

### The update hypothesis

For a pretrained linear weight:

$$
W_0\in\mathbb{R}^{d_{out}\times d_{in}},
$$

full fine-tuning learns an unrestricted update $\Delta W$ of the same shape. LoRA freezes $W_0$ and constrains the update to rank at most $r$:

$$
\Delta W=sBA,
$$

with:

$$
A\in\mathbb{R}^{r\times d_{in}},
\qquad
B\in\mathbb{R}^{d_{out}\times r},
\qquad
s=\frac{\alpha}{r}.
$$

The forward pass is:

$$
y=W_0x+sB(Ax)+b.
$$

Only $A$ and $B$ are trainable.

### Parameter reduction

Full update parameters:

$$
d_{out}d_{in}.
$$

LoRA update parameters:

$$
r(d_{in}+d_{out}).
$$

For a square $d\times d$ matrix, the ratio is approximately $2r/d$. Calculate this for the actual selected modules before training.

### Initialization

Initialize $A$ with small random values and $B=0$. Then:

$$
\Delta W=0
$$

at step zero, so the adapter initially reproduces the base model.

### Predict the first backward pass

With $B=0$:

1. Is the gradient of $B$ zero or nonzero?
2. Is the gradient of $A$ zero or nonzero?
3. After one optimizer step, what changes on the next backward pass?

Because the path into $A$ is multiplied by $B$, $A$ commonly receives zero gradient at the exact initial state while $B$ receives a signal through $Ax$. Once $B$ moves, both train.

### Skeleton

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        # Freeze base parameters.
        # A: [rank, in_features]
        # B: [out_features, rank]
        # scale = alpha / rank
        ...

    def forward(self, x):
        # base(x) + scale * ((dropout(x) @ A.T) @ B.T)
        ...

    def merged_weight(self):
        # base.weight + scale * (B @ A)
        ...
```

### Correctness tests

- step-zero output exactly/closely matches the base layer;
- base weights have `requires_grad=False` and receive no gradient;
- only adapter parameters appear in the optimizer;
- analytical trainable-parameter count matches `numel()`;
- unmerged forward equals a linear layer using merged weight;
- merge then unmerge recovers the original within dtype tolerance;
- save/load reproduces outputs;
- selected gradients match finite differences in float64;
- first-step $A/B$ gradient behavior matches the prediction;
- dropout is active only during training.

### Rank is capacity, not a quality dial with a universal optimum

Rank bounds the update matrix’s rank. Higher rank increases trainable parameters and possible update directions, but may overfit small data or add no useful capacity.

Run at most one bounded comparison, such as $r=4$ versus $r=16$, under identical:

- base model and target modules;
- data order;
- token budget;
- learning-rate policy;
- validation selection.

Do not search rank, alpha, dropout, targets, and learning rate simultaneously.

### Scaling alpha

$\alpha/r$ separates rank from update scale. Record both. “Rank 8 LoRA” is incomplete without alpha, target modules, dropout, and initialization.

### Target modules

Start from the model architecture, not guessed names. Print candidate linear modules with shapes. Common choices include attention query/value projections or all major linear projections, but no target set is universally optimal.

For each candidate set, calculate:

- trainable parameters;
- adapter bytes by dtype;
- percentage of base parameters adapted;
- coverage by layer and projection type.

### Merge semantics

At inference, the adapter can remain separate or be merged:

$$
W'=W_0+sBA.
$$

A correctly merged linear update adds no extra matrix multiplications at inference. Merging changes artifact identity and may be lossy when mixed dtypes or quantized weights are involved. Preserve base model revision and adapter metadata.

### Failure modes

- transposing $A$ or $B$ incorrectly;
- training the base weight accidentally;
- both $A$ and $B$ initialized to zero, preventing learning;
- forgetting $\alpha/r$;
- applying dropout to the base path;
- adapting modules whose names differ across model revisions;
- counting frozen parameters as trainable;
- merging twice;
- saving only merged weights without base/license/provenance metadata.

### Source connection

The original LoRA work freezes pretrained weights, trains a low-rank update, initializes one factor randomly and the other to zero, and scales the update by $\alpha/r$. It also motivates merging the update for no added inference latency. This chapter reproduces that mechanism before using a library.

## Chapter 6 — Use PEFT and understand QLoRA’s boundary

**Classification:** Core — must do  
**Time:** 60 minutes  
**Artifact:** pinned adapter configuration and memory/accounting report

### Why switch to a library now

Injecting LoRA into one known linear layer is educational. Reliably handling dozens of architecture families, tied parameters, serialization, adapter loading, mixed precision, and distributed hooks is maintenance work. Use a maintained PEFT implementation after your reference layer passes.

### Configuration audit

Before training, print and save:

- exact base model identifier and immutable revision;
- base versus instruct checkpoint type;
- tokenizer identifier/revision and chat-template fingerprint;
- LoRA rank, alpha, dropout, bias policy;
- exact resolved target-module names and shapes;
- modules intentionally saved outside LoRA, if any;
- total and trainable parameter counts;
- dtype of frozen weights, adapter weights, optimizer state, and compute;
- software/library versions.

Fail if zero modules or an unexpected number of modules are adapted.

### PEFT equivalence spot check

Choose one small layer and:

1. copy your reference $A/B$ weights into the library adapter or vice versa;
2. disable dropout;
3. compare the layer output;
4. compare merged and unmerged output;
5. compare trainable-parameter count.

You are validating your interpretation, not re-testing the entire library.

### Full fine-tuning, LoRA, and QLoRA

| Method | Base weights | Trainable weights | Main memory implication |
|---|---|---|---|
| Full fine-tuning | floating, updated | all selected base parameters | gradients and optimizer state for the full model |
| LoRA | floating, frozen | low-rank adapters | base activations still matter; optimizer state is adapter-sized |
| QLoRA | typically 4-bit stored, frozen | low-rank adapters in higher precision | reduces frozen-base storage; compute uses dequantized values |

QLoRA does **not** train integer base weights. Gradients flow through operations using a frozen quantized/dequantized base into the adapter parameters.

### QLoRA mechanisms to understand now

At a conceptual level:

- 4-bit NormalFloat targets normally distributed weight values;
- double quantization compresses quantization constants;
- paged optimizers address memory spikes;
- low-rank adapters remain the trainable path.

Implement none of these kernels here. Volume IV will study quantization numerically. In this volume, use QLoRA only if hardware memory requires it and evaluate it as a training configuration, not as “free compression.”

### Memory accounting

Before and during loading, record:

- process/GPU memory before model load;
- after frozen base load;
- after adapter injection;
- after optimizer creation;
- peak after first forward/backward;
- activation-memory sensitivity to sequence length and batch size.

Do not attribute all savings to trainable-parameter count. Activations can dominate, and optimizer/library implementations differ.

### Base versus instruct model

Use an instruct checkpoint for the core’s fastest path to a useful extraction interface. It already knows instruction/chat conventions. Keep the distinction visible:

- adapting a **base** model teaches both instruction behavior and task behavior;
- adapting an **instruct** model specializes existing instruction behavior.

Evaluate the frozen baseline either way. Do not claim the adapter created abilities already present in the base.

### Current maintained workflow

Current PEFT and TRL documentation supports injecting a `LoraConfig` into supervised fine-tuning and training prompt-completion or conversational data. Treat exact API names as versioned implementation details: pin versions, save configs, and verify the produced loss mask and target-module resolution.

### Failure modes

- selecting a model because it is popular rather than suitable/license-compatible;
- loading “latest” without a revision;
- assuming matching module suffixes mean intended coverage;
- enabling 4-bit loading and calling it QLoRA without adapter-only training;
- comparing LoRA and QLoRA with different base revisions or prompts;
- saving adapters without tokenizer/template identity;
- merging into quantized weights without understanding library semantics;
- ignoring base-model license and redistribution terms.

### Fine-tuning mechanics checkpoint

You are ready for the full run if you can:

- derive LoRA dimensions and parameter count;
- predict first-step $A/B$ gradients;
- prove library injection targets the intended modules;
- visualize the exact supervised loss mask;
- distinguish full fine-tuning, LoRA, and QLoRA memory/state;
- reload the base plus adapter and reproduce fixed logits.

---

# Part III — Bounded Adaptation and Evaluation

## Chapter 7 — Select a model and design one useful experiment

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** model-selection worksheet and frozen run plan

### Selection criteria

Do not select a model solely because it appears on a leaderboard. Score candidates on:

- local hardware fit for LoRA/QLoRA and later inference;
- German and multilingual competence measured on a fixed prompt panel;
- tokenizer behavior on rare names, dates, and OCR noise;
- context length relative to notice distribution;
- structured JSON reliability before fine-tuning;
- base/instruct availability;
- architecture support in maintained PEFT/runtime tools;
- license, redistribution, and commercial-use constraints;
- immutable revision availability;
- eventual quantized local-runtime support.

Choose one small-enough instruct model for the core. Record rejected candidates and reasons. If hardware is unknown, do not lock the textbook to a parameter size; calculate feasibility from actual memory measurements.

### Baseline prompt audit

Before adaptation, run the frozen candidate on:

- several reviewed development examples;
- every required synthetic challenge category;
- empty/irrelevant OCR;
- malformed/noisy OCR;
- overlength input.

Use deterministic decoding for extraction. Save raw responses locally and parse through the same evaluator used later.

### Training composition

A time-efficient core dataset can combine:

- reviewed real training examples;
- deterministic synthetic schema examples;
- controlled OCR corruption variants derived only from training examples;
- explicit negative/abstention examples;
- schema-format examples.

Do not generate variations from validation/test documents. Do not let synthetic volume overwhelm real task distribution without recording the mixture.

### One primary experiment

Question:

> Does LoRA SFT improve reviewed OCR-text extraction over the frozen pretrained model without increasing hallucination or degrading rare-entity and general-control suites?

Freeze:

- base revision and tokenizer;
- prompt/schema/template version;
- training records and mixture weights;
- LoRA targets/rank/alpha/dropout;
- optimizer, learning rate, token/epoch budget;
- maximum length and truncation policy;
- validation cadence and checkpoint rule;
- decoding settings;
- metric implementation and slices;
- seed policy.

### Hyperparameter discipline

Allow:

- one learning-rate pilot on training/development data;
- one rank comparison only if it answers a capacity question;
- early stopping on the primary validation metric with hallucination as a gate.

Do not perform a broad sweep on 44 labels. Hyperparameter flexibility is another form of model capacity and will overfit the benchmark.

### Predict

1. Which fields improve first: common dates/names or sparse occupations/places?
2. Will schema-valid rate improve before field accuracy?
3. Which slice is most vulnerable to overfitting?
4. Will adapter training harm the base model’s general instruction behavior?
5. What result would cause immediate rejection even if field F1 improves?

## Chapter 8 — Run SFT and diagnose adaptation, not just loss

**Classification:** Core — must do  
**Time:** 60 minutes active work plus bounded training wall time  
**Artifact:** reloadable adapter and training report

### Correctness run

Before the real run:

1. use 4–8 examples;
2. disable packing and adapter dropout;
3. inspect rendered sequences and masks;
4. overfit completions;
5. verify JSON generation under deterministic decoding;
6. save/reload adapter and reproduce logits;
7. verify base weights remain unchanged by checksum or selected tensor equality.

### Main run measurements

Log:

- supervised-token training and validation loss;
- token accuracy only as a diagnostic;
- schema-valid rate on development generation;
- field metrics and hallucination rate at evaluation cadence;
- learning rate and gradient norm;
- adapter update norm by layer/module;
- trainable parameter count and optimizer bytes;
- tokens/second and peak memory;
- truncation/rejection counts;
- checkpoint identity and stop reason.

Training loss is dominated by token frequency. Repeated punctuation and keys can improve while rare field values remain wrong. Generation-based task metrics are therefore required during validation.

### Adapter update diagnostics

For each adapted matrix, inspect:

- $\|sBA\|_F$;
- $\|sBA\|_F/(\|W_0\|_F+\epsilon)$;
- singular values of $BA$ for a few representative layers;
- gradient norms of $A$ and $B$;
- near-zero or anomalously large modules.

These do not prove semantic importance. They reveal dead adapters, scale explosions, and uneven adaptation.

### Overfitting signals

- train loss falls while validation field metrics flatten;
- schema becomes perfect but values worsen;
- outputs copy training-specific names/templates;
- rare-field recall rises with a larger hallucination increase;
- adapter update norms grow while general-control performance declines;
- later checkpoints become more verbose despite exact-format instruction.

### Catastrophic forgetting versus specialization

LoRA freezes base weights, but the active adapter can still alter behavior broadly. “Frozen base” means recoverability by disabling the adapter, not immunity from adapter-induced forgetting.

Evaluate:

- adapter disabled: must reproduce base baseline;
- adapter enabled on domain task;
- adapter enabled on a small general-control suite;
- optionally merged adapter: must match unmerged outputs within tolerance.

### Required broken labs

#### A. Prompt tokens included in loss

Compare mask statistics and behavior under a paraphrased prompt. Explain whether the model learned task output or memorized interface text.

#### B. Both LoRA factors initialized to zero

Observe zero gradients/learning in the low-rank branch. Preserve a regression test.

#### C. Train/test duplicate

Inject one duplicate into a toy fixture and show the score inflation. Do not contaminate the actual test set.

#### D. Blank-label hallucination ignored

Run an intentionally wrong metric that skips empty truth fields, then the correct metric. Record how model ranking can change.

### Stop rules

Stop when:

- the frozen token/epoch budget ends;
- primary validation metric stops improving under declared patience;
- hallucination exceeds the gate;
- update norms or gradients become non-finite;
- truncation/data errors invalidate the run;
- compute cap is reached.

Do not continue because training loss still decreases.

### Checkpoint selection

Select by a declared validation objective, for example:

$$
J = F_{1,field} - \lambda_h\,R_{hallucination},
$$

subject to schema and critical-slice gates. Choose $\lambda_h$ before seeing test. A multi-metric lexicographic rule is often clearer than pretending all costs combine naturally.

### Failure modes

- selecting the final step by default;
- evaluating with sampling enabled;
- changing generation length between baseline and adapter;
- reporting only valid parsed outputs;
- adapter silently not attached to intended modules;
- evaluating a merged model with different dtype/runtime;
- including validation examples in synthetic augmentation;
- saving private training text inside adapter metadata.

## Chapter 9 — Evaluate quality, robustness, grounding, and forgetting

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** paired benchmark report with uncertainty and error taxonomy

### Freeze inference behavior

For extraction comparison, keep identical:

- prompt and chat template;
- input fields;
- tokenizer/base revision;
- deterministic decoding policy;
- maximum output length;
- JSON constraint/repair policy;
- timeout/retry policy;
- evaluator and normalization.

If one system uses constrained decoding and another does not, report it as a system comparison and include schema-valid rate—not as an invisible model-only change.

### Paired model table

For each model/system, report:

| Metric | Base | Prompt-only | LoRA | Current text variant | Current VLM variant |
|---|---:|---:|---:|---:|---:|
| Documents attempted |  |  |  |  |  |
| Missing/error rate |  |  |  |  |  |
| Schema-valid rate |  |  |  |  |  |
| Exact-record accuracy |  |  |  |  |  |
| Field precision |  |  |  |  |  |
| Field recall |  |  |  |  |  |
| Field F1 |  |  |  |  |  |
| Hallucinated-empty-field rate |  |  |  |  |  |
| Rare-name exact match |  |  |  |  |  |
| Date exact/normalized match |  |  |  |  |  |
| OCR-noise paired delta |  |  |  |  |  |
| Latency/cost when measured |  |  |  |  |  |

Do not populate unavailable values with estimates.

### Paired error transitions

Aggregate metrics hide how behavior changed. Count documents moving:

- wrong → correct;
- correct → wrong;
- missing → predicted correct;
- empty truth → hallucinated;
- valid schema → invalid schema;
- exact rare name → corrupted rare name.

Manually inspect a blinded sample from each transition type.

### Error taxonomy

Tag errors without changing metrics:

- OCR source unreadable;
- name/hint conflict;
- field-role confusion;
- date swap or invalid normalization;
- relative’s attributes assigned to deceased;
- unsupported inference;
- rare-entity character corruption;
- output-format/schema failure;
- truncation;
- prompt refusal or irrelevant prose;
- candidate ground-truth ambiguity.

If review reveals label ambiguity, correct the benchmark through a versioned label update and rerun all systems. Do not silently fix only the favored model’s examples.

### Hallucination and abstention

Include challenge cases with absent fields and irrelevant/insufficient OCR. Measure:

- unsupported non-empty fields;
- appropriate empty outputs;
- refusal/abstention behavior under unreadable input;
- confidence calibration only if a meaningful confidence target is defined.

A model that fills every field may have high recall and be unusable.

### Grounded-answer suite

Create a small separate suite where the model receives retrieved snippets with stable source IDs and must answer with citations. Include:

- answer present in one source;
- conflicting sources;
- no supporting source;
- distractor with a similar name;
- OCR-corrupted evidence;
- answer requiring two supported facts.

Score deterministic citation IDs and claim support first. Use human review for nuanced support. Do not merge these results with extraction F1.

### General-control/forgetting suite

Keep it small and fixed:

- basic German/English instruction following;
- strict JSON on a non-domain schema;
- copying a supplied arbitrary identifier;
- simple arithmetic/date formatting;
- refusal to invent from absent context;
- base-model safety/control behavior relevant to local use.

Compare base with adapter enabled. The suite detects broad regressions; it is not a general intelligence benchmark.

### Human evaluation

For outputs not fully captured by exact metrics:

- blind model identity;
- randomize output order;
- use a concise rubric;
- collect independent ratings on a shared subset;
- record disagreement and adjudication;
- separate factual support, completeness, formatting, and preference.

“Which answer do you like?” is not a sufficient rubric for family-history facts.

### Model-as-judge

Use only when deterministic or human scoring is impractical. Before trusting it:

1. create human labels on a calibration subset;
2. compare judge agreement with humans;
3. randomize candidate order to test position bias;
4. hide model names;
5. test verbosity/style bias;
6. require evidence-based structured reasons;
7. report judge model/prompt/version;
8. never let judge score be the sole release gate.

Judge disagreement is a signal to inspect, not noise to average away.

### Contamination and test reuse

Record every time humans or models inspect test outputs. Once errors influence prompt, data, rank, checkpoint, normalization, or code, that set becomes development data. Version it honestly and acquire a new sealed test set for the next final claim.

### Statistical reporting

Report:

- counts and point estimates;
- document-level paired bootstrap intervals;
- per-slice support;
- the number of model/config choices tried;
- practical effect size, not only interval sign;
- unresolved label ambiguity.

With a small benchmark, “inconclusive” is often the correct result.

### Evaluation checkpoint

You are ready for the capstone if you can explain why:

- exact-record accuracy and field F1 answer different questions;
- blank-field handling controls hallucination measurement;
- paired comparisons are stronger than unrelated aggregate scores;
- a frozen adapter can still alter general behavior;
- model judges require calibration;
- repeated test inspection destroys the test role.

---

# Volume III Capstone — A useful adapter must survive rejection tests

**Classification:** Core — must do  
**Time:** 90 minutes  
**Artifact:** adapter package, regression suite, and evidence-based decision

## Capstone question

> Under a frozen prompt, dataset, and deterministic evaluator, does the LoRA adapter improve reviewed OCR-text extraction enough to justify shadow-mode investigation without increasing hallucination, rare-entity corruption, or general-control regressions?

## Required deliverables

### Dataset package

- prompt/completion renderer and template fingerprint;
- schema version and surname-field mapping test;
- reviewed/synthetic provenance counts;
- grouped split manifest;
- field/slice support table;
- leakage and truncation report;
- privacy/retention policy.

### Adapter package

- immutable base model and tokenizer revisions;
- LoRA rank, alpha, dropout, target modules, and bias policy;
- adapter weights and config;
- trainable/total parameter counts;
- training token budget and checkpoint rule;
- library/environment versions;
- reload and base-weight-freeze evidence;
- model/license notes.

### Evaluation package

- deterministic parser and schema validator;
- exact and normalized field metrics;
- blank-field hallucination metric;
- per-field and per-slice reports;
- paired bootstrap implementation;
- OCR corruption pairs;
- rare-entity/date/relationship challenge fixtures;
- grounding and general-control suites;
- versioned raw prediction records kept under the appropriate privacy policy.

### Failure analysis

Include the four required broken labs:

1. wrong SFT loss mask;
2. dead zero/zero LoRA initialization;
3. duplicate leakage;
4. hallucination-blind metric.

For each, report prediction, symptom, first failed invariant, root cause, and regression test.

## Decision matrix

| Outcome | Meaning | Next action |
|---|---|---|
| Reject | hard gate fails or core metric regresses | preserve evidence; change one justified factor on development data |
| Inconclusive | interval/data coverage too weak | expand reviewed benchmark, not rhetoric |
| Continue research | directional gains with unresolved slice/forgetting risk | targeted data/evaluation work |
| Shadow-mode candidate | all frozen gates pass with acceptable uncertainty | Volume IV inference benchmark, then reversible non-authoritative integration |

This volume does not authorize production replacement.

## Required report structure

```text
1. Decision and claim level
2. Task/schema/prompt contract
3. Data provenance and split
4. Base model selection
5. LoRA mechanism and configuration
6. Training behavior and stop reason
7. Primary paired results with uncertainty
8. Per-field and slice results
9. Hallucination and abstention
10. OCR robustness and rare entities
11. Grounding/provenance suite
12. General-control/forgetting suite
13. Human/judge agreement if used
14. Error transitions and taxonomy
15. Privacy, contamination, and limitations
16. Volume IV handoff
```

## Volume IV handoff

If an adapter survives the gates, the next questions are systems questions:

- What work is repeated during naive autoregressive generation?
- How do prefill and decode differ?
- Which keys and values can be cached?
- How does cache memory grow with layers, heads, context, batch, and dtype?
- What are time to first token and inter-token latency?
- What numerical error and quality loss result from INT8/INT4 quantization?
- Does the adapter remain correct after merge, quantization, or runtime conversion?
- Which operation is the profiler actually spending time on?

Do not implement those optimizations here. Save the frozen adapter, prompts, and regression suite; they are Volume IV’s quality oracle.

---

# Volume III Mastery Examination

## A. Explain without notes

You should be able to explain:

1. Why SFT uses the same next-token primitive as pretraining.
2. Why completion-only masking changes the optimization target.
3. Why chat-template and EOS identity are model behavior, not formatting trivia.
4. Why unreviewed teacher output is not ground truth.
5. The dimensions of $W_0$, $A$, $B$, and $BA$ in LoRA.
6. Why random $A$ plus zero $B$ preserves the base output.
7. Why $A$ may receive zero gradient on the first exact step while $B$ does not.
8. How LoRA trainable parameters compare with a full update.
9. What rank and alpha change.
10. Why LoRA can be merged without an extra inference operation.
11. How QLoRA differs from ordinary LoRA and post-training quantization.
12. Why frozen base weights do not prevent adapter-induced forgetting.
13. Why invalid/missing predictions stay in evaluation denominators.
14. How a wrong non-empty field contributes FP and FN.
15. Why blank reviewed fields are essential for hallucination measurement.
16. Why router metrics and extraction metrics have different blank-label semantics.
17. Why uncertainty is bootstrapped by document.
18. Why paired model differences are preferable.
19. Why exact match, normalized match, and fuzzy similarity must remain separate.
20. Why a model judge cannot be the sole release gate.
21. Why 44 reviewed rows support plumbing tests but not a broad production claim.

## B. Reconstruct from a blank editor

In one focused session:

1. serialize one canonical prompt/completion record;
2. create labels that mask prompt and padding tokens;
3. implement a LoRA linear layer;
4. prove base and step-zero adapter outputs match;
5. merge $BA$ into the base weight and compare outputs;
6. parse one model response into the canonical storage schema;
7. compute TP/FP/FN for three fields including a blank truth;
8. compute exact-record accuracy;
9. write a document-level paired bootstrap outline.

## C. Debug blind

Diagnose two injected bugs:

- completion boundary shifted by one token;
- EOS masked out unintentionally;
- both LoRA matrices initialized to zero;
- base weights accidentally trainable;
- target-module pattern matches no layers;
- adapter merged twice;
- missing predictions removed before scoring;
- blank truth fields skipped;
- normalization changes rare surnames;
- bootstrap resamples fields instead of documents;
- test data included in synthetic augmentation;
- adapter loaded with a different chat template.

Use invariants and minimal fixtures before reading implementation details.

## D. Design

Design an evaluation answering:

> Does adaptation improve extraction from noisy OCR without increasing unsupported facts for sparse notices?

Specify:

- reviewed and synthetic data sources;
- split/group unit;
- paired corruption generator;
- primary metric and hallucination gate;
- sparse-notice slice definition;
- baseline systems;
- decoding and parsing policy;
- uncertainty method;
- human-review sample;
- rejection rule;
- claim boundary.

## Passing standard

Proceed to Volume IV only when:

- the task formatter and loss mask are directly inspected and tested;
- your hand-built LoRA layer passes forward, gradient, merge, and serialization audits;
- the maintained PEFT run resolves the intended target modules;
- base, tokenizer, prompt, adapter, and dataset identities are frozen;
- a tiny subset can be overfit and the adapter reloads reproducibly;
- missing/invalid outputs and blank-field hallucinations are scored correctly;
- paired baseline/adapter predictions cover identical documents;
- critical slices and challenge sets were defined before test;
- general-control and grounding checks are reported separately from extraction F1;
- uncertainty and sample counts accompany headline metrics;
- the decision follows predeclared gates;
- private data and outputs remain under the declared policy.

If the adapter “looks better” but the evaluator cannot penalize an invented occupation, Volume III is not complete. If the adapter is rejected by a trustworthy test, the volume has done its job.

---

# Compact formula and metric reference

## SFT objective

$$
\mathcal{L}_{SFT}
=
-\sum_{t\in\text{completion}}
\log p_\theta(y_t\mid x,y_{<t}).
$$

## LoRA

$$
W'=W_0+\frac{\alpha}{r}BA,
$$

$$
A\in\mathbb{R}^{r\times d_{in}},
\qquad
B\in\mathbb{R}^{d_{out}\times r}.
$$

Trainable parameters per adapted linear layer:

$$
P_{LoRA}=r(d_{in}+d_{out}).
$$

## Field metrics

$$
P=\frac{TP}{TP+FP},
\qquad
R=\frac{TP}{TP+FN},
\qquad
F_1=\frac{2PR}{P+R}.
$$

## Hallucinated-empty-field rate

One explicit definition:

$$
R_{hallucination}
=
\frac{\#\{(d,f): y_{d,f}=\emptyset,\ \hat y_{d,f}\ne\emptyset\}}
{\#\{(d,f): y_{d,f}=\emptyset\}}.
$$

Use only fields whose empty reviewed truth means “confirmed absent,” not “unlabeled.”

## Exact-record accuracy

$$
A_{exact}
=
\frac1N\sum_{d=1}^{N}
\mathbf{1}[\hat y_d=y_d].
$$

## Paired metric difference

$$
\Delta=m(\text{adapter})-m(\text{base}),
$$

with bootstrap resampling performed over shared document indices.

## Adapter update scale

$$
R_{update}
=
\frac{\|\frac{\alpha}{r}BA\|_F}
{\|W_0\|_F+\epsilon}.
$$

---

# Run checklist

## Before formatting data

- [ ] task/schema version frozen;
- [ ] model-visible inputs match production availability;
- [ ] reviewed versus synthetic versus teacher provenance explicit;
- [ ] duplicate groups defined;
- [ ] privacy/retention policy recorded;
- [ ] storage/model surname mapping tested.

## Before training

- [ ] train/validation/test roles frozen;
- [ ] test sealed;
- [ ] baseline predictions captured;
- [ ] prompt/template/tokenizer revisions pinned;
- [ ] completion-only mask inspected;
- [ ] truncation coverage reported;
- [ ] target modules resolved and counted;
- [ ] step-zero adapter equals base;
- [ ] rejection gates written;
- [ ] tiny-batch overfit passes.

## During training

- [ ] supervised-token loss logged;
- [ ] development field metrics and hallucination logged;
- [ ] gradient and adapter-update norms checked;
- [ ] memory/throughput measured;
- [ ] truncation/data failures counted;
- [ ] checkpoint stop/selection rule followed;
- [ ] no test outputs inspected.

## Before final evaluation

- [ ] adapter checkpoint frozen;
- [ ] deterministic decoding frozen;
- [ ] parser/normalizer version frozen;
- [ ] baseline and adapter use identical document set;
- [ ] slices fixed;
- [ ] missing and invalid outputs retained;
- [ ] bootstrap seed and procedure recorded;
- [ ] human/judge rubric frozen if used.

## After evaluation

- [ ] counts, intervals, and effect sizes reported;
- [ ] per-field and per-slice support shown;
- [ ] hallucination and abstention explicit;
- [ ] error transitions inspected blind;
- [ ] label corrections versioned and applied to all systems;
- [ ] contamination/test inspections recorded;
- [ ] decision follows gates;
- [ ] raw private outputs remain protected.

---

# Optional Deep Dives — outside the 10-hour core

- full fine-tuning of a small pretrained model;
- rank/alpha/target-module factorial studies;
- alternative PEFT methods;
- preference optimization or RL-based post-training;
- continued domain pretraining before SFT;
- vision-language fine-tuning on source images;
- constrained decoding implementation;
- calibration models for field confidence;
- large human-evaluation studies;
- active learning for review selection;
- benchmark power/sample-size analysis;
- adapter composition and multi-adapter serving;
- safety red-teaming beyond the scoped local task.

Promote one only when the frozen benchmark identifies a concrete need.

---

# Primary references and implementation anchors

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — canonical low-rank update, initialization, scaling, and merging rationale.
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf) — frozen 4-bit base with trainable LoRA adapters, NF4, double quantization, and paged optimizers.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/main/index) — maintained adapter integration; pin a stable release rather than assuming main-branch behavior.
- [TRL SFTTrainer documentation](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md) — current dataset formats, prompt/completion loss behavior, chat templates, packing, and PEFT integration.

The papers define mechanisms and evidence from their experiments. They do not guarantee that a given rank, target-module set, quantization configuration, or trainer default is optimal for this project.

---

# End of Volume III

Stop here. Volume IV begins by profiling naive autoregressive generation, separating prefill from decode, implementing a KV cache, measuring cache growth, and then quantizing the frozen model while using this volume’s regression suite as the quality guardrail.
