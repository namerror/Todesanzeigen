# From OCR Pipeline to Owned Model Stack

## A compressed, implementation-driven ML systems curriculum for the Family History Intelligence Platform

This document contains exactly two things:

1. the architecture of the complete 50–70 hour core curriculum;
2. Volume I in full: reverse-mode autodiff, neural-network mechanics, and tiny language models.

It deliberately stops after Volume I. Tokenizer implementation, attention, Transformers, fine-tuning, inference, quantization, and production changes are specified in the architecture but not taught here yet.

The governing loop is:

> **Predict → Implement → Run → Observe → Explain → Improve**

Reading without running the experiments does not count as completing the course. Conversely, building every optional feature is not required. The objective is mechanistic ownership of the highest-value path through the model lifecycle.

---

# Curriculum Architecture

## 1. Starting from the platform that actually exists

This curriculum is grounded in the repository as inspected on 2026-09-15.

The platform is currently an OCR and structured-extraction system for German death notices. Its useful ML-system boundaries are already present:

- local Tesseract OCR produces text and layout-aware TSV artifacts;
- SQLite records documents, OCR outputs, model outputs, reviewed labels, experiment lineage, and dataset memberships;
- text-model and vision-model extraction paths coexist;
- an extraction variant is identified by method, provider, model, and prompt version rather than by a vague label such as “the LLM”;
- field-level evaluation, latency, token, and cost fields already exist;
- the router is formulated as predicting failure of the cheaper OCR-plus-text-model path and selecting a route under a quality/cost objective;
- existing router features are deliberately separated from raw labels and model outputs.

Current local state:

| Item | Count | Consequence for the course |
|---|---:|---|
| Documents | 1,168 | Enough real domain variation for corpus and robustness studies |
| OCR outputs | 1,168 | A ready source of noisy German domain text |
| Extraction outputs | 2,326 | Existing baselines can be compared by exact variant |
| Reviewed ground-truth labels | 44 | Enough to test evaluation plumbing; too few for broad quality claims |
| Feature snapshots | 0 | Router training is not yet a ready-made core exercise |
| Named dataset splits | 0 | A leakage-aware split must precede any defensible learned result |

That final distinction matters. A run can prove that code, gradients, serialization, or evaluation logic works. It cannot prove generalization from 44 labels. Throughout the course, every claim is labeled as one of:

- **mechanism claim:** the implementation behaves as the mathematics predicts;
- **experiment claim:** the result held for a frozen dataset and configuration;
- **product claim:** the result held on representative, held-out platform data under a product-relevant metric.

Volume I mostly earns mechanism claims. Later volumes earn experiment claims. Production integration requires product claims.

## 2. What is deliberately compressed

You already know Python, software architecture, applied PyTorch, basic ML, Linux, distributed computing, and experiment pipelines. The core therefore does not teach:

- Python syntax or object-oriented design;
- generic calculus review detached from code;
- generic REST, database, Git, or Linux workflows;
- every historical neural architecture;
- a broad survey of every fine-tuning or serving technique;
- custom kernels or distributed model parallelism before a measured need exists.

It does not compress the mechanisms on which later reasoning depends:

- reverse-mode autodiff and gradient accumulation;
- stable softmax and cross-entropy;
- initialization, activations, residual paths, and training diagnostics;
- tokenization and causal attention;
- autoregressive decoding and KV caching;
- quantization error and memory accounting;
- evaluation design and leakage control.

## 3. The core path: 56–64 active hours

“Active” means reading with notes, implementing, running, inspecting, and debugging. Passive background reading is not included. The schedule has a 56-hour nominal path and up to roughly 8 hours of bounded contingency. Optional deep dives do not count toward the core.

| Volume | Core topics | Nominal active work | Exit artifact | Cumulative milestone |
|---|---|---:|---|---|
| I. Gradients and tiny language models | scalar autograd; MLP; stable loss; count, neural-bigram, and context models; diagnostics | 12 h | tested scalar engine, tiny LM comparison, four-bug diagnosis report | ~12 h: first-principles NN understanding |
| II. Tokens and Transformers | byte-aware BPE; tokenizer audit; causal attention; multi-head attention; normalization; residuals; decoder-only Transformer; small training run | 15 h | custom tokenizer, tested Transformer, checkpoint and samples | ~27 h: functioning Transformer |
| III. Fine-tuning and evaluation | LoRA mechanism; production PEFT workflow; SFT; data formatting; leakage; field, provenance, rare-name, OCR-noise, and grounded-answering evaluation | 10 h | useful adapter plus versioned regression suite | ~37 h: useful fine-tuned model and credible evaluation |
| IV. Inference, quantization, and profiling | naive generation; prefill/decode; manual KV cache; batching concepts; INT8/INT4 mechanics; production quantizer; basic GPU profiling; one data-parallel training-step trace | 12 h | cached and quantized inference benchmark with quality deltas plus compute/memory/communication accounting | ~49 h: owned local inference path |
| V. Platform integration | component selection; local serving boundary; retrieval/model division of labor; routing; provenance; fallback; monitoring; end-to-end acceptance test | 7 h | quantized/profiled model integrated behind a reversible platform boundary | ~56 h: complete lifecycle in the real system |
| Contingency | debugging or one high-value “Useful” experiment per volume | 0–8 h | deeper evidence where the actual work needs it | ~56–64 h total |

The ranges requested for individual topics are respected inside these volumes:

| Topic budget | Core allocation |
|---|---:|
| Autograd and neural-network mechanics | 5.5–6.5 h |
| Tiny language models and training behavior | 5.5–6.5 h |
| Tokenization | 3–4 h |
| Transformer from scratch | 11–12 h |
| Fine-tuning and evaluation | 9–10 h |
| Inference and quantization | 9–11 h |
| Basic profiling and distributed/performance reasoning | 2–3 h, embedded in inference and integration |
| Production integration | 6–8 h |

The architecture is intentionally not a 12-week calendar. A week is not a unit of understanding; an executable checkpoint is.

## 4. Stop-safe milestones

Stopping early must still leave something worth keeping.

### After approximately 6 hours

You have a scalar reverse-mode autodiff engine, numerical gradient checks, and a tiny MLP. You can explain what `backward()` computes and why gradients accumulate.

### After approximately 12 hours

You have count and neural language-model baselines, stable cross-entropy, embeddings, a context MLP, and a repeatable debugging method. You can reason about optimization from internal signals rather than only loss.

### After approximately 27 hours

You have a custom tokenizer and a decoder-only Transformer whose major operations, shapes, masks, and gradients you implemented and tested. This is the first complete-model milestone.

### After approximately 37 hours

You have a useful fine-tuned model and an evaluation suite that can detect regressions in extraction fidelity, rare names, dates, OCR noise, provenance, and grounded answers.

### After approximately 49 hours

You can account for decode-time work and memory, demonstrate a correct KV cache, quantify the effect of INT8/INT4 compression, and identify bottlenecks with a profiler.

### After approximately 56–64 hours

A selected local model runs behind a reversible interface in the Family History Intelligence Platform. Its quality, latency, memory, provenance, and fallback behavior are measured against existing variants.

The core distributed objective is deliberately narrow: trace one data-parallel training step, account for gradient all-reduce bytes, and explain why two devices are not automatically twice as fast. Running the same experiment on two devices is useful when hardware is available, but tensor/pipeline/model-parallel implementations remain optional.

## 5. Dependency graph

```text
document boundaries + privacy + frozen splits
                     │
                     ▼
local derivatives → dynamic graph → reverse-mode accumulation
                     │
                     ▼
parameters → MLP → optimizer behavior → diagnostics
                     │
                     ▼
probabilities → stable cross-entropy → tiny language models
                     │
                     ▼
character inefficiency → BPE tokenizer → tokenizer audits
                     │
                     ▼
embeddings + matmul + softmax + normalization + residuals
                     │
                     ▼
causal attention → multi-head attention → Transformer training
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   LoRA/SFT mechanics     naive autoregressive decode
          │                     │
          ▼                     ▼
 rigorous evaluation       KV cache + batching
          └──────────┬──────────┘
                     ▼
          quantization + profiling
                     │
                     ▼
 retrieval/model routing + local serving + provenance
```

The order is not decorative. KV caching comes after attention because a cache is stored attention state. Quantization comes after distribution inspection because scales and clipping act on distributions. Fine-tuning comes after pretraining because otherwise LoRA is merely an API recipe. Production integration comes after evaluation because without a rejection test, integration is guesswork.

## 6. Manual-versus-library boundary

Use the smallest implementation that exposes the mechanism, then stop rebuilding infrastructure once the abstraction has been earned.

| Component | Level | Decision |
|---|---:|---|
| Scalar reverse-mode autodiff | 1 — first principles | Implement completely in Volume I |
| Neuron, MLP, SGD | 1 | Implement with the scalar engine |
| Momentum and Adam | 2 — implement once, then use library | One compact implementation and equivalence checks |
| Stable softmax/cross-entropy | 1 then 2 | Implement math, test against PyTorch, then use framework primitive |
| Embedding lookup | 1 then 2 | Demonstrate one-hot equivalence, then use `Embedding` |
| BPE | 1 | Implement the learning and encoding loops in Volume II |
| Transformer block | 1 then 2 | Build component-by-component, then compare/use optimized kernels |
| LoRA | 1 then 3 | Implement low-rank update once, then use a maintained PEFT library |
| KV cache | 1 then 3 | Build a transparent cache, then use a serving engine |
| Quantization | 1 then 3 | Quantize small tensors manually, then use production tooling |
| Plotting, serialization, database, OCR | 3 — production library/existing code | Reimplementation has poor learning return |
| Custom CUDA/Triton | Optional Deep Dive | Only after profiling identifies a specific operation |
| Tensor/pipeline parallelism | Optional Deep Dive | Conceptual core only unless hardware/project need appears |

## 7. The five-volume artifact chain

Artifacts are designed to feed forward rather than become abandoned toy projects.

```text
Volume I
  scalar gradients + diagnostics + domain character corpus
       │
       ▼
Volume II
  tokenizer + Transformer + checkpointed trainer
       │
       ▼
Volume III
  adapter + versioned domain evaluation suite
       │
       ▼
Volume IV
  cached/quantized model + latency/memory/quality benchmark
       │
       ▼
Volume V
  local inference adapter + routing/fallback/provenance integration
```

Educational implementations should live beside, not inside, the production package until they pass their exit tests. A future implementation workspace might be `learning/`, but this document does not create it or change production code.

## 8. Data strategy for this domain

The platform data is valuable precisely because it is difficult: German names, dates, places, honorifics, historical spellings, Unicode, OCR substitutions, layout loss, and long-tail entities. It is also private or sensitive enough that convenience is not a data policy.

Use three tiers:

1. **Synthetic public-safe corpus.** Deterministically generated notices with invented identities, controlled templates, and injected OCR noise. This is the default for committed educational code and fixtures.
2. **Redacted local corpus.** Approved OCR excerpts with names and identifying details replaced in a length- and character-aware way. Store locally; do not assume it is safe to publish.
3. **Private local corpus.** Raw OCR and reviewed records. Use only when the experiment needs genuine distributions. Never put examples, generations, checkpoints, or logs into version control without an explicit review.

The split unit is a document or a correlated document group, never a random character window. Near duplicates, shared templates, source/year groups, and OCR variants of the same image must stay in one subset. A model that sees the first 90% of a notice during training has not generalized to its last 10% during validation.

## 9. Measurement spine

Instrumentation grows only when it answers a live question.

| Stage | Minimum evidence |
|---|---|
| Scalar autodiff | analytic gradient, central-difference gradient, relative error, branch-reuse test |
| MLP | loss, parameter/gradient/update norms, one-batch overfit, PyTorch equivalence |
| Tiny LM | train/validation NLL, bits per character, slice NLL, samples, parameter count |
| Transformer | plus activation statistics, tokens/s, peak memory, checkpoint reproducibility |
| Fine-tuning | plus task metrics, forgetting, per-slice confidence intervals, provenance accuracy |
| Inference | time to first token, inter-token latency, throughput, cache bytes, batch sensitivity |
| Quantization | model bytes, numerical error, latency, throughput, quality delta by slice |
| Distributed reasoning | gradient bytes, collective time, compute/communication overlap, scaling efficiency |
| Integration | end-to-end latency, route rate, failure rate, cost, fallback rate, provenance coverage |

Do not introduce a tracking server in Volume I. A configuration file, append-only metrics file, and short report are enough.

## 10. Optional Deep Dives outside the core budget

These are valuable, but they are not prerequisites for the platform outcome:

- forward-mode autodiff, higher-order derivatives, and custom backward graphs;
- deep optimizer theory beyond behavior needed to debug training;
- custom CUDA allocators or kernels;
- extensive Triton programming;
- GPU warp scheduling and occupancy tuning beyond profiler interpretation;
- tensor, pipeline, and expert parallelism implementations;
- large-scale distributed checkpointing and serving control planes;
- speculative decoding, paged attention internals, or continuous-batching implementation.

Promote one into the core only when a measured platform bottleneck or research objective makes it necessary.

---

# Volume I — Gradients, Neural Networks, and Tiny Language Models

## Volume contract

Right now, `loss.backward()` can be useful without being legible. This volume earns the right to use it as an abstraction.

At the end, you should be able to:

- draw the dynamic computation graph of a scalar expression;
- distinguish a local derivative from the total derivative of the loss;
- explain reverse-mode autodiff as repeated vector–Jacobian products;
- implement reverse topological traversal and gradient accumulation;
- verify gradients numerically and against PyTorch;
- train a tiny MLP and interpret optimizer behavior;
- derive stable softmax cross-entropy and its logit gradient;
- explain the relationship among counts, logits, one-hot vectors, and embeddings;
- train count, neural-bigram, and context-MLP character models;
- diagnose stale gradients, bad initialization, unstable softmax, data leakage, and shape errors;
- state exactly why the next volume needs a tokenizer and causal attention.

### Core timebox

| Block | Time |
|---|---:|
| Chapters 1–4: autograd and NN mechanics | 5.5–6.5 h |
| Chapters 5–9: tiny LMs and training behavior | 5.5–6.5 h |
| Total | 11–13 h |

If an implementation is still broken at the upper bound, use the hint ladder and preserve the bug as a test. Do not silently spend four extra hours polishing an educational framework.

### Exit artifact

One small, separate educational package containing:

- a scalar autograd engine;
- finite-difference and PyTorch gradient audits;
- neuron/layer/MLP primitives and SGD;
- a deterministic synthetic Family History corpus manifest;
- count-bigram, neural-bigram, and context-MLP models;
- a compact experiment ledger;
- a report diagnosing four controlled failures.

The artifact is successful even if generated text is poor. Its job is to make the mechanics inspectable and to produce measured reasons for tokenization and attention.

## Working protocol

For every experiment, write down before running:

1. the question;
2. the prediction;
3. the independent correctness oracle;
4. the one variable being changed;
5. the metric that would falsify the prediction.

Use a minimal run record:

```json
{
  "run_id": "v1-mlp-context3-seed17",
  "question": "Does context length 3 improve date-format prediction over a bigram?",
  "prediction": "Validation NLL falls most on the date slice.",
  "seed": 17,
  "dataset_fingerprint": "...",
  "split_id": "synthetic-v2-grouped",
  "model": {"kind": "context_mlp", "context": 3, "embedding_dim": 16},
  "optimizer": {"kind": "sgd", "lr": 0.1},
  "status": "planned"
}
```

Do not log raw private text. Metrics and aggregate slice labels are enough for the core.

---

# Part I — Reverse-Mode Autodiff

## Chapter 1 — Local derivatives are executable contracts

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** finite-difference probes for scalar functions

### Why this exists

A derivative rule in an autograd engine is not a symbolic decoration. It is a local software contract: given an upstream sensitivity, how much sensitivity reaches each input?

Start with a scalar function:

$$
f(x)=x^2+3x-4.
$$

Its derivative is:

$$
f'(x)=2x+3.
$$

At $x=2$, the derivative is 7. This means a small perturbation $\Delta x$ should change the output by approximately $7\Delta x$. The derivative is a local linear model of the program.

### Predict

Before running anything, answer:

1. For $h=10^{-2},10^{-5},10^{-12}$, which central-difference estimate will be best in double precision?
2. If $g(x)=f(x)^2$, is $g'(2)$ just $2f(2)$, just $f'(2)$, or their product?
3. If the same variable appears twice in `x * x`, should its two gradient contributions replace or add to each other?

### Mathematics connected to code

The central-difference approximation is:

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
$$

Large $h$ has truncation error because the local linear approximation spans too wide a region. Tiny $h$ has floating-point cancellation because the two function values become nearly equal. The numerical gradient is an oracle, not a definition and not infallible.

For a composition $L=g(f(x))$:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial f}
\frac{\partial f}{\partial x}.
$$

Read it as: “upstream sensitivity times local sensitivity.” That sentence becomes the implementation of every backward rule.

### Implement

Write this interface yourself:

```python
def central_difference(f, x: float, h: float = 1e-6) -> float:
    """Return a centered finite-difference estimate of df/dx."""
    # TODO
```

Then create probes for:

- $x^2$;
- $x^2+3x-4$;
- $\tanh(x)$;
- $\log(x)$ at positive values;
- the composed function $(x^2+3x-4)^2$.

### Tests

Use relative error:

$$
\operatorname{relerr}(a,b)=
\frac{|a-b|}{\max(1,|a|,|b|)}.
$$

Require approximately `1e-6` or better for smooth, well-scaled double-precision probes. Do not choose the tolerance first and then cherry-pick `h`; inspect an `h` sweep.

### Expected observations

- central differences generally beat one-sided differences at the same moderate `h`;
- error falls as `h` shrinks, then rises when cancellation dominates;
- the derivative of a composition multiplies local sensitivities;
- a repeated input creates multiple paths whose contributions must add.

### Failure modes

- testing at a nondifferentiable point and treating disagreement as an engine bug;
- using integer arithmetic;
- confusing absolute with relative error near large gradients;
- choosing `h` so small that both perturbed outputs round to the same value.

### Checkpoint

Continue only if you can explain why finite differences are independent enough to test autograd, yet too expensive and noisy to train a network.

## Chapter 2 — Build the graph while the program runs

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** scalar `Value` graph with local backward closures

### Why this exists now

To apply the chain rule to an arbitrary expression, the system must remember what operation produced each value and which earlier values it used. A dynamic graph records the execution that actually happened.

For:

```python
a = Value(2.0)
b = Value(-3.0)
c = a * b
d = c + a
```

the graph is:

```text
a ───────┐
│        ▼
├──► (*) c ──► (+) d
│    ▲          ▲
│    │          │
└────┼──────────┘
     b
```

Notice that `a` has two paths to `d`. That shape is why assignment in a backward rule is wrong; gradient contributions must accumulate.

### Minimal interface

```python
class Value:
    def __init__(self, data, children=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        # TODO: coerce scalar, create output, attach closure
        ...

    def __mul__(self, other):
        # TODO
        ...

    def __neg__(self): ...
    def __sub__(self, other): ...
    def __truediv__(self, other): ...
    def __pow__(self, exponent): ...
    def tanh(self): ...
    def exp(self): ...
    def log(self): ...
```

Delay `backward()` until the local rules work.

### Local rules

For $z=x+y$:

$$
\frac{\partial z}{\partial x}=1,
\qquad
\frac{\partial z}{\partial y}=1.
$$

Therefore the closure performs:

```text
x.grad += z.grad
y.grad += z.grad
```

For $z=xy$:

$$
\frac{\partial z}{\partial x}=y,
\qquad
\frac{\partial z}{\partial y}=x.
$$

Therefore:

```text
x.grad += y.data * z.grad
y.grad += x.data * z.grad
```

The closure reads current input data and current output gradient. It does not recursively call parent closures. Traversal is a separate responsibility.

### Predict

For `d = a*b + a` with `a=2`, `b=-3`, calculate by hand:

- `d.data`;
- $\partial d/\partial a$;
- $\partial d/\partial b$.

Then predict the wrong value of `a.grad` if the `+` closure overwrites the contribution from `*`.

### Implementation steps

1. Coerce Python numbers to `Value` objects inside operators.
2. Compute forward data immediately.
3. Record parents and operation name.
4. Attach a closure that applies only the local derivative.
5. Use `+=` for every gradient contribution.
6. Add reflected operators so `2 + value` and `2 * value` behave correctly.
7. Keep the engine scalar. Broadcasting and tensor storage are explicitly out of scope.

### Hint ladder

**Hint 1:** The output owns the closure because backward begins from the output and asks how it was produced.

**Hint 2:** Capture `self`, `other`, and `out` in each closure. Update parent gradients using `out.grad`.

**Hint 3:** For multiplication, the closure is structurally:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

### Required tests

- forward arithmetic against plain Python;
- mixed `Value`/number arithmetic;
- `x + x` gives gradient 2;
- `x * x` gives gradient `2*x`;
- `a*b + a` preserves both paths;
- `tanh`, `exp`, and `log` agree with finite differences at safe points;
- division and powers handle the domain you claim to support.

### Likely bugs

- using a list of parents and accidentally traversing duplicate nodes as if they were distinct objects;
- using a set but defining equality without a matching identity hash;
- late-binding loop variables in closures;
- implementing subtraction separately and introducing a sign bug;
- mutating `data` after graph construction, so the closure no longer represents the forward pass.

### What to inspect

Print a small graph as rows containing `label`, `data`, `grad`, `op`, and parent labels. You do not need a graph-visualization package. The purpose is to verify identity and connectivity.

### Platform connection

The production extraction graph is not a differentiable graph, but the lineage idea is the same: outputs are only interpretable when you know which inputs and operations produced them. The repository’s exact extraction-variant identity is the experiment-level analogue of preserving graph parents.

## Chapter 3 — Reverse traversal and gradient accumulation

**Classification:** Core — must do  
**Time:** 90 minutes  
**Artifact:** complete reverse pass plus randomized gradient audit

### Why reverse mode

A neural network has many parameters and one scalar loss. Forward-mode differentiation would propagate one parameter direction at a time. Reverse mode starts from the one output and computes sensitivity with respect to every reachable parameter in one reverse traversal.

If a node $x$ influences loss through children $y_1,\dots,y_k$:

$$
\frac{\partial L}{\partial x}
=
\sum_{j=1}^{k}
\frac{\partial L}{\partial y_j}
\frac{\partial y_j}{\partial x}.
$$

The sum is not an implementation detail. It is the multivariable chain rule at a branch.

### Why order matters

Suppose `x -> y -> z -> L`. The closure for `y` needs the complete `y.grad`. That gradient is not complete until every downstream path has contributed. Therefore downstream nodes run before upstream nodes: reverse topological order.

### Implement

Add:

```python
def backward(self):
    # 1. build a topological order by DFS
    # 2. seed output sensitivity to 1
    # 3. execute closures in reverse order
    ...
```

The seed is 1 because $\partial L/\partial L=1$. A more general vector output would need a supplied upstream vector; your scalar engine deliberately avoids that API.

### Predict before running

1. What happens if closures execute in forward topological order?
2. What happens if `self.grad` is not seeded?
3. What happens when `backward()` is called twice without clearing gradients?
4. Should intermediate-node gradients be cleared between independent training steps, or only parameter gradients?

### Gradient audit

Construct a deterministic graph with branching and nonlinearities, for example:

$$
L=\tanh(a b+c)+a^2+\log(1+e^b).
$$

For each leaf:

1. evaluate analytic gradient with your engine;
2. rebuild the graph at $x+h$ and $x-h$;
3. compute central difference;
4. report absolute and relative error.

Rebuild the graph for perturbations. Mutating a leaf after the forward pass invalidates saved intermediate values.

### Randomized testing

Generate dozens of small expression trees from safe operations. Keep inputs away from invalid log/division domains. Compare each leaf gradient with finite differences. Save the seed on failure so the graph is reproducible.

The test is stronger if operations are reused and nodes branch. A chain-only test misses the most important accumulation bug.

### PyTorch equivalence

Recreate three graphs with `torch.float64` and compare:

- scalar output;
- every leaf gradient;
- behavior under shared nodes.

Use float64 because the goal is derivative agreement, not accelerator throughput.

### Deliberately broken lab

Change exactly one `+=` to `=`. Do not inspect which test fails first. Predict which graph structures still pass and which fail. Then design the smallest test that detects the bug.

Expected result: chains may pass because each node receives one contribution. Branches fail. This is a general debugging lesson: a test suite that lacks the structural condition behind a rule can certify the wrong program.

### Memory mental model

Dynamic autograd stores enough forward information to execute local backward rules. The graph lives until it is released. Large tensor frameworks additionally save tensors, version counters, device metadata, and optimized kernel context. This is why training generally uses more memory than inference.

### Checkpoint questions

Explain without notes:

- why reverse mode fits scalar-loss training;
- why the output gradient is seeded with 1;
- why gradients add at branches;
- why reverse topological order is necessary;
- why a second backward pass accumulates unless gradients are cleared;
- why finite differences require rebuilding the forward computation.

## Chapter 4 — Parameters, MLPs, and controlled motion

**Classification:** Core — must do  
**Time:** 2.5–3 hours  
**Artifact:** scalar MLP trained with SGD and audited against PyTorch

### Objective

Turn the graph engine into a trainable neural network and make every state transition in a training step explicit.

### Architecture

A neuron computes:

$$
z=\sum_{i=1}^{n}w_i x_i+b,
\qquad
a=\tanh(z).
$$

A layer evaluates several neurons on the same input. An MLP composes layers.

```text
x[0:n]
  │
  ▼
Linear(n, h) → tanh
  │ [h]
  ▼
Linear(h, 1)
  │
  ▼
prediction scalar
```

### Interfaces to write

```python
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, n_in, *, nonlinear=True, rng=None): ...
    def __call__(self, x): ...
    def parameters(self): ...

class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs): ...
    def __call__(self, x): ...
    def parameters(self): ...

class MLP(Module):
    def __init__(self, n_in, widths, **kwargs): ...
    def __call__(self, x): ...
    def parameters(self): ...
```

Use an explicit random-number generator. Hidden global randomness makes equivalence and debugging harder.

### Parameter identity

Parameters are leaf values selected for optimization. Deduplicate by object identity if parameter traversal can expose shared weights. Two parameters with equal numeric values are not the same parameter. One shared object appearing twice must be updated once using its accumulated gradient.

### Predict: symmetry

If all neurons in a layer start with identical weights and biases, will they diverge during training? Explain the gradient symmetry before running.

Expected observation: neurons receiving identical inputs and downstream treatment receive identical gradients. They remain duplicates. Random initialization is not merely “to add noise”; it breaks parameter symmetry.

### First task

Use a four-example binary toy problem. Do not use platform records yet. The purpose is to prove mechanics with a dataset whose behavior can be completely inspected.

Use mean squared error initially:

$$
L=\frac{1}{N}\sum_i(\hat y_i-y_i)^2.
$$

One training step is exactly:

```text
1. forward: predictions = model(inputs)
2. loss: combine predictions and targets
3. clear: set every parameter gradient to zero
4. backward: fill gradients by reverse-mode autodiff
5. update: parameter.data -= learning_rate * parameter.grad
```

Reorder these mentally and predict the failure. In particular, clearing after backward erases the update signal; omitting clearing adds gradients from previous steps.

### Measurements

At each selected step, log:

- loss;
- parameter norm $\|\theta\|_2$;
- gradient norm $\|g\|_2$;
- update norm $\|\eta g\|_2$;
- update-to-parameter ratio $\|\eta g\|/(\|\theta\|+\epsilon)$;
- min/max prediction.

The ratio is diagnostic, not a universal target. Huge ratios indicate destructive movement; vanishing ratios indicate nearly frozen parameters.

### Learning-rate experiment

Clone the same initial parameters and train with three rates:

- a clearly small rate;
- a plausible rate;
- a rate large enough to destabilize learning.

Before running, predict the loss curve and update ratio for each. Keep initialization and example order identical.

Expected patterns:

- small: smooth but slow progress;
- plausible: rapid initial decrease and eventual flattening;
- too large: oscillation, saturation, divergence, or non-finite values.

### Gradient clipping experiment

Implement global norm clipping only as an experiment:

$$
g \leftarrow g\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).
$$

Clipping can prevent one destructive step. It does not fix a wrong derivative, bad target, or systematically excessive learning rate. Record how often clipping activates; a permanently active clip is evidence, not success.

### Momentum and Adam in one compact pass

SGD follows the current gradient. Momentum keeps an exponentially weighted direction:

$$
v_t=\beta v_{t-1}+(1-\beta)g_t,
\qquad
\theta_t=\theta_{t-1}-\eta v_t.
$$

Adam keeps first- and second-moment estimates:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

Because both start at zero, early estimates are biased toward zero. Correct them:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

then update:

$$
\theta_t=\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Implement these state updates once for a small parameter vector. Feed a constant gradient and calculate the first two steps by hand. Compare every state value and parameter update with a framework optimizer configured to the same convention. Libraries differ in details such as weight decay and epsilon placement, so state the convention you test.

Predictions:

1. Without bias correction, are the first Adam steps too large or too small under a constant gradient?
2. What additional bytes of optimizer state does Adam require per parameter in this simplified full-precision case?
3. Why is gradient accumulation from omitted `zero_grad()` not equivalent to momentum?

Do not spend time tuning all three optimizers on every tiny experiment. After this equivalence exercise, use SGD when its simplicity makes behavior clearer and Adam when later Transformer training needs its adaptive scaling.

### PyTorch audit

Build the same small MLP in PyTorch with float64. Copy weights exactly. For one fixed batch compare:

1. outputs;
2. scalar loss;
3. every parameter gradient;
4. one SGD update.

Match parameter order explicitly. “The lists had the same length” is not proof that corresponding values represent the same weights.

### What PyTorch now safely hides

After the audit, you may use PyTorch autograd for tensor models. You have earned the abstraction because you can state what it provides:

- tensor-valued dynamic graphs;
- broadcasting and reduction backward rules;
- device and dtype dispatch;
- saved tensors and graph-lifetime management;
- efficient kernels;
- in-place mutation checks;
- vector–Jacobian products.

Do not build tensor storage, a dispatcher, or a kernel system. Those are important, but not the shortest path to the current learning objective.

### Required failure labs

1. **Stale gradients:** remove `zero_grad()`. Observe gradient norm and loss. Explain why this is not ordinary momentum.
2. **Saturated activation:** multiply initial weights by 20. Inspect preactivations, tanh outputs, and gradients.
3. **Symmetric neurons:** initialize each neuron identically. Show that their outputs and gradients remain equal.
4. **Detached parameter:** replace one `Value` with its `.data` in the forward pass. Identify the missing graph edge.

### Mastery checkpoint: first-principles neural networks

You are ready for language models if you can, without notes:

- implement `+`, `*`, `tanh`, and `backward()`;
- explain every term in a multiplication backward rule;
- show a branch where gradients must add;
- train the four-example MLP;
- identify whether a failed run first goes wrong in forward values, loss, gradients, or update;
- match a PyTorch forward pass and gradients numerically.

This is the approximately six-hour stop-safe artifact.

---

# Part II — Tiny Language Models

## Chapter 5 — Build a privacy-safe prediction corpus

**Classification:** Core — must do  
**Time:** 30 minutes  
**Artifact:** deterministic grouped corpus plus manifest

### Why characters first

Tokenization is a major topic, but introducing it now would entangle two mechanisms: learning conditional probabilities and deciding the units over which probabilities are defined. Characters are inefficient but transparent. Their inefficiency will become measured evidence for Volume II.

### Synthetic notice generator

Generate invented records from controlled components:

- given name and surname lists that contain no real family data;
- German date formats;
- invented towns;
- short notice templates;
- optional title, maiden name, occupation, and relationship phrases;
- OCR corruptions such as `rn↔m`, `1↔l`, dropped punctuation, merged spaces, broken umlauts, and missing or incorrect date digits.

Each record should include metadata but the language model sees only text:

```json
{
  "document_id": "syn-template03-family017-002",
  "group_id": "template03-family017",
  "text": "...",
  "slices": ["date", "umlaut", "ocr-rn-to-m"],
  "privacy": "synthetic",
  "generator_version": "v2"
}
```

### Split rule

Hash `group_id` plus a fixed seed into train, validation, and test. All records from the same template/family group stay together. Freeze the assignment before model work.

Do not randomly split character windows. That leaks nearly identical context across subsets.

### Boundary contract

Add explicit beginning/end behavior. One simple design uses a sentinel character not otherwise present. Never construct examples across document boundaries.

For a document with characters `abc`, the bigram events are:

```text
<BOS> → a
a     → b
b     → c
c     → <EOS>
```

You may use one shared sentinel for both boundaries in a bigram model, but separate tokens make semantics clearer and transfer better to later autoregressive code.

### Manifest

Record:

- generator version and configuration;
- seed;
- document and character counts by subset;
- vocabulary and its ordering;
- group split rule;
- slice counts;
- content fingerprint;
- privacy class.

Vocabulary order is model identity. Sorting by Unicode code point is simple and deterministic. Changing the order while loading a checkpoint silently changes every embedding and output class.

### Predict

1. Will a random-window split produce better validation NLL than a grouped split?
2. Which OCR corruption increases vocabulary size most?
3. Will characters make rare surnames easier or harder to represent than a fixed word vocabulary?
4. Which is longer: character sequences or future BPE sequences, and what downstream operation pays for that length?

### Tests

- deterministic generation under the same seed;
- no `group_id` appears in more than one subset;
- every validation/test character is representable;
- no training event crosses document boundaries;
- corpus fingerprint changes when source text or split assignment changes;
- private paths and text are absent from the manifest.

### Platform connection

The real OCR corpus can later be exported through a read-only, explicitly approved path. For Volume I, synthetic data isolates mechanics and prevents a checkpoint or sample log from reproducing private identities.

## Chapter 6 — The count bigram is baseline and oracle

**Classification:** Core — must do  
**Time:** 60 minutes  
**Artifact:** smoothed count model with exact NLL

### Objective

Estimate the next-character distribution without gradients. This model is both a baseline and an oracle for the neural bigram.

Let $N_{ij}$ be the number of times character $j$ follows character $i$. With additive smoothing $\alpha$:

$$
p(j\mid i)=
\frac{N_{ij}+\alpha}{\sum_k(N_{ik}+\alpha)}.
$$

Tensor shapes:

- count matrix `N`: `[V, V]`;
- one input character index: scalar;
- one probability row: `[V]`;
- a batch of input indices: `[B]`;
- selected next-character probabilities: `[B]`.

### Predict

For a vocabulary of size $V$:

1. What is the uniform-model NLL? Answer: derive it before reading on.
2. What happens to unseen validation bigrams when $\alpha=0$?
3. As $\alpha\to\infty$, which baseline does the model approach?
4. Why can training NLL worsen while validation NLL improves as smoothing increases from zero?

The uniform model assigns $1/V$, so one event contributes $-\log(1/V)=\log V$ nats. That is your first loss-at-initialization oracle.

### Implement

Write:

```python
class CountBigram:
    def __init__(self, vocab_size: int, alpha: float): ...
    def fit(self, documents: list[list[int]]): ...
    def log_probs(self) -> Tensor: ...       # [V, V]
    def nll(self, documents) -> float: ...
    def sample(self, generator, max_length): ...
```

It is fine to use PyTorch tensors. Do not use autograd.

### Stable NLL

For events $(x_b,y_b)$:

$$
L=-\frac1B\sum_{b=1}^{B}\log p(y_b\mid x_b).
$$

Report nats per character and bits per character:

$$
\mathrm{bpc}=L/\log 2.
$$

Perplexity $e^L$ is mathematically valid, but bits per character is easier to interpret for character models. Do not compare perplexity across tokenizers with different units without qualification.

### Hand-worked fixture

Use two or three tiny strings and count every transition on paper. Test exact counts, row sums, and selected probabilities. If the model cannot reproduce a hand-counted fixture, larger-corpus metrics are meaningless.

### Experiments

Sweep $\alpha$ over `{0, 0.01, 0.1, 1, 10}`.

Record:

- train and validation NLL;
- number of zero-probability validation events;
- per-slice NLL for dates and OCR corruption;
- average generated length and termination rate.

Keep test sealed.

### Expected observations

- unigram should beat uniform because character frequencies are unequal;
- bigram should beat unigram on local spelling and formatting;
- zero smoothing may assign infinite validation loss;
- too much smoothing moves rows toward uniform;
- generated text can be locally plausible while globally incoherent.

### Failure modes

- normalizing columns instead of rows;
- omitting boundary events;
- crossing document boundaries;
- silently dropping unseen events when computing NLL;
- sampling from counts while evaluating smoothed probabilities;
- inspecting test results during smoothing selection.

### Checkpoint

Explain why the count model “learns” without gradients and why it nevertheless defines the target behavior of a correctly optimized neural bigram.

## Chapter 7 — Logits, stable cross-entropy, and the neural bigram

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** manual cross-entropy and neural-bigram equivalence study

### Why logits

A neural model emits unconstrained scores $z\in\mathbb{R}^V$, called logits. Softmax turns them into probabilities:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

The naive formula overflows for large positive logits. Subtracting $m=\max_j z_j$ changes neither probabilities nor loss:

$$
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}.
$$

Cross-entropy for target class $y$ is:

$$
L=-\log p_y
=-z_y+\log\sum_j e^{z_j}.
$$

Use the stable log-sum-exp form:

$$
\operatorname{LSE}(z)=m+\log\sum_j e^{z_j-m}.
$$

### Derive the gradient

For each logit $z_i$:

$$
\frac{\partial L}{\partial z_i}
=p_i-\mathbf{1}[i=y].
$$

Interpretation:

- every incorrect class receives a positive gradient proportional to assigned probability, so gradient descent lowers it;
- the correct class receives $p_y-1$, so gradient descent raises it unless probability is already one;
- gradients sum to zero across classes, matching softmax invariance to a common logit shift.

### Predict

1. If all logits are zero, what is the loss?
2. If the correct-class probability is `0.01`, what NLL does that event contribute?
3. What happens to probabilities if 100 is added to every logit?
4. Which incorrect class gets the largest gradient?
5. Why does confident wrong prediction produce a stronger correction than uncertain wrong prediction?

### Implement twice

Version A, for one vector:

```python
def stable_cross_entropy(logits, target):
    # logits: [V]
    # return scalar loss without calling framework cross_entropy
    ...
```

Version B, for a batch:

```python
def batch_cross_entropy(logits, targets):
    # logits: [B, V], targets: [B]
    # reduce over classes, then mean over examples
    ...
```

Compare forward values and gradients with the framework primitive in float64, including extreme logits such as `[1000, 0, -1000]`.

### Neural bigram

Let $W\in\mathbb{R}^{V\times V}$. For input character index $x$, logits are row $W_x$.

Using one-hot vector $e_x$:

$$
z=e_xW=W_x.
$$

This is an embedding lookup whose embedding width happens to equal the number of output classes.

Tensor shapes:

```text
x                 [B]
one_hot(x)        [B, V]
W                 [V, V]
logits            [B, V]
targets           [B]
loss              []
```

### Equivalence experiment

Train `W` on the same events as the count model. With enough optimization and compatible regularization/smoothing, each learned row should approach the empirical conditional distribution.

Compare rows using:

- maximum probability difference;
- KL divergence where both distributions have support;
- top-five next-character ranking;
- train/validation NLL.

The point is not that neural training beats counting. It is that gradient descent can recover a table that counting estimates directly. Later models share parameters across contexts where direct counting becomes sparse.

### Initialization experiment

Use identical data and compare:

- zero/small logits;
- moderate random logits;
- very large random logits.

Predict initial loss and gradient behavior. Large logits create confident arbitrary predictions and may yield a high initial loss. Small logits begin near the uniform-loss oracle.

### Tests

- softmax rows sum to one;
- common logit shifts do not change loss;
- extreme logits stay finite;
- manual and framework loss agree;
- manual and framework gradients agree;
- one-hot matrix multiplication equals row lookup;
- a tiny batch can be overfit.

### Likely bugs

- subtracting one global maximum across the batch instead of one per row;
- gathering along the wrong dimension;
- averaging logits before computing per-example loss;
- passing probabilities into a function that expects logits;
- applying softmax twice;
- using the wrong target shift.

### What abstraction is now safe

After equivalence tests, use framework cross-entropy. Keep logits unnormalized at the model boundary. The production primitive is safer and fused; the mathematics is no longer hidden from you.

## Chapter 8 — Embeddings and a context MLP

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** wider-context character model with explicit shape ledger

### Why the bigram is insufficient

A bigram knows only the previous character. It cannot reliably distinguish context such as:

- a digit after `19` versus after a space;
- `sch` from other uses of `s`;
- whether a period closes a date, abbreviation, or sentence;
- long local patterns in names and places.

Use a fixed context of $T$ previous characters.

### Architecture

Let:

- vocabulary size $V$;
- embedding width $C$;
- context length $T$;
- hidden width $H$;
- batch size $B$.

Parameters:

- embedding table $E\in\mathbb{R}^{V\times C}$;
- first-layer weights $W_1\in\mathbb{R}^{TC\times H}$;
- bias $b_1\in\mathbb{R}^{H}$;
- output weights $W_2\in\mathbb{R}^{H\times V}$;
- bias $b_2\in\mathbb{R}^{V}$.

Forward pass:

```text
context indices x                 [B, T]
        │ embedding lookup
        ▼
embeddings                        [B, T, C]
        │ flatten context/features
        ▼
flat                             [B, T*C]
        │ @ W1 + b1
        ▼
preactivation                    [B, H]
        │ tanh
        ▼
hidden                           [B, H]
        │ @ W2 + b2
        ▼
logits                           [B, V]
        │ cross entropy(targets [B])
        ▼
loss                             []
```

Before coding, derive the parameter count:

$$
VC+(TC)H+H+HV+V.
$$

### Dataset construction

For each document, initialize a context filled with the BOS index. For each character plus EOS:

1. emit current context as input;
2. emit current character as target;
3. shift context left and append current character.

Reset at every document boundary.

### Skeleton

```python
class ContextMLP(nn.Module):
    def __init__(self, vocab_size, context, embedding_dim, hidden_dim):
        super().__init__()
        # TODO: E, W1, b1, W2, b2

    def forward(self, x):
        # x: [B, T]
        # TODO: lookup -> [B,T,C]
        # TODO: reshape -> [B,T*C]
        # TODO: linear -> tanh -> logits [B,V]
        ...
```

Write parameters explicitly before replacing them with `Embedding` and `Linear`. This keeps shapes and initialization visible.

### Predict

1. Which parameters receive gradients for one batch: all embedding rows or only rows indexed by the batch?
2. If context order is accidentally reversed, can training loss still decrease?
3. If `T` grows while `H` is fixed, which parameter matrix grows?
4. Why does flattening bind different parameters to different relative positions?
5. Which structures should improve first over a bigram: dates, common suffixes, or whole-document coherence?

### Initialization

If inputs have unit-scale independent components and a linear layer has fan-in $n$, choose weight standard deviation on the order of:

$$
1/\sqrt{n}.
$$

This keeps preactivation variance from growing with fan-in. For tanh, inspect rather than worship a formula: the desired scale depends on actual embedding statistics and depth.

Initialize the output layer relatively small so early logits are near uniform. Then initial cross-entropy should be near $\log V$. If it is dramatically larger, inspect output-logit scale before tuning the optimizer.

### Training loop

Use deterministic minibatch sampling. At a modest cadence evaluate full train and validation NLL under `no_grad()`. Separate:

- training-mode stochastic update loss;
- evaluation-mode full-subset loss;
- final sealed test loss.

Do not call the current minibatch loss “training-set performance.”

### Minimal experiment matrix

Hold total update steps approximately constant and compare:

| Variable | Values | Primary question |
|---|---|---|
| Context length | 1, 3, 8 | What gains come from wider fixed context? |
| Initialization scale | 0.1×, 1×, 10× | How do activations and initial loss change? |
| Hidden width | one small, one medium | Is the model capacity-limited or data-limited? |

Do not launch a combinatorial grid. Three controlled comparisons answer more than 30 underinterpreted runs.

### Diagnostics

At initialization and selected steps, record:

- mean, standard deviation, min, max of preactivations;
- fraction of `|tanh_output| > 0.97`;
- gradient norm per parameter tensor;
- parameter norm and update/parameter ratio;
- logit standard deviation;
- train and validation NLL;
- examples/second or characters/second.

Use hooks only after you can collect the same values directly from this explicit forward pass.

### Correctness tests

- every intermediate shape matches the ledger;
- a one-example batch works;
- context construction resets at boundaries;
- one-hot embedding multiplication equals lookup;
- manual and module-based forward passes agree after copying parameters;
- manual and framework cross-entropy agree;
- the model overfits a tiny batch;
- checkpoint reload reproduces fixed-batch logits exactly on the same device/dtype.

### Expected observations

- context 1 should reproduce neural-bigram-like limitations;
- wider context should help structured local sequences, especially dates and frequent phrases;
- it should not create robust document-level semantics;
- large initialization should saturate tanh and create uneven gradients;
- training loss can continue falling after validation loss stops improving;
- a random-window split looks suspiciously better than a grouped split.

### Failure modes

- flattening `[B,T,C]` with the wrong permutation;
- target sequence shifted by zero or two positions instead of one;
- evaluating with training data or an unfrozen vocabulary;
- reinitializing parameters during evaluation;
- saving weights without vocabulary/split identity;
- reporting a lucky single seed as an architecture result.

### Platform connection

This model is not an extraction replacement. Its useful platform role is diagnostic: compare NLL slices for clean versus corrupted OCR, dates, umlauts, and rare character sequences. The result identifies representation and robustness problems that later tokenizer and Transformer work must solve.

## Chapter 9 — Diagnose the first bad state, not the final bad loss

**Classification:** Core — must do  
**Time:** 45–60 minutes  
**Artifact:** four controlled bug reports and regression tests

### Debugging ladder

Use this order:

1. **Freeze randomness.** Same seed, batch, order, and initial state.
2. **Check data contracts.** Print a few index/text/context/target examples.
3. **Check expected initial loss.** Near-uniform logits imply approximately $\log V$.
4. **Prove the forward pass.** Shapes, finite values, independent manual calculation.
5. **Prove the loss.** Manual versus framework on fixed logits.
6. **Prove gradients.** Finite difference or framework equivalence on a tiny case.
7. **Prove the update.** Check sign and magnitude for one parameter.
8. **Only then tune optimization.** Learning rate is not the first suspect when targets are misaligned.

### Required sanity tests

#### Overfit one batch

A sufficiently expressive model should drive one small batch to very low loss. Failure suggests a bug, insufficient capacity, destructive regularization, or poor optimization. Success proves only that the path can fit; it does not prove generalization.

#### Zero-learning-rate test

With learning rate zero, parameters and fixed-batch logits must remain bitwise unchanged under deterministic execution. If metrics move, hidden state or data sampling is changing.

#### Shuffled-target test

Training loss may fall by memorization, but validation should not improve systematically. If it does, inspect leakage or metric logic.

#### Initial-loss test

Near-zero logits should produce NLL near $\log V$. A large deviation localizes the problem to logits, targets, or loss—before the optimizer has acted.

### Four broken runs

Run and diagnose each. Do not reveal the fix to yourself until you have captured the first bad statistic.

#### A. Stale gradients

Remove gradient clearing.

Predict: Are accumulated gradients equivalent to increasing batch size? Why not, given that parameters change between steps?

Inspect: total gradient norm and update/parameter ratio.

#### B. Unstable softmax

Replace stable cross-entropy with direct exponentiation and multiply logits by 100.

Inspect: first non-finite tensor and whether the framework primitive remains finite.

#### C. Reversed or leaked context

Either include the target inside the context or cross document boundaries.

Inspect: suspiciously low initial/validation loss and decoded examples. This is the required data/evaluation bug.

#### D. Saturated tanh

Multiply first-layer initialization by 20.

Inspect: preactivation standard deviation, saturation fraction, hidden gradient, and output gradient. Distinguish a healthy large output gradient from a blocked gradient reaching earlier layers.

### Bug-report template

For each failure, record:

```text
Prediction:
Visible symptom:
First bad internal statistic:
Smallest reproducer:
Root cause:
Why loss alone was or was not sufficient:
Regression test:
General lesson for later Transformer training:
```

### Mastery checkpoint

Given a run with flat loss, you should be able to propose an ordered diagnostic procedure rather than a list of random hyperparameters.

---

# Volume I Capstone — The smallest model stack you fully own

**Classification:** Core — must do  
**Time:** 60 minutes of integration/reporting after chapter work  
**Artifact:** reproducible comparison and readiness decision for Volume II

## Question

> How much local structure in privacy-safe German death-notice text is captured by count bigrams, neural bigrams, and a fixed-context MLP—and which measured failures justify a tokenizer and causal attention?

## Required systems

Compare:

1. uniform next-character baseline;
2. unigram baseline;
3. smoothed count bigram;
4. neural bigram;
5. context MLP with at least two context lengths.

Do not add LSTMs, CNNs, attention, or BPE. Those dilute the dependency lesson.

## Freeze before the final run

- corpus generator version and fingerprint;
- grouped train/validation/test split;
- vocabulary and boundary symbols;
- model configurations;
- optimizer and stop rule;
- validation selection metric;
- slice definitions;
- random seeds;
- qualitative sampling prompts/prefixes.

## Predictions to write first

1. Expected uniform NLL from $V$.
2. Expected ordering of validation NLL across baselines.
3. Which slice benefits most from context length.
4. Which slice remains hardest.
5. Whether count and neural bigram rows converge closely.
6. Which initialization produces the highest saturation.
7. How much worse grouped validation will be than a deliberately leaky split.

## Required evidence

### Correctness

- scalar primitives pass central-difference checks;
- shared-node gradients pass branch tests;
- scalar MLP matches PyTorch on a fixed batch;
- bigram fixture counts are exact;
- probability rows sum to one;
- stable loss survives extreme logits;
- manual and framework cross-entropy gradients agree;
- context model overfits one tiny batch;
- checkpoint reload reproduces fixed logits.

### Results

Report for each model:

- parameter count;
- train and validation NLL;
- bits per character;
- date, Unicode/umlaut, rare-sequence, and OCR-noise slice NLL;
- training throughput;
- checkpoint size;
- termination rate and a fixed sample panel.

Open the test subset only after choosing smoothing, context length, widths, stop step, and seed-reporting policy. Test is for the final frozen comparison, not iteration.

### Training behavior

Include plots or compact tables for:

- loss versus step;
- gradient norm versus step;
- first-layer saturation fraction at initialization and later;
- update/parameter ratio;
- train/validation gap.

### Failure analysis

Include all four required bug reports. At least one must end in a data-contract regression test.

## Interpretation guide

Expected, but not guaranteed:

- uniform is worst;
- unigram improves by learning marginal character frequency;
- count bigram improves local transitions;
- neural bigram approaches the count model if optimization and smoothing/regularization are compatible;
- context MLP improves dates, suffixes, and common phrase fragments;
- grouped validation is worse and more honest;
- OCR-noise and rare-sequence slices remain difficult;
- samples become locally plausible without becoming factually meaningful.

If your ordering differs, first test correctness. If correctness holds, explain the data regime. A measured surprise is a result; an unexplained surprise is unfinished work.

## Boundary comparison with the platform

| Capability | Volume I model | Current platform | Decision |
|---|---|---|---|
| Local character regularity | Measured directly | Implicit in OCR/provider models | Keep as diagnostic evidence |
| Structured field extraction | Not established | OCR+LLM and VLM variants exist | Do not replace production extraction |
| Rare-name preservation | Character/slice NLL only | Reviewed field metrics can measure it later | Build a dedicated Volume III slice |
| Cost-aware routing | Not an LM task | Router objective and telemetry already exist | Revisit after labels/features/splits are adequate |
| Provenance-grounded answers | No grounding mechanism | Document and output lineage exist | Preserve existing lineage boundary |

## Why Volume II is now necessary

End the report with measurements, not slogans:

- **Character sequences are long.** Record character length and estimate attention sequence-length cost; this motivates learned subword units.
- **Fixed context truncates history.** Show examples where the required cue lies outside $T$; this motivates causal attention.
- **Flattening ties parameters to positions.** Explain why `W1` treats each context position separately; this motivates reusable token interactions.
- **Global NLL hides harms.** Show at least one slice whose trend differs from aggregate NLL; carry slice evaluation forward.
- **Training instrumentation matters.** Preserve the initial-loss, tiny-batch, gradient, and checkpoint tests for the Transformer.

Do not begin BPE or attention in this artifact. The measured limitations are the handoff.

---

# Volume I Mastery Examination

## Explain without notes

You should be able to answer:

1. What is the difference between a local derivative and total loss sensitivity?
2. Why is reverse mode efficient for a scalar loss with many parameters?
3. Why must reverse traversal be topological?
4. Why do gradients add at branches and across batch examples?
5. Why does `zero_grad()` exist?
6. Why can finite differences fail at very large or very small `h`?
7. Why is `x*x` a critical autograd test?
8. What exactly makes a leaf a parameter?
9. Why do identical neurons remain identical?
10. How do learning rate and gradient scale determine update size?
11. Why is stable log-sum-exp mathematically equivalent to naive softmax?
12. Why is the cross-entropy logit gradient `p - one_hot`?
13. How is a neural bigram related to a count table?
14. How is embedding lookup related to one-hot matrix multiplication?
15. What is every shape in the context MLP?
16. Why does large initialization saturate tanh?
17. Why does a grouped split usually look worse than a random-window split?
18. Why does better character NLL not imply better extraction accuracy?
19. What does PyTorch autograd hide that your scalar engine does not implement?
20. What measured limitation requires tokenization and what measured limitation requires attention?

## Implement from a blank editor

In one focused session:

1. implement a `Value` with addition, multiplication, tanh, and reverse traversal;
2. write a shared-node gradient test;
3. build one neuron and a two-layer MLP;
4. train it on four examples;
5. implement stable cross-entropy from logits;
6. show one-hot lookup equals direct row indexing;
7. write the tensor-shape ledger for a context MLP.

You may check function names, but not copy a finished implementation.

## Debug blind

Have your future self or another person inject two of:

- `=` instead of `+=` in one backward rule;
- forward-order closure execution;
- omitted gradient clearing;
- target included in context;
- softmax over the batch axis;
- excessively large tanh initialization;
- detached embedding lookup;
- context carried across document boundaries.

Diagnose with probes before reading the offending line.

## Passing standard

Proceed to Volume II only when:

- all correctness tests pass;
- finite-difference and PyTorch audits agree within declared tolerances;
- you can overfit one tiny batch;
- you diagnosed all four required controlled failures;
- count and neural bigram behavior are empirically connected;
- the context model’s run is reproducible from its manifest;
- the test subset remained sealed until final selection;
- no private content entered committable artifacts;
- the capstone contains measured reasons for BPE and causal attention.

If the models run but you cannot localize a broken gradient, Volume I is not complete. If a model fails and you isolate the first incorrect state, fix it, and preserve the regression test, that is exactly the competence this volume is designed to build.

---

# Compact reference

## Core equations

Branching chain rule:

$$
\frac{\partial L}{\partial x}
=
\sum_y
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}.
$$

Mean squared error:

$$
L=\frac1N\sum_i(\hat y_i-y_i)^2,
\qquad
\frac{\partial L}{\partial\hat y_i}=\frac{2}{N}(\hat y_i-y_i).
$$

Stable softmax:

$$
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad m=\max_j z_j.
$$

Cross-entropy:

$$
L=-z_y+\operatorname{LSE}(z),
\qquad
\frac{\partial L}{\partial z_i}=p_i-\mathbf1[i=y].
$$

SGD:

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

Global norm clipping:

$$
g\leftarrow g\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).
$$

Fan-in scale:

$$
\operatorname{Std}(W)\approx\frac1{\sqrt{\text{fan-in}}}.
$$

Language-model NLL:

$$
L=-\frac1N\sum_{t=1}^{N}\log p(x_t\mid x_{<t}).
$$

Bits per character:

$$
\mathrm{bpc}=L_{\text{nats}}/\log2.
$$

## Run checklist

Before:

- [ ] one-sentence question;
- [ ] written prediction;
- [ ] independent oracle;
- [ ] corpus fingerprint and split identity;
- [ ] seed, dtype, device, and configuration;
- [ ] expected initial loss;
- [ ] selection metric and stop rule;
- [ ] test subset sealed.

During:

- [ ] data examples decoded and checked;
- [ ] shapes match ledger;
- [ ] values and gradients finite;
- [ ] gradient and update norms plausible;
- [ ] fixed diagnostic batch preserved;
- [ ] first failing state captured.

After:

- [ ] result compared with prediction and baseline;
- [ ] validation and slice metrics reported;
- [ ] close comparisons checked across seeds;
- [ ] checkpoint reload tested;
- [ ] every fixed bug has a regression test;
- [ ] claim boundary stated;
- [ ] no private raw text appears in logs or samples.

---

# End of Volume I

Stop here. Volume II begins only after the passing standard, with measured character-sequence inefficiency, a byte-aware BPE implementation, tokenizer audits on German names/dates/OCR noise, and then causal attention.
