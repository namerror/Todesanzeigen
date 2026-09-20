# From OCR Pipeline to Owned Model Stack

## Volume II — Tokenization, Attention, and Domain Pretraining

Volume I ended with three measured limitations:

1. character sequences are transparent but unnecessarily long;
2. a fixed-context MLP cannot use evidence outside its hard window;
3. flattening context assigns different parameters to different positions instead of learning a reusable interaction between tokens.

This volume resolves those limitations in that order. First you will build a reversible byte-level BPE tokenizer and audit what it does to German names, dates, Unicode, and OCR corruption. Then you will build causal self-attention from explicit matrix operations, assemble a decoder-only Transformer, and train a small domain model with a checkpointable, inspectable training loop.

The volume stops there. It does not fine-tune a modern pretrained model, build the full domain evaluation suite, implement KV caching, quantize weights, or modify the production extraction path. Those belong to later volumes because they depend on the model and measurements built here.

The working loop remains:

> **Predict → Implement → Run → Observe → Explain → Improve**

---

# Volume contract

## What you should already own

Do not start by copying a Transformer implementation. Begin only when the Volume I passing standard is substantially true. In particular, you should be able to:

- explain reverse-mode autodiff and gradient accumulation;
- use PyTorch autograd without treating it as magic;
- derive stable cross-entropy and predict the uniform-loss baseline;
- trace the shapes in an embedding-based context MLP;
- overfit one tiny batch;
- inspect activations, gradients, and update magnitudes;
- preserve document boundaries, split identity, and vocabulary identity;
- diagnose a data bug before tuning the optimizer.

If one of those is weak, keep moving but return to the corresponding Volume I checkpoint when it becomes a blocker. Do not spend this volume rebuilding scalar autograd.

## Core timebox: 15 hours

| Part | Core work | Time |
|---|---|---:|
| I. Tokenization | byte contract, BPE learning, encoding/decoding, domain audit | 3.5 h |
| II. Attention | positional information, single-head and multi-head causal attention | 3.5 h |
| III. Transformer | normalization, residuals, feed-forward path, complete model | 3 h |
| IV. Domain pretraining | trainer, diagnostics, broken labs, bounded training run | 3 h |
| Capstone and mastery check | frozen comparison, report, reconstruction | 2 h |
| **Total** |  | **15 h** |

The training job’s wall-clock duration depends on hardware and is not permission to expand the architecture indefinitely. Choose a model that completes useful experiments within the timebox.

## Exit artifact

By the end, you should have a separate educational package containing:

- a reversible byte-level BPE tokenizer;
- a tokenizer manifest and domain audit report;
- a transparent single-head attention implementation;
- a multi-head causal self-attention module with a fused-QKV equivalence test;
- a pre-normalized decoder-only Transformer;
- deterministic dataset packing and validation;
- a checkpointable trainer with AdamW, clipping, and a learning-rate schedule;
- one small domain-model checkpoint;
- a report covering quality, activation/gradient behavior, throughput, memory, and controlled failures.

A suitable future layout is:

```text
learning/vol2/
  tokenizer.py
  tokenizer_audit.py
  data.py
  attention.py
  transformer.py
  train.py
  generate.py
  tests/
  fixtures/
  runs/
    <run-id>/
      config.json
      tokenizer.json
      metrics.jsonl
      checkpoint.pt
      report.md
```

This is a suggested implementation layout, not a requirement to modify the production package. Nothing under `src/todesanzeigen/` should import the educational model.

## Claim boundary

This volume can support:

- **mechanism claims:** attention masking, shape contracts, gradients, and checkpoint resumption are correct;
- **bounded experiment claims:** one frozen small Transformer models the selected corpus better than specified baselines.

It cannot support:

- a claim that the model extracts death-notice fields reliably;
- a claim that it preserves facts or provenance;
- a claim that it should replace an OCR+LLM or VLM extraction variant;
- a claim that a small validation NLL difference matters to users.

Those require the task-specific evaluation designed in Volume III.

## Working rules

1. Train the tokenizer on the training subset only.
2. Freeze tokenizer identity before model comparison.
3. Preserve document boundaries with explicit special tokens.
4. Keep test sealed until architecture and training choices are frozen.
5. Use tiny deterministic fixtures before corpus-scale runs.
6. Every tensor-producing component gets a shape test.
7. Every optimized implementation gets a transparent reference comparison.
8. Every checkpoint records tokenizer, corpus, model, optimizer, scheduler, and RNG identity.
9. Do not log private raw text or generations by default.
10. A lower token-level loss across different tokenizers is not automatically a fair comparison.

## Manual-versus-library boundary

| Component | What you do in this volume | Why |
|---|---|---|
| Pair counting, BPE merge learning, ranked encoding, decoding | Write yourself | Token boundaries and merge order are the mechanism under study |
| Production-tokenizer comparison | Use its maintained library | The value is behavioral comparison, not reverse-engineering its software |
| Tensor storage, matmul, autograd, device transfer | Use PyTorch | Volume I already earned autograd; rebuilding tensor infrastructure has poor return |
| Single-head attention | Write from explicit tensor operations | Makes mask, scale, softmax axis, and value mixing inspectable |
| Multi-head attention | Write explicit and fused versions | Earns the optimized representation through equivalence |
| LayerNorm | Implement the equation once, then use the library | The reduction axis matters; maintaining another norm kernel does not |
| GELU, AdamW, plotting, serialization | Use libraries | Their full production implementations are not the current learning bottleneck |
| Decoder-only Transformer and trainer | Assemble yourself | The interfaces and state transitions are central to model ownership |
| Fused attention kernel, FlashAttention, Triton/CUDA | Optional Deep Dive | First establish correctness and a measured bottleneck |

## Repository boundary for this volume

The natural project data source is OCR text already represented by `ocr_outputs`, together with document/source/year lineage. Do not make the educational trainer query arbitrary production state during every batch. Instead, use a read-only, versioned export with:

- one record per document;
- stable local document/group identifiers;
- text or an explicitly redacted transformation;
- source/year grouping metadata when approved;
- privacy class and export fingerprint;
- no extraction outputs treated as language-model truth.

Synthetic corpus data remains the default committable fixture. A private export, tokenizer trained on it, checkpoints, generations, and run logs stay local unless separately reviewed. This volume does not write model predictions to SQLite, alter extraction variants, or change the OCR pipeline.

---

# Part I — Tokenization

## Chapter 1 — Measure the representation problem before solving it

**Classification:** Core — must do  
**Time:** 30 minutes  
**Artifact:** character/byte baseline report and tokenizer contract

### Objective

Turn “characters are inefficient” into measurements, then define what a tokenizer must guarantee.

### Why we need this now

The Transformer’s attention matrix has one row and column per sequence position. For sequence length $T$, each attention head constructs a score matrix of shape $[T,T]$. Ignoring constants, attention work and score-memory scale as $O(T^2)$.

If a notice is 800 characters but 300 subword tokens, the attention-score ratio is approximately:

$$
\frac{800^2}{300^2}\approx7.1.
$$

Tokenization therefore affects more than vocabulary aesthetics. It changes sequence length, memory, compute, context coverage, and the units over which loss is measured.

### Establish three baselines

On the frozen Volume I training corpus, measure:

1. Unicode code points per document;
2. UTF-8 bytes per document;
3. distinct code points and distinct byte values.

Repeat by slice:

- clean German text;
- names and locations;
- dates;
- umlauts and `ß`;
- OCR-corrupted text;
- any multilingual examples intentionally included.

### Predict

Write answers before measuring:

1. Which German characters occupy more than one UTF-8 byte?
2. Can a code-point vocabulary encode an unseen script without an unknown token?
3. Can a byte vocabulary encode every possible UTF-8 string?
4. Does byte-level coverage imply that malformed or surprising Unicode will be represented compactly?
5. If two tokenizers produce different sequence lengths, can their token-level perplexities be compared directly?

### The base-unit decision

Use UTF-8 bytes as the base vocabulary for the educational BPE:

- IDs `0..255` correspond to byte values;
- learned merge IDs begin at `256`;
- special tokens occupy a separate, explicitly reserved range;
- ordinary text is encoded to UTF-8 bytes before merges;
- decoded ordinary tokens reconstruct bytes, then UTF-8 text.

This gives complete ordinary-text coverage without an unknown token. It does not make every arbitrary byte sequence valid UTF-8. Define your decoder’s policy for invalid bytes: strict failure for tests, and an explicit replacement/error mode only if needed for debugging.

### Special-token contract

At minimum reserve:

- `<BOS>` — optional beginning of a training sequence;
- `<EOS>` — document boundary/end;
- `<PAD>` — only if batches require padding.

Special tokens are control symbols, not byte strings that can accidentally occur in a document. Keep their IDs outside the learned merge range and serialize them in the tokenizer manifest.

Do not add dozens of task tokens in this volume. They complicate the representation before a task needs them.

### Tokenizer invariants

Write these tests before the BPE algorithm:

- `decode(encode(text)) == text` for valid Unicode fixtures;
- every input string encodes without `<UNK>`;
- special tokens round-trip only through the explicit special-token API;
- token IDs are deterministic under the same tokenizer file;
- every learned token recursively expands to base bytes;
- tokenizer training never reads validation or test documents;
- changing merges changes the tokenizer fingerprint.

### Fair comparison units

Token-level NLL depends on token boundaries. For cross-tokenizer comparison, report a base-unit normalized metric such as bits per byte:

$$
\mathrm{bpb}
=
\frac{-\sum_{t=1}^{N_{tok}}\log_2 p(z_t\mid z_{<t})}
{N_{bytes}}.
$$

Within one fixed tokenizer, token NLL and perplexity are fine. Across tokenizers, use bytes or characters as the denominator and state exactly what was counted.

### Checkpoint

You are ready to learn merges if you can explain how tokenization changes both modeling units and attention cost, and why byte coverage is different from good compression.

## Chapter 2 — Learn byte-pair merges from first principles

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** deterministic BPE trainer and merge table

### Mental model

BPE starts with sequences of base symbols. It repeatedly finds a frequent adjacent pair, creates a new symbol for that pair, and replaces its non-overlapping occurrences.

Suppose the corpus currently contains:

```text
[b, e, r, g]
[b, e, r, g, e, r]
```

If `(b,e)` is the chosen pair, assign a new ID `be` and rewrite:

```text
[be, r, g]
[be, r, g, e, r]
```

The new token can later participate in another merge. Tokens therefore form a recursive binary construction over bytes.

### Algorithm

Given training documents represented as lists of byte IDs:

1. count every adjacent token pair within each document;
2. choose the highest-frequency pair;
3. break ties deterministically;
4. assign the next token ID;
5. replace non-overlapping occurrences of that pair;
6. record `(left_id, right_id) -> new_id` and its rank;
7. repeat until the vocabulary target or stopping condition is reached.

Never count a pair across document boundaries. An `<EOS>` token may be present inside a sequence, but do not accidentally merge ordinary bytes with control tokens unless that behavior is explicitly intended. For this course, exclude special-token pairs from learning.

### Deterministic tie-breaking

Two pairs often have equal counts. Choose a stable rule, for example:

1. higher count first;
2. then lexicographically smaller `(left_id, right_id)`.

Record the rule. Without it, identical corpora can produce different vocabularies on different runs or Python versions.

### Non-overlapping replacement

For sequence `[a,a,a]` and merge `(a,a)`, a left-to-right replacement produces `[aa,a]`, not overlapping `[aa,aa]`. Define this behavior in a unit test.

### Skeleton

```python
@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    new_id: int
    rank: int

def pair_counts(documents: list[list[int]]) -> dict[tuple[int, int], int]:
    # Count only within each document.
    ...

def replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    # Left-to-right, non-overlapping replacement.
    ...

def train_bpe(documents: list[bytes], target_vocab_size: int) -> list[Merge]:
    # IDs 0..255 are bytes. Learned IDs start at 256.
    ...
```

### Predict

1. Which kinds of strings receive early merges: frequent whole words, frequent byte pairs, or semantically meaningful units?
2. Why might the UTF-8 bytes of `ä` merge early even if no full word containing `ä` does?
3. What happens to a rare surname that never appears in training?
4. Does a larger vocabulary always reduce total training or inference cost?
5. What is the training-time complexity of recounting every pair after every merge, and why is that acceptable here?

### Tests

Use tiny hand-worked corpora and assert:

- exact pair counts;
- no cross-document pair;
- exact chosen pair under a tie;
- exact result of overlapping candidates;
- monotonically assigned IDs;
- no merge references a future token;
- every merge reduces corpus token count by its replacement count;
- repeated training on the same input yields byte-identical serialized merges.

### Deliberately naive implementation

Recounting all pairs after every merge is inefficient, but it keeps state transitions visible. Use it for the core. Do not spend the timebox implementing an incremental priority queue unless tokenizer training becomes a measured bottleneck.

This is a deliberate Level 1 implementation: pedagogical correctness over production throughput.

### Stop conditions

Support:

- target vocabulary size;
- no pair with count above a minimum;
- optional maximum number of merges.

If the requested vocabulary exceeds what the corpus can support, stop cleanly and report why.

### What to inspect

For the first 30 and last 30 merges, print:

- rank and count at selection time;
- left/right token byte expansions;
- new token byte expansion;
- decoded text when valid and printable;
- escaped bytes otherwise.

Do not assume a merge is meaningful because its decoded form looks like a morpheme. BPE optimizes pair frequency, not linguistic theory.

### Failure modes

- pair counting across documents;
- replacing overlapping occurrences twice;
- using string replacement and corrupting byte boundaries;
- learning merges involving `<PAD>` or `<EOS>`;
- nondeterministic ties;
- assigning token IDs differently during reload;
- training on validation/test data because “the tokenizer has no labels.”

### Hint ladder

**Hint 1:** Treat a document as a list of integer token IDs. Do not manipulate decoded strings during learning.

**Hint 2:** In `replace_pair`, scan with index `i`; when `ids[i:i+2]` matches, append `new_id` and advance by two, otherwise append one ID and advance by one.

**Hint 3:** Recompute pair counts from the rewritten corpus on every iteration. It is slow but hard to make subtly stale.

### Checkpoint

Explain why a BPE token has no intrinsic semantic meaning, why merge rank is part of tokenizer identity, and why a deterministic but slow trainer is the right educational implementation.

## Chapter 3 — Encoding, decoding, and tokenizer identity

**Classification:** Core — must do  
**Time:** 60 minutes  
**Artifact:** reversible encoder/decoder plus serialized tokenizer

### Encoding is not “apply every merge globally”

At inference time, convert text to UTF-8 bytes, then repeatedly apply the available merge with the best learned rank among adjacent pairs until no learned pair remains.

The rank matters. Later merges were learned in a corpus already transformed by earlier merges. Applying merges in arbitrary dictionary order can produce a different tokenization.

For a short educational implementation, the simplest correct loop is:

1. begin with byte IDs;
2. list adjacent pairs that appear in the merge table;
3. choose the pair with the smallest rank;
4. replace its non-overlapping occurrences;
5. repeat.

This is not the fastest encoder. It is transparent and sufficient for the core.

### Decoder

Construct a byte expansion for every token:

- base token `i` expands to `bytes([i])`;
- merged token expands to `expansion[left] + expansion[right]`.

Decoding ordinary token IDs concatenates their byte expansions, then decodes UTF-8.

Special tokens are handled explicitly. Do not pretend `<EOS>` is ordinary text.

### Interface

```python
class ByteBPETokenizer:
    def __init__(self, merges, special_tokens): ...

    def encode(self, text: str, *, add_bos=False, add_eos=False) -> list[int]: ...

    def decode(self, ids: list[int], *, allow_special=False) -> str: ...

    def token_bytes(self, token_id: int) -> bytes: ...

    def save(self, path): ...

    @classmethod
    def load(cls, path): ...
```

### Manifest

Serialize at least:

- format version;
- base encoding (`utf-8`);
- ordered merge records with rank and token IDs;
- special-token strings and IDs;
- training-corpus fingerprint;
- training-split identity;
- target and actual vocabulary size;
- trainer configuration and tie-break rule;
- tokenizer fingerprint over canonical serialized content.

Canonical means stable key ordering and stable list order. A fingerprint is only useful if semantically identical tokenizers serialize identically.

### Correctness tests

Round-trip fixtures containing:

- ASCII;
- `Müller`, `Großaitingen`, `Bärbel`;
- composed and decomposed Unicode forms;
- punctuation and line breaks;
- empty text;
- emoji or a non-Latin sample;
- OCR-like fragments and replacement characters.

Also test:

- save/load preserves every encoding exactly;
- unknown text always falls back to bytes;
- decode rejects invalid token IDs;
- special tokens cannot be injected through ordinary text;
- all merge expansions terminate at bytes;
- tokenizer fingerprint changes if one merge rank changes.

### Unicode normalization decision

Do not silently normalize text inside the tokenizer. Unicode normalization can make equivalent-looking strings share representation, but it can also erase distinctions or hide OCR behavior.

For the core:

- treat normalization as an explicit corpus-preprocessing policy;
- record the policy in the corpus manifest;
- include composed/decomposed examples in the audit;
- keep tokenizer encode/decode reversible for the provided string.

Later, the production pipeline may choose normalization for task reasons. That decision belongs at a named preprocessing boundary, not inside an opaque tokenizer.

### Predict

1. Will `decode(encode(x))` preserve the exact code-point sequence if preprocessing normalizes `x` first?
2. Can two tokenizers with the same token byte expansions but different IDs load the same model checkpoint safely?
3. Why must the tokenizer fingerprint be stored with model weights?
4. What failure appears if training uses one `<EOS>` ID and generation uses another?

### Failure modes

- decoding token display strings instead of byte expansions;
- forgetting merge rank;
- treating a special-token spelling in user text as control syntax;
- reassigning IDs after sorting tokens for display;
- loading a model with a tokenizer that has the same vocabulary size but different IDs;
- normalizing only during training or only during inference.

### What abstraction is now safe

You may use a maintained production tokenizer later, but only after its behavior is measured with the same audit. “Industry standard” does not answer how it fragments a rare surname or corrupted date.

## Chapter 4 — Audit the tokenizer on the domain

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** tokenizer scorecard and frozen vocabulary decision

### Objective

Choose a vocabulary size using domain evidence, not aesthetics.

### Candidate sizes

Train at least three tokenizers, for example:

- 256 bytes plus specials: no learned merges;
- approximately 512 tokens;
- approximately 1,024 or 2,048 tokens, depending on corpus size.

Do not chase a large vocabulary on a small corpus. Rare learned tokens receive few training updates and enlarge both the embedding and output matrices.

### Metrics

For train and validation separately, report:

- bytes per token;
- characters per token;
- tokens per document distribution, not only mean;
- compression by domain slice;
- fraction of token occurrences contributed by the most frequent 1%, 10%, and 50% of token types;
- number of learned tokens never observed in validation;
- token count for a fixed rare-name/date/location panel;
- frequency of single-byte fallback tokens.

For model-size consequences, calculate embedding/output parameters:

$$
P_{emb}=VC,
\qquad
P_{head}=CV+V.
$$

If weights are tied, the shared matrix removes one $VC$ term, but the vocabulary still affects memory and output computation.

### Domain panel

Create synthetic or approved examples containing:

- common and rare German surnames;
- hyphenated and multi-part names;
- umlauts and `ß`;
- dates such as `03.01.1927`, textual months, and OCR-confused digits;
- towns with frequent suffixes;
- titles and abbreviations;
- line breaks and layout-loss artifacts;
- `rn/m`, `1/l/I`, broken spacing, and punctuation corruption;
- optional English and Chinese examples if multilingual support is relevant.

Display tokens as ID plus escaped byte expansion. A pretty decoded token alone can conceal byte fallback.

### Rare-name preservation is not “one token per name”

A rare name being split is not automatically harmful. Byte fallback guarantees representability. The relevant questions are:

- how many autoregressive decisions must reproduce it;
- whether common substrings are shared;
- whether corruption causes extreme fragmentation;
- whether later evaluation shows disproportionate name errors.

Volume III will measure extraction/entity outcomes. Here you measure representation burden.

### Compare with one production tokenizer

Use a production tokenizer available in your eventual model family and run the same panel. Do not reimplement it. Compare:

- coverage;
- sequence length;
- byte fallback behavior;
- treatment of whitespace and Unicode;
- vocabulary size cost.

The comparison is behavioral, not a popularity contest.

### Selection rule

Freeze the smallest vocabulary that gives a meaningful sequence-length reduction without creating a disproportionate embedding/output matrix for the planned small model.

Write the decision as:

```text
Chosen tokenizer:
Training-only corpus fingerprint:
Vocabulary size:
Median and p95 tokens/document:
Bytes/token overall and by critical slice:
Embedding/head parameter cost at model width C:
Known weaknesses:
Reason rejected alternatives lost:
```

### Tokenizer mastery checkpoint

You are ready for attention if you can:

- implement BPE training and ranked encoding from memory;
- explain byte coverage versus compression;
- round-trip arbitrary valid Unicode fixtures;
- prevent split leakage and special-token ambiguity;
- compare tokenizers in bits per byte rather than raw token perplexity;
- justify the frozen vocabulary with domain measurements.

This is the first stop-safe artifact of Volume II.

---

# Part II — Causal Self-Attention

## Chapter 5 — Define the autoregressive tensor contract

**Classification:** Core — must do  
**Time:** 30 minutes  
**Artifact:** token-block dataset and shape ledger

### From token stream to training examples

For token sequence:

```text
[t0, t1, t2, ..., tT]
```

input and target are shifted by one:

```text
x = [t0, t1, ..., t(T-1)]
y = [t1, t2, ..., tT]
```

For batch size $B$ and context length $T$:

- input IDs `x`: `[B,T]`;
- target IDs `y`: `[B,T]`;
- token embeddings: `[B,T,C]`;
- output logits: `[B,T,V]`;
- flattened loss inputs: `[B*T,V]` and `[B*T]`;
- scalar mean loss: `[]`.

Every position predicts its next token. Unlike the Volume I context MLP, one block provides $B\times T$ supervised next-token events in parallel.

### Packing documents

For the core, use one of two explicit policies:

1. **Packed with `<EOS>`.** Shuffle training documents deterministically, concatenate each with `<EOS>`, and sample blocks. The model can attend across the boundary, but the boundary is visible. Reshuffle document order by epoch if practical.
2. **Document-contained blocks.** Sample windows only within documents; pad short documents and mask padded loss. This preserves isolation but adds padding/mask complexity.

Choose one and test it. Do not concatenate raw documents with no boundary marker.

For time efficiency, packed-with-`<EOS>` is the default. Evaluation should still report document-aware loss so arbitrary packing order does not become the claim.

Perform exact/near-duplicate grouping before the split, not after tokenization. OCR reruns, repeated notices, and template-derived synthetic variants must remain in one subset. Tokenizing first and then deduplicating token windows can conceal document-level leakage.

### Positional information

Self-attention without position is permutation-equivariant: swapping input positions swaps outputs but does not tell the operation which token came first.

Use learned absolute position embeddings for the core:

$$
h_{b,t}=E_{token}[x_{b,t}] + E_{pos}[t].
$$

Shapes:

- token table: `[V,C]`;
- position table: `[T_max,C]`;
- token lookup: `[B,T,C]`;
- position lookup: `[T,C]`, broadcast across batch;
- sum: `[B,T,C]`.

Learned absolute positions are not necessarily the best modern choice. They are the shortest route to understanding why order information is needed. Rotary position embeddings are an Optional Deep Dive after attention itself is correct.

### Predict

1. What is the random-logit expected loss for vocabulary size $V$?
2. What happens if `x` and `y` are identical instead of shifted?
3. What happens if position embeddings are removed?
4. Can a model with context limit 128 accept 129 tokens under learned absolute positions without a policy change?
5. If a block starts midway through a document, what positional index should its first token receive under this simple training scheme?

### Tests

- decode several `(x,y)` pairs and inspect the one-token shift;
- no example accesses outside the token stream;
- `<EOS>` appears between packed documents;
- train/validation streams use the frozen tokenizer;
- token IDs remain below `V`;
- blocks of lengths 1 and `T_max` work;
- position IDs have shape `[T]` and remain in range;
- test stream is not sampled during tuning.

### Failure mode to preserve

Create one regression test where the target token appears inside the input at the same position. A model can then learn an identity shortcut and produce implausibly low loss. This is the sequence-model version of label leakage.

## Chapter 6 — Single-head causal self-attention

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** transparent single-head attention with hand-inspected matrices

### Why we need this now

The context MLP combines a fixed number of positions by flattening them. Attention instead lets each position construct a data-dependent weighted mixture of earlier token representations using shared projections.

Input:

$$
X\in\mathbb{R}^{B\times T\times C}.
$$

For one head with key/query width $D$:

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V,
$$

where $W_Q,W_K,W_V\in\mathbb{R}^{C\times D}$.

Shapes:

```text
X                         [B,T,C]
Wq, Wk, Wv               [C,D]
Q, K, V                  [B,T,D]
K transpose              [B,D,T]
scores = Q @ K^T         [B,T,T]
causal mask              [T,T]
weights = softmax(scores) [B,T,T]
output = weights @ V     [B,T,D]
```

For $H$ heads with $D=C/H$, the two attention matrix multiplications cost on the order of:

$$
O(BHT^2D)=O(BT^2C).
$$

The QKV/output projections cost on the order of $O(BTC^2)$. Which term dominates depends on sequence length and width. Materialized attention weights contain $BHT^2$ elements. Keep these quantities visible; Volume IV will use them to explain prefill cost and why caching changes decode work but not model weights.

### Query, key, and value mental model

At position $t$:

- the **query** represents what this position is looking for;
- each earlier position’s **key** represents what it offers for matching;
- its **value** represents the information transferred if selected.

This language is useful but incomplete. They are learned linear projections, not symbolic database fields. Inspect behavior rather than assigning human semantics too early.

### Scaled dot-product attention

Scores are:

$$
S=\frac{QK^\top}{\sqrt D}.
$$

If query/key components have roughly unit variance, an unscaled dot product sums $D$ terms and has variance proportional to $D$. Larger $D$ creates larger logits, sharper softmax distributions, and potentially poor gradients. Dividing by $\sqrt D$ keeps score scale roughly stable.

### Causal mask

At position $t$, the model may use positions $s\le t$ and must not use $s>t$.

For a score matrix, mask future entries before softmax:

```text
allowed (1) / forbidden (0)

[[1,0,0,0],
 [1,1,0,0],
 [1,1,1,0],
 [1,1,1,1]]
```

Forbidden logits become negative infinity—or a sufficiently negative representable value—so their softmax probability is zero.

Masking after softmax without renormalizing is wrong: allowed rows no longer sum to one, and forbidden values may already have contributed.

### Predict before implementing

1. For the first token, how many keys are visible?
2. For the last token in a full block, how many are visible?
3. Along which dimension must softmax operate?
4. If all allowed scores in a row are equal, what are the attention weights?
5. What changes if scaling by $\sqrt D$ is removed?
6. If the causal mask is transposed, what information leaks?
7. Does a causal mask prevent a token from attending to itself?

### Implement in explicit stages

```python
class SingleHeadCausalAttention(nn.Module):
    def __init__(self, model_dim, head_dim, max_context, bias=False):
        # TODO: q, k, v projections and causal mask buffer
        ...

    def forward(self, x, *, return_debug=False):
        # x: [B,T,C]
        # q,k,v: [B,T,D]
        # scores: [B,T,T]
        # apply mask for current T
        # weights: row-normalized over key positions
        # out: [B,T,D]
        ...
```

Keep score matrices accessible behind `return_debug` for tiny fixtures. Do not return them during normal training; their memory cost is quadratic.

### Tiny hand-inspection lab

Use `B=1`, `T=4`, `C=4`, `D=2`. Choose simple deterministic projection matrices. Print:

- $Q$, $K$, and $V$;
- raw scores;
- scaled scores;
- masked scores;
- attention weights;
- output.

For at least one row, calculate the dot products and weighted value sum manually.

Assertions:

- each weight row sums to one;
- all future weights are exactly or numerically zero;
- output row is a convex combination of visible value rows;
- changing a future input token does not change earlier outputs;
- changing an earlier token can change later outputs.

The future-token perturbation test is more convincing than looking at a triangular matrix.

### Gradient tests

- loss on all output elements produces finite gradients for $X,W_Q,W_K,W_V$;
- a loss using only output position 0 gives zero gradient to future input positions under a correct mask;
- compare selected parameter gradients with double-precision finite differences on the tiny fixture;
- compare against a direct reference equation written outside the module.

### Broken experiment: remove scaling

Run identical random inputs for several $D$ values with and without $1/\sqrt D$. Record:

- score standard deviation;
- attention entropy;
- maximum weight;
- query/key gradient norms.

Predict how each changes as $D$ grows. Do this before training a full model, where many effects become entangled.

### Attention entropy

For one query row with weights $a_s$:

$$
H(a)=-\sum_s a_s\log(a_s+\epsilon).
$$

Low entropy means concentrated attention; high entropy means diffuse attention. Neither is automatically good. Use it to identify collapse, saturation, or unexpected masking—not to assign interpretability by itself.

### Likely bugs

- transposing the wrong dimensions;
- softmax over query positions instead of key positions;
- scaling by $\sqrt C$ instead of $\sqrt D$;
- mask on CPU while scores are on GPU;
- mask of `[T,T]` not sliced to current sequence length;
- using a finite mask value that is insufficient for a low-precision dtype;
- applying dropout during deterministic equivalence tests;
- leaking future tokens through an incorrectly oriented triangle.

### Hint ladder

**Hint 1:** Each query position needs one score against every key position. The final two score dimensions are query and key.

**Hint 2:** `q @ k.transpose(-2, -1)` yields `[B,T,T]` for one head.

**Hint 3:** Apply a lower-triangular mask to scores, then softmax over the last dimension, then multiply by `v`.

### Checkpoint

Draw every tensor shape from memory and explain how a perturbation test proves causality.

## Chapter 7 — Multi-head attention and the fused implementation

**Classification:** Core — must do  
**Time:** 60 minutes  
**Artifact:** multi-head module with transparent/fused equivalence

### Why multiple heads

One head produces one attention distribution per query. Multiple heads learn several projected subspaces in parallel. They can represent different interaction patterns, though no head is guaranteed to align with a human concept.

Choose model width $C$ and head count $H$ such that:

$$
D=C/H.
$$

For the simplest implementation, require exact divisibility.

### Shape ledger

```text
input x                    [B,T,C]
projected q,k,v            [B,T,C] each
reshape                     [B,T,H,D]
transpose                   [B,H,T,D]
scores                      [B,H,T,T]
weights                     [B,H,T,T]
head outputs                [B,H,T,D]
transpose + contiguous      [B,T,H,D]
concatenate                 [B,T,C]
output projection           [B,T,C]
```

The output projection lets information from heads mix back into model width.

### Implement twice

#### Version A — explicit heads

Use a module list of single-head modules, concatenate outputs, and apply an output projection. This is easy to inspect.

#### Version B — fused QKV

Use one projection from $C$ to $3C$, split into Q/K/V, reshape into heads, compute all heads in one batched operation, concatenate, and project.

The fused version reduces Python/module overhead and resembles production code more closely. It must not be accepted until it matches the explicit version after copying parameters.

### Equivalence procedure

1. Disable dropout.
2. Create one deterministic input.
3. Copy each explicit head’s Q/K/V weights into the corresponding slices of the fused projection.
4. Copy the output projection.
5. Compare outputs.
6. Apply the same scalar loss and compare input and parameter gradients.

Use float64 for the audit. Document the tolerance.

### Predict

1. If $C$ is fixed, does increasing head count change the size of the QKV projection matrices?
2. How does it change per-head dimension and score scaling?
3. What happens if heads are concatenated in a different order than the output projection expects?
4. Why can a reshape after transpose require `contiguous()` or `reshape()` rather than `view()`?
5. Which tensor is the main quadratic-memory object?

### Attention dropout

Apply dropout to attention weights only after a no-dropout implementation is correct. The expected weighted sum is preserved under standard inverted dropout, but individual row sums during training are not necessarily one after dropout.

Do not use row-sum assertions while dropout is active.

### Tests

- explicit and fused forward outputs agree;
- gradients agree;
- output shape is `[B,T,C]` for several `B,T`;
- future-token perturbation still cannot change earlier outputs;
- head count 1 reduces to the single-head structure;
- invalid `C % H != 0` fails clearly;
- evaluation mode is deterministic;
- dropout changes training outputs but not evaluation outputs.

### Failure modes

- splitting QKV along the wrong axis;
- mixing batch and head dimensions;
- applying scale based on total width;
- forgetting the output projection;
- using one shared mask with an accidental head dimension mismatch;
- comparing stochastic dropout runs and blaming numerical error.

## Chapter 8 — Position is information, not an index bookkeeping detail

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** positional ablation and inspected attention examples

### Objective

Prove that content-only attention cannot distinguish all relevant orderings, then establish the positional mechanism used by the complete model.

### Thought experiment

Consider token embeddings for `Anna sah Maria` and `Maria sah Anna`. Without positional information, self-attention receives the same multiset of content vectors in a different order. The operation is equivariant to that permutation; it has no absolute or relative order signal beyond the permuted arrangement itself.

Language modeling needs order because the next-token distribution changes when subjects, dates, names, or negation move.

### Learned absolute positions

Add position embeddings before the first block:

$$
H^{(0)}=E_{token}[x]+E_{pos}[0:T].
$$

Use positions relative to the current block (`0..T-1`) for this educational model. Record that this differs from maintaining absolute document offsets across sampled windows.

### Required ablation

Train two tiny, otherwise identical models for a short fixed budget:

- with learned position embeddings;
- with the position embeddings forced to zero.

Use synthetic sequences where order is essential, such as permutations with different next targets. First confirm the no-position model cannot solve the constructed distinction; then compare on the domain corpus.

### Predict

1. Can causal masking alone provide some positional signal because earlier positions have different visible-prefix lengths?
2. Is that signal sufficient to identify exact relative distances and token order in all cases?
3. What happens to a learned absolute position table when generation exceeds its trained maximum context?
4. Why might rotary or relative position methods extrapolate differently?

The careful answer to the first question is yes: the causal structure breaks full permutation symmetry because visibility differs by position. But it is not a rich, explicit encoding of relative order. The ablation should be reasoned about empirically, not reduced to the slogan “attention has no order.”

### Optional Deep Dive — rotary positions

After the core model works, implement rotary position embeddings on tiny Q/K tensors and verify that dot products acquire relative-position dependence. Do not replace the core position system midstream; that would confound the training comparison.

### Platform connection

Dates and relationship phrases are order-sensitive. `geb. 1920, gest. 1990` is not interchangeable with the reverse. Positional ablations should include structured synthetic examples resembling these domain distinctions without using private records.

### Attention mastery checkpoint

You are ready to build a Transformer block when you can:

- derive Q, K, V, score, weight, and output shapes;
- explain and test causal masking;
- explain the $1/\sqrt D$ scale;
- match explicit and fused multi-head implementations;
- describe what learned positions add and where they fail;
- inspect a tiny attention matrix without treating it as a complete explanation of model reasoning.

---

# Part III — The Decoder-Only Transformer

## Chapter 9 — Normalization, residual paths, and the feed-forward network

**Classification:** Core — must do  
**Time:** 90 minutes  
**Artifact:** tested pre-normalized Transformer block

### Why attention alone is not the block

Attention mixes information across sequence positions. The feed-forward network transforms each position independently. Residual paths preserve an identity route through depth. Normalization controls scale and improves optimization.

The core pre-normalized block is:

```text
x [B,T,C]
 │
 ├──────────────────────────────┐
 ▼                              │
LayerNorm → Causal MHA → dropout│
 │                              │
 └──────────── add ◄────────────┘
 x1 [B,T,C]
 │
 ├──────────────────────────────┐
 ▼                              │
LayerNorm → Linear → GELU        │
          → Linear → dropout     │
 │                              │
 └──────────── add ◄────────────┘
 x2 [B,T,C]
```

In equations:

$$
x_1=x+\operatorname{MHA}(\operatorname{LN}(x)),
$$

$$
x_2=x_1+\operatorname{MLP}(\operatorname{LN}(x_1)).
$$

### Layer normalization

For each token vector $x\in\mathbb{R}^C$:

$$
\mu=\frac1C\sum_i x_i,
\qquad
\sigma^2=\frac1C\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LN}(x)_i
=
\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

Normalization is over the feature dimension independently at each batch/position pair. It does not average across tokens or examples.

Implement the formula once and compare forward values and gradients with the framework module in float64. Then use the framework module.

### Feed-forward network

Use expansion factor 4 for the default:

$$
\operatorname{FFN}(x)=W_2\,\operatorname{GELU}(W_1x+b_1)+b_2.
$$

Shapes:

- $W_1$: `[C,4C]` in mathematical orientation;
- hidden activations: `[B,T,4C]`;
- $W_2$: `[4C,C]`;
- output: `[B,T,C]`.

Framework linear layers may store weights transposed relative to the equation. State both the mathematical and library storage shape when debugging.

### GELU mental model

GELU smoothly gates values by magnitude rather than clipping all negatives to zero. You do not need to derive its closed form here. Inspect its input/output distribution and gradient on a small range; use the production primitive afterward.

### Residual path

A residual sublayer computes:

$$
y=x+F(x).
$$

Its Jacobian includes an identity term:

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
$$

This gives information and gradients a direct route. It does not guarantee stability, but removing it makes deep optimization substantially harder.

### Predict

1. What shape must every residual branch return?
2. What happens if LayerNorm normalizes across time instead of features?
3. If the residual is removed, can a shallow model still train?
4. Why does pre-normalization tend to improve gradient flow in small educational models?
5. Which module—attention or the 4× FFN—contains more parameters at fixed $C$?

### Parameter accounting

Ignoring biases:

- attention QKV projections: $3C^2$;
- attention output projection: $C^2$;
- FFN: $C(4C)+(4C)C=8C^2$;
- norms: small $O(C)$ terms.

The FFN usually contains roughly twice the dense parameters of attention projections in this design. Attention’s distinctive cost comes from its $T^2$ interaction matrix, not necessarily from having the most parameters.

### Block skeleton

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = ...
        self.attn = ...
        self.ln2 = ...
        self.ffn = ...

    def forward(self, x):
        # TODO: pre-norm attention residual
        # TODO: pre-norm feed-forward residual
        ...
```

### Tests

- input/output shape equality;
- LayerNorm manual/framework equivalence;
- zeroing all sublayer weights makes the block an identity map;
- future-token perturbation cannot change earlier block outputs;
- all parameters receive finite gradients on a nondegenerate loss;
- train/eval behavior differs only where stochastic layers such as dropout require it;
- a stack of two blocks preserves shape and causality.

### Broken lab: remove one residual

Train a deeper tiny model for a fixed short budget with and without one residual path. Record:

- loss;
- gradient norm by depth;
- activation standard deviation by depth;
- update/parameter ratio by block.

Do not claim that residuals are always necessary from one run. Explain the observed optimization path and the identity-gradient mechanism.

### Failure modes

- post-norm code while reasoning about pre-norm;
- in-place residual mutation that interferes with autograd;
- wrong normalization axis;
- FFN output width not returning to $C$;
- applying one LayerNorm instance to both sublayers unintentionally;
- dropout active during equivalence tests.

## Chapter 10 — Assemble the complete decoder-only model

**Classification:** Core — must do  
**Time:** 90 minutes  
**Artifact:** full Transformer with parameter and shape audit

### Architecture

```text
token IDs [B,T]
   │
   ├── token embedding [B,T,C]
   └── position embedding [T,C]
              │ add
              ▼
        hidden [B,T,C]
              │
      Transformer block × L
              │
          final LayerNorm
              │
        language-model head
              ▼
         logits [B,T,V]
```

### Configuration

Use one immutable configuration object containing:

- vocabulary size $V$;
- maximum context $T_{max}$;
- model width $C$;
- layer count $L$;
- head count $H$;
- FFN expansion;
- dropout probabilities;
- bias choices;
- tokenizer fingerprint.

Validate `C % H == 0` and all positive dimensions at construction.

### Core forward interface

```python
class DecoderOnlyTransformer(nn.Module):
    def forward(self, input_ids, targets=None):
        # input_ids: [B,T]
        # logits: [B,T,V]
        # loss: scalar or None
        return logits, loss
```

Do not hide target shifting inside two different places. The dataset should produce aligned `x,y`, and the model should compute cross-entropy over corresponding positions.

### Final normalization

With pre-normalized blocks, apply one final normalization before the language-model head. Test its presence as part of the architecture identity.

### Weight tying

The token embedding and output head both map between vocabulary and model dimensions:

- input embedding: token ID $\to$ row in `[V,C]`;
- output head: hidden `[C]` $\to$ logits `[V]`.

Weight tying reuses one matrix for both roles, usually with transpose implied by the linear operation. It reduces parameters and couples input/output representations.

For the core:

1. implement untied matrices first;
2. add tying as one explicit option;
3. verify object identity/storage sharing, not merely equal initial values;
4. compare parameter counts and one short run.

### Initialization

Use a documented small normal initialization for embeddings and linear weights, zero biases, and norm scales of one. The exact standard deviation is less important than verifying:

- initial logits are not excessively large;
- initial loss is near $\log V$;
- activation standard deviations do not explode with depth;
- gradients reach early blocks.

If residual output projections receive special scaling, derive and record it. Do not paste an unexplained constant from another architecture.

### Parameter count audit

Compute total parameters two ways:

1. analytical formula from configuration;
2. sum of `numel()` over unique parameters.

Break down:

- token/position embeddings;
- attention projections;
- FFNs;
- norms;
- output head.

The two totals must agree, accounting for tied storage correctly.

### Naive generation

Implement a simple autoregressive loop only to inspect the trained model:

1. take the last at most `T_max` tokens;
2. run the entire model;
3. select logits at the final position;
4. apply temperature and optional top-k;
5. sample one token;
6. append and repeat until `<EOS>` or length cap.

This deliberately recomputes the full prefix. Do not add a KV cache here. Volume IV begins by profiling this waste and deriving the cache from attention state.

### Predict

1. At random initialization, what loss should you expect?
2. Which parameter groups dominate total count?
3. If input/output weights are tied, can their gradients come from both embedding and output uses?
4. Why does generation become slower as the prefix grows in this naive loop?
5. What happens when generated length exceeds `T_max`?

### Correctness ladder

1. Forward shapes for `B=1,T=1`.
2. Forward shapes for maximum context.
3. Finite loss near $\log V$.
4. Causality perturbation through the entire model.
5. Tiny-batch overfit.
6. Save/load fixed-logit equality.
7. One optimizer step changes parameters and usually the fixed-batch loss.
8. Generation terminates or respects a hard cap.

### Full-model causality test

Create two sequences identical through position $k$ and different afterward. In evaluation mode, logits at positions `<= k` must agree within tolerance. This test catches leakage anywhere in the stack, not only in the attention module.

### What abstraction is now safe

After explicit attention and fused equivalence, you may compare against an optimized framework attention primitive. Keep the transparent path for tests and tiny matrix inspection. Production kernels may fuse score, mask, softmax, dropout, and value multiplication without materializing the full score matrix, but the mathematics remains the same contract.

---

# Part IV — Train a Small Domain Model

## Chapter 11 — Build the smallest trustworthy trainer

**Classification:** Core — must do  
**Time:** 75 minutes  
**Artifact:** deterministic trainer with resumable checkpoints

### Objective

Move from a correct model to a reproducible training experiment without constructing a general MLOps platform.

### Default scale

Choose a model that allows several complete runs. A reasonable starting region—not a mandate—is:

```text
vocabulary:       512–2,048
context:          128
model width:      128
heads:            4
layers:           4
FFN expansion:    4
dropout:          0.0–0.1
```

Calculate parameters and activation sizes before launching. If CPU-only or memory-constrained, reduce width/layers/context. A 2-million-parameter model you can test repeatedly teaches more than a 50-million-parameter model you cannot debug.

### Batch and token budget

Define training budget in tokens, not only steps:

$$
N_{tokens}=\text{steps}\times B\times T\times\text{grad-accumulation steps}.
$$

This makes comparisons fair when batch or context changes.

### Gradient accumulation

If memory limits batch size, split an effective batch into microbatches. Divide each microbatch loss by the number of accumulation steps before backward so the accumulated gradient corresponds to the mean effective-batch loss.

Predict what happens if you forget that division. Then verify on a deterministic batch that one large batch and accumulated microbatches produce matching gradients when dropout is disabled.

### AdamW

Use a maintained optimizer after the Volume I state-update audit. Separate decayed and non-decayed parameters deliberately. A common educational policy is:

- decay matrix weights;
- do not decay biases or normalization scale/shift;
- document the grouping rather than copying a name heuristic blindly.

Record optimizer hyperparameters and grouping in the checkpoint.

### Learning-rate schedule

Use a short linear warmup followed by cosine decay or a fixed rate for very short diagnostic runs.

Warmup:

$$
\eta_t=\eta_{max}\frac{t}{t_{warmup}}.
$$

Cosine decay after warmup:

$$
\eta_t=\eta_{min}
+\frac12(\eta_{max}-\eta_{min})
\left(1+\cos\left(\pi\frac{t-t_{warmup}}{t_{total}-t_{warmup}}\right)\right).
$$

The scheduler state is part of checkpoint identity. Resuming at the wrong step silently changes optimization.

### Gradient clipping

Log the unclipped global norm, clip to a declared threshold, and record whether clipping activated. Persistent clipping means the threshold or training dynamics deserve investigation.

### Mixed precision

Mixed precision is **Useful — do if hardware supports it**, not required for the first correctness run.

Order:

1. establish a float32 reference run;
2. enable autocast;
3. use the appropriate gradient-scaling behavior for the dtype/device;
4. compare loss trajectory, speed, memory, and non-finite counts.

Do not debug masking, checkpointing, and mixed precision simultaneously.

### Checkpoint contents

A resumable checkpoint contains:

- model state;
- optimizer state;
- scheduler state;
- global step and tokens processed;
- gradient-accumulation position if saving mid-cycle;
- model configuration;
- tokenizer file or immutable fingerprint plus path policy;
- corpus and split fingerprints;
- Python, NumPy, CPU, and accelerator RNG states used by the run;
- best validation metric and selection state;
- software version/commit when available.

Never load weights under a tokenizer or model configuration that merely has matching dimensions. Validate fingerprints.

### Resume equivalence test

On CPU or another deterministic path:

1. run $N$ steps continuously;
2. run $K$ steps, save, reload, and run $N-K$ steps;
3. compare next-batch identity, learning rate, fixed-batch logits, optimizer state, and parameters.

Exact bitwise equality may not hold on all accelerators. State the deterministic scope and use a justified tolerance. Large trajectory divergence after one resumed step is a bug.

### Evaluation cadence

At a fixed token interval:

- switch to evaluation mode;
- disable gradients;
- evaluate a fixed number of train and validation tokens or the full small corpus;
- record token NLL, perplexity, and bits per byte;
- restore training mode.

Do not let a random validation subset change at every report unless its sampling is fixed and large enough for the intended comparison.

### Minimal metrics record

```json
{
  "step": 500,
  "tokens_seen": 8192000,
  "train_nll": 3.12,
  "validation_nll": 3.28,
  "validation_bpb": 1.41,
  "learning_rate": 0.0006,
  "grad_norm_unclipped": 0.87,
  "clip_applied": false,
  "tokens_per_second": 18400,
  "peak_memory_bytes": 1234567890
}
```

Use actual measured values; the example numbers are schema illustrations, not targets.

### Training loop skeleton

```python
for step in range(start_step, total_steps):
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(accum_steps):
        x, y = next_train_batch()
        logits, loss = model(x, y)
        (loss / accum_steps).backward()

    grad_norm = clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    scheduler.step()

    if should_evaluate(step):
        evaluate_and_record(...)

    if should_checkpoint(step):
        save_complete_state(...)
```

Treat it as a contract, not a finished implementation. Add device transfer, autocast, error guards, and run-state handling explicitly.

### Failure modes

- schedule stepped before/after optimizer inconsistently across resume;
- accumulation loss not divided;
- stale gradients across effective batches;
- evaluation with dropout active;
- selecting checkpoints on training loss;
- checkpoint omits tokenizer identity;
- logging only step count when effective token count differs;
- resuming data iteration at a different location;
- overwriting the only good checkpoint with a corrupt partial write.

### Safe checkpoint writes

Write to a temporary file in the same output directory, flush/close it, then atomically replace the target. Keep at least the best validation checkpoint and a recent recovery checkpoint. This is normal implementation discipline, not production infrastructure excess.

## Chapter 12 — Diagnose Transformer training behavior

**Classification:** Core — must do  
**Time:** 45 minutes  
**Artifact:** diagnostic dashboard/table and four broken-run reports

### Keep the Volume I ladder

Before tuning:

1. inspect decoded data and target shift;
2. verify tokenizer/model fingerprints;
3. check expected initial loss near $\log V$;
4. prove full-model causality;
5. overfit one tiny batch;
6. inspect activations and gradients by depth;
7. verify one optimizer update;
8. only then change schedule or architecture.

### What to record by depth

For selected batches and checkpoints:

- residual-stream mean and standard deviation at each block input/output;
- normalized activation mean/std;
- attention-output and FFN-output norm relative to residual-stream norm;
- gradient norm per block;
- update/parameter ratio per major parameter group;
- attention-score standard deviation;
- attention entropy by head, summarized rather than dumped;
- fraction of non-finite values;
- training throughput and peak memory.

Do not collect every tensor every step. Diagnostic hooks can dominate runtime and memory. Use a fixed diagnostic batch at sparse intervals.

### Four required broken labs

#### A. Broken causal mask

Allow each position to see the next token.

Prediction: training and validation token loss may become implausibly good because the answer is directly visible. Full-model future-token perturbation must fail.

Lesson: a strong metric can be evidence of leakage, not success.

#### B. Missing attention scaling

Remove $1/\sqrt D$.

Prediction: score variance and maximum attention weight increase with head dimension; entropy tends to fall; gradients may become less well behaved.

#### C. Removed residual path

Remove one or both residual additions in a multi-block model.

Prediction: early-layer gradient flow and optimization worsen, especially with depth. Verify rather than assuming catastrophic failure in a shallow model.

#### D. Tokenizer/checkpoint mismatch

Load weights with a tokenizer whose vocabulary size matches but whose ID assignment differs. Your loader should reject it before generation. If you bypass the guard, outputs are structurally meaningless despite valid tensor shapes.

### Additional quick ablations

Choose only one if time permits:

- zero position embeddings;
- overly large initialization;
- incorrect target shift;
- weight decay applied to norm parameters;
- dropout left active during evaluation;
- accumulation without loss division.

### Bug-report format

```text
Prediction:
Run identity and single intentional change:
Visible symptom:
First bad internal statistic or invariant:
Minimal reproducer:
Root cause:
Regression test:
Would aggregate loss alone reveal it?
Implication for later fine-tuning/inference:
```

### Sampling discipline

Generate from a fixed prefix panel and fixed sampling seeds. Include:

- empty/BOS start;
- a common notice opening;
- a date prefix;
- a rare synthetic surname prefix;
- an OCR-corrupted prefix.

Use greedy or fixed temperature/top-k settings. Samples are diagnostics, not evaluation. A model can improve NLL while a handful of samples look worse, and vice versa.

### Interpretability caution

Attention matrices can reveal where weight was assigned in one layer/head, but they are not a complete causal explanation of the output. Residual paths, value vectors, later layers, and MLPs all matter. Use attention views to debug masks, collapse, and obvious patterns—not as proof that the model “understood” a relationship.

## Chapter 13 — Run a bounded small-domain pretraining experiment

**Classification:** Core — must do  
**Time:** 60 minutes active work plus bounded training wall time  
**Artifact:** selected checkpoint and experiment report

### Question

> Does a small decoder-only Transformer with the frozen BPE tokenizer improve honest held-out language modeling—especially on structured local dependencies—over the Volume I context MLP, while remaining reproducible and diagnostically healthy?

### Baselines

Carry forward:

- smoothed count bigram;
- context MLP;
- byte-only or character representation result;
- optionally a Transformer with byte-only tokenizer under a matched token/compute budget.

Do not compare raw token perplexity across tokenizers. Use bits per byte or another declared base unit.

### Two-stage training plan

#### Stage 1 — correctness run

Use a very small model and tiny data subset:

- no dropout;
- float32;
- fixed batches;
- enough steps to overfit;
- frequent invariant checks.

Exit only when the model can overfit, causality holds, and checkpoint resume works.

#### Stage 2 — bounded experiment

Use the chosen small configuration on the full approved training subset. Freeze:

- tokenizer;
- model configuration;
- total token budget;
- optimizer/schedule;
- validation cadence;
- checkpoint selection rule;
- seed policy.

Allow at most one learning-rate correction after a clearly documented pilot. Do not conduct an open-ended architecture search.

### Validation slices

At minimum preserve or derive:

- dates;
- names/locations or synthetic rare entities;
- umlaut/Unicode;
- OCR corruption;
- long-range synthetic dependencies whose cue lies beyond the Volume I MLP context;
- clean/common text.

Report counts and uncertainty where practical. A slice with 12 events should not be discussed like one with 12,000.

### Expected observations

You should expect, but must verify:

- initial loss near $\log V$;
- rapid tiny-batch overfit;
- Transformer improvement over fixed-context MLP on dependencies beyond the MLP window;
- BPE reducing sequence length relative to bytes/characters;
- rare and corrupted strings remaining more fragmented and difficult;
- training NLL improving before samples become coherent;
- validation flattening or worsening before training loss does;
- naive generation latency increasing with context length.

### Stop rules

Stop when one occurs:

- fixed token budget is exhausted;
- validation metric has failed to improve for the declared patience;
- non-finite state cannot be localized within the debugging budget;
- the model has clearly overfit and the remaining run cannot answer a new question;
- wall-clock/compute cap is reached.

Record the stop reason. “The run ended” is not an experimental condition.

### Minimum report

1. Question and claim boundary.
2. Corpus, privacy class, and grouped split.
3. Tokenizer audit and fingerprint.
4. Model configuration and analytical/measured parameter count.
5. Training token budget and optimizer schedule.
6. Correctness evidence.
7. Train/validation NLL and bits per byte.
8. Slice results.
9. Activation and gradient diagnostics.
10. Throughput, peak memory, and checkpoint bytes.
11. Fixed sample panel.
12. Four broken-run analyses.
13. Limitations and Volume III handoff.

### Memorization and privacy probe

For any run using approved private text, add a bounded local-only probe:

- insert a few synthetic canary strings only into training data and measure whether prompted generation reproduces them;
- compare generated substrings against training text with a simple exact longest-match report;
- inspect whether rare names are reproduced verbatim under training prefixes;
- keep raw matches and generations out of committable reports;
- report only aggregate match lengths/counts unless a private review allows examples.

This is not a complete privacy audit. It establishes that lower NLL and fluent samples can coexist with undesirable memorization, and it creates a regression target for later fine-tuning.

### Platform connection

The checkpoint is an educational domain language model, not a new extraction method. Its immediate useful outputs are:

- a tokenizer stress test for names, dates, locations, and OCR noise;
- a pretraining and checkpointing pipeline you understand;
- a model whose internal mechanics can support LoRA and inference work;
- evidence about which domain slices are hard;
- a baseline for later task-specific fine-tuning.

Do not insert it into `config/extraction_variants.toml`, write predictions to the authoritative database, or compare it with production field F1 as though it had been trained for extraction.

---

# Volume II Capstone — One Transformer you can account for end to end

**Classification:** Core — must do  
**Time:** 2 hours  
**Artifact:** frozen model package, audit, and readiness decision

## Capstone deliverables

### 1. Tokenizer package

- canonical tokenizer file;
- deterministic fingerprint;
- training-split/corpus identity;
- round-trip and save/load tests;
- vocabulary/compression audit;
- fixed domain tokenization panel;
- explicit known weaknesses.

### 2. Model package

- immutable configuration;
- state dictionary;
- tokenizer fingerprint guard;
- analytical and measured parameter count;
- maximum context and position policy;
- model-card-style claim boundary.

### 3. Trainer package

- reproducible run config;
- metrics ledger;
- optimizer/scheduler state;
- atomic checkpoints;
- deterministic resume test within declared scope;
- best-checkpoint selection record.

### 4. Correctness evidence

Required tests:

- BPE hand fixture and deterministic ties;
- Unicode round trips;
- no cross-document merge counts;
- tokenizer save/load identity;
- attention row/mask invariants;
- future-token perturbation at attention and full-model levels;
- explicit/fused multi-head forward and gradient agreement;
- manual/framework LayerNorm agreement;
- block identity when sublayer weights are zero;
- initial loss near $\log V$;
- tiny-batch overfit;
- checkpoint resume equivalence;
- tokenizer mismatch rejection.

### 5. Frozen comparison

Use a compact table:

| Model | Representation | Context | Parameters | Train bpb | Validation bpb | Date slice | Rare-entity slice | OCR-noise slice | Tokens/s | Peak memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count bigram | Volume I unit | 1 | ... | ... | ... | ... | ... | ... | ... | ... |
| Context MLP | Volume I unit | fixed $T$ | ... | ... | ... | ... | ... | ... | ... | ... |
| Transformer | frozen BPE | $T_{max}$ | ... | ... | ... | ... | ... | ... | ... | ... |

Do not fill missing values with invented numbers. Use `not measured` and explain why.

## Predictions before the capstone run

1. Which tokenizer gives the best validation bits per byte under the chosen model budget?
2. Which parameter group dominates model size?
3. Which slice gains most over the context MLP?
4. Which slice remains weakest?
5. Which layer has the largest gradient norm early in training?
6. Does attention entropy become more or less concentrated with training?
7. At what context length does naive generation noticeably slow?
8. Will tied embeddings help parameter efficiency without hurting the bounded run?

## Required ablation discipline

Run only comparisons that answer a named question. At minimum include:

- one tokenizer-size comparison using compression and model-cost evidence;
- one positional-information ablation on a constructed order task;
- one scaling or residual failure lab;
- one baseline comparison against the context MLP.

Do not sweep heads, layers, widths, dropout, schedules, context, tokenizer size, and seeds simultaneously. The core is about mechanisms, not leaderboard search.

## Final interpretation

Your report should distinguish:

### What is now owned

- how Unicode becomes bytes, merged token IDs, and embeddings;
- how positions enter the model;
- how Q, K, and V create a causal weighted mixture;
- how heads are combined;
- why scaling, normalization, residuals, and FFNs exist;
- how logits and next-token loss arise at every position;
- how parameters are updated, checkpointed, and resumed;
- where sequence length, vocabulary size, parameters, and attention memory enter cost.

### What is not established

- instruction following;
- structured extraction quality;
- factuality or provenance;
- rare-name preservation in generated/extracted fields;
- robustness under realistic OCR distribution shift;
- usefulness relative to current provider-backed variants;
- efficient autoregressive serving.

That gap is not a failure. It defines Volume III and Volume IV.

## Volume III handoff

Volume III should begin from this model and answer:

- How does pretraining objective differ from supervised instruction/extraction objectives?
- Which pretrained base model is appropriate for useful local fine-tuning?
- What does a low-rank update change mathematically?
- How should domain examples be formatted without leaking answers or provenance?
- Does fine-tuning improve extraction, grounded answering, or OCR cleanup under held-out tests?
- What is forgotten or degraded?
- How do deterministic metrics, human review, and model judges disagree?

Do not implement those in this volume.

---

# Volume II Mastery Examination

## A. Explain without notes

You should be able to explain:

1. Why UTF-8 bytes guarantee coverage but not compact tokenization.
2. How one BPE merge is selected and applied.
3. Why tie-breaking and merge rank are tokenizer identity.
4. Why BPE must not learn across document boundaries.
5. Why token perplexity is not directly comparable across tokenizers.
6. How vocabulary size affects embeddings, output heads, and sequence length.
7. Every tensor shape in single-head attention.
8. Why scores are divided by $\sqrt D$.
9. Why the causal mask is applied before softmax.
10. How to prove no future-token leakage with perturbation.
11. How multi-head attention reshapes and combines heads.
12. Why an output projection follows concatenation.
13. What positional information contributes.
14. The difference between attention’s cross-position mixing and the FFN’s per-position transformation.
15. Why residual paths help gradients and representation flow.
16. What LayerNorm normalizes and why the axis matters.
17. Why initial cross-entropy should be near $\log V$.
18. Why gradient accumulation requires loss scaling.
19. What must be stored to resume training faithfully.
20. Why naive generation recomputes work and gets slower with context.
21. Why a lower validation language-model loss is not evidence of accurate structured extraction.

## B. Reconstruct from a blank editor

In one focused session:

1. implement pair counting and non-overlapping BPE replacement;
2. encode and decode a Unicode fixture through byte expansions;
3. write Q, K, V projections for one attention head;
4. apply scaling, a causal mask, softmax, and value mixing;
5. extend shapes to multiple heads;
6. write one pre-norm Transformer block;
7. assemble token/position embeddings, two blocks, final norm, and LM head;
8. run a future-token perturbation test;
9. overfit one tiny batch.

You may consult API names and tensor-operation syntax. You may not copy the finished module.

## C. Debug blind

Have another person or your future self inject two bugs:

- nondeterministic BPE tie-breaking;
- encoding merges in arbitrary order;
- cross-document merge counting;
- softmax on the query dimension;
- upper-triangular mask used in the wrong orientation;
- scale by $\sqrt C$ instead of $\sqrt D$;
- QKV split along the sequence axis;
- LayerNorm over time;
- target not shifted;
- tokenizer ID mapping changed on reload;
- accumulation loss not divided;
- scheduler step restored incorrectly.

Diagnose with invariants and the first bad tensor, not by scanning every line.

## D. Design

Design an experiment answering:

> Does OCR corruption increase representation burden and language-model uncertainty more for rare names than for common prose?

Specify:

- deterministic corruption types and severities;
- synthetic/approved data policy;
- tokenizer metrics;
- model metric normalized to bytes;
- paired clean/corrupt examples;
- rare/common slice definition fixed before results;
- uncertainty or seed policy;
- claim boundary.

## Passing standard

Proceed to Volume III only when:

- tokenizer round trips and identity checks pass;
- vocabulary size is justified by a domain audit;
- single-head matrices and shapes can be reconstructed from memory;
- explicit and fused multi-head implementations agree;
- causality perturbation passes through the complete model;
- a tiny Transformer overfits a tiny batch;
- the bounded domain run has a frozen config and honest validation result;
- checkpoint resumption works within the declared deterministic scope;
- all four required broken labs have regression tests;
- the report states what is mechanism evidence versus product evidence;
- test data remained sealed until model/tokenizer selection ended;
- no private raw text or memorized generation entered committable artifacts.

If a Transformer trains but its mask has not been independently tested, this volume is not complete. If the final model is modest but every representation, tensor, gradient path, and checkpoint transition is explainable, the objective has been met.

---

# Compact formula and shape reference

## Tokenizer-neutral language-model metric

$$
\mathrm{bits\ per\ byte}
=
\frac{-\sum_t\log_2 p(z_t\mid z_{<t})}{N_{bytes}}.
$$

## Attention

$$
Q=XW_Q,
\qquad K=XW_K,
\qquad V=XW_V.
$$

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right).
$$

$$
Y=AV.
$$

For one head:

```text
X       [B,T,C]
Q,K,V   [B,T,D]
scores  [B,T,T]
A       [B,T,T]
Y       [B,T,D]
```

For $H$ heads with $D=C/H$:

```text
Q,K,V        [B,H,T,D]
scores       [B,H,T,T]
head output  [B,H,T,D]
concat       [B,T,C]
projection   [B,T,C]
```

## Layer normalization

$$
\operatorname{LN}(x)_i
=
\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

## Pre-normalized block

$$
x_1=x+\operatorname{MHA}(\operatorname{LN}(x)),
$$

$$
x_2=x_1+\operatorname{FFN}(\operatorname{LN}(x_1)).
$$

## Cross-entropy baseline

For uniform logits over vocabulary size $V$:

$$
L_{init}\approx\log V.
$$

## Training-token budget

$$
N_{tokens}=\text{steps}\times B\times T\times\text{accumulation steps}.
$$

## Approximate dense parameters per block

Ignoring biases and norms for a 4× FFN:

$$
P_{attn}\approx4C^2,
\qquad
P_{ffn}\approx8C^2,
\qquad
P_{block}\approx12C^2.
$$

## Attention score storage

Materialized score elements:

$$
BHT^2.
$$

Approximate bytes for one score tensor at $s$ bytes per element:

$$
BHT^2s.
$$

Backward may require additional saved tensors; this is not total training memory.

---

# Run checklist

## Before tokenizer training

- [ ] training documents only;
- [ ] document boundaries preserved;
- [ ] Unicode/preprocessing policy recorded;
- [ ] deterministic tie-break defined;
- [ ] special-token IDs reserved;
- [ ] target vocabulary justified as a candidate, not assumed.

## Before model training

- [ ] tokenizer frozen and fingerprinted;
- [ ] corpus/split fingerprint recorded;
- [ ] several blocks decoded and target shift inspected;
- [ ] parameter count derived and measured;
- [ ] expected initial loss calculated;
- [ ] full-model causality test passes;
- [ ] tiny-batch overfit passes;
- [ ] token budget and stop rule fixed;
- [ ] validation selection metric fixed;
- [ ] test sealed.

## During training

- [ ] loss and learning rate logged;
- [ ] unclipped gradient norm logged;
- [ ] non-finite guard active;
- [ ] sparse depth diagnostics collected;
- [ ] tokens/s and peak memory measured;
- [ ] validation uses evaluation mode and fixed data;
- [ ] checkpoint writes are atomic;
- [ ] no private samples enter general logs.

## After training

- [ ] best checkpoint reloads with exact tokenizer guard;
- [ ] fixed-batch logits reproduce;
- [ ] validation and slice metrics reported with counts;
- [ ] cross-tokenizer results normalized to bytes;
- [ ] fixed sample panel generated under recorded settings;
- [ ] four broken-run reports completed;
- [ ] limitations and claim boundary stated;
- [ ] test opened only for the frozen final comparison.

---

# Optional Deep Dives — outside the 15-hour core

Choose only when they serve a real next objective:

- efficient incremental BPE training and encoding;
- unigram language-model tokenization;
- Unicode normalization and grapheme-cluster analysis;
- rotary or relative position embeddings;
- grouped-query or multi-query attention;
- RMSNorm and gated/SwiGLU feed-forward networks;
- framework fused scaled-dot-product attention;
- activation checkpointing;
- mixed-precision numerical analysis;
- scaling-law experiments;
- custom attention kernels;
- distributed pretraining.

These topics are not omitted because they are unimportant. They are postponed because none is required to understand and train the first complete Transformer.

---

# End of Volume II

Stop here. Volume III begins with the difference between pretraining and task adaptation, implements the low-rank update behind LoRA, then uses a maintained fine-tuning stack and builds the first rigorous Family History evaluation suite.
