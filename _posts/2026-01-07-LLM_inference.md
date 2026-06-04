# NVIDIA Solutions Architect — Study Notes

---

## 1. Why LLM inference is hard

Running LLMs in production has one core problem: **the GPU is mostly idle during token generation**.

During the **decode phase**, generating each new token requires loading all model weights from HBM memory into SRAM (on-chip) every single step. The actual math is tiny relative to the data movement — this makes decode extremely **memory-bandwidth-bound**, not compute-bound.

### Memory hierarchy
- **HBM** (High Bandwidth Memory) — large storage next to the chip. Weights live here (~80GB on H100).
- **SRAM / registers** — tiny, ultra-fast memory on the chip inside SMs where actual math happens.
- Weights must travel **HBM → SRAM → Tensor Cores** every token, because SRAM is too small to hold all weights permanently.

### Why weights can't stay in SRAM
Weights are processed in **tiles** (small chunks):
1. Load a tile of weights from HBM → SRAM
2. Multiply that tile with current token's activations
3. Accumulate partial result
4. Discard tile, load next one
5. Repeat until all weights processed

The GPU tries to pipeline this (fetch tile N+1 while computing tile N) but for LLM decode the tiles are so small relative to fetch time that compute finishes and has to **wait** for the next tile.

### Concrete example
- H100 HBM bandwidth: ~3.35 TB/s
- 70B model in FP16 = ~140GB of weights
- At 3.35 TB/s → ~42ms per token just in memory transfers
- Tensor Cores finish their math almost instantly and sit **waiting**

### KV cache vs model weights — important distinction
- **Model weights** — billions of parameters, never change, must be streamed from HBM every decode step
- **KV cache** — saved key/value vectors from attention for previous tokens in the sequence, avoids recomputing attention over past tokens
- KV cache does NOT eliminate weight loading — these are two separate things

---

## 2. Batch size and arithmetic intensity

**Arithmetic intensity** = FLOPs ÷ bytes transferred

- Batch size 1: stream all 140GB weights, do math against one token → very low arithmetic intensity → memory-bound
- Batch size 32: stream the same 140GB weights, do math against 32 tokens simultaneously → 32× higher arithmetic intensity → Tensor Cores stay busier

**Same memory cost, 32× the output.**

### Why not always use huge batch sizes?
1. **Latency** — you have to wait until you have 32 requests before starting
2. **KV cache memory** — each request needs its own KV cache, growing with sequence length

---

## 3. Static batching and its problems

Naive approach: wait for N requests, process all together, repeat.

### Two kinds of waste
1. **Waiting waste** — requests queue before the batch fills up (request 1 arrives at t=0ms, waits until request 32 arrives at t=3000ms)
2. **Padding waste** — short requests finish early but their GPU slot sits empty waiting for the slowest request

---

## 4. Continuous batching

Instead of a rigid "wait for N, process all, repeat" cycle, the batch is a **rolling window** — as soon as one sequence finishes, a new request slides in immediately.

### How it works
Each decode step, process whatever requests are currently in-flight:
- Step 1: requests A, B, C generate their next token
- Step 3: A finishes → request D boards immediately → B, C, D continue

**No waiting. No empty slots.**

### Why this is possible
Each decode step is a discrete unit — generate one token and pause. That natural pause point is the perfect moment to check: has anyone finished? is anyone new waiting?

### Key properties
- No batch formation phase — the moment one request arrives, it starts processing
- Batch size at any moment = however many requests happen to be in-flight
- GPU utilization goes from `[===busy===][idle][===busy===]` to `[==========always busy==========]`

---

## 5. KV cache memory management

### Static batching KV cache
Pre-allocate for worst case:
```
32 requests × max_sequence_length × 2 (K and V) = total allocation
```
Most of that allocation sits empty — paying for worst case for everyone, always.

### The paged KV cache problem it solves
With continuous batching, requests join and leave constantly, each with a KV cache of unpredictable size. Pre-allocation becomes nearly impossible.

### Paged KV cache (borrowed from OS virtual memory)
- HBM is divided into fixed-size **pages**
- Each request gets pages allocated **on demand**, one at a time, as tokens are generated
- Pages from different requests are scattered through HBM
- A **page table** tracks where everything is

**Result**: instead of pre-allocating 1000 pages per request "just in case", you allocate pages one at a time. Requests that finish early free their pages immediately for new requests.

### Memory capacity calculation example
- H100: 80GB total
- 7B model in FP16: ~14GB for weights
- ~2GB for runtime overhead
- ~64GB remaining for KV cache
- Each page holds 4 tokens, costs 1MB → 64,000 pages total
- Max 4000 tokens per request → 1000 pages worst case per request
- Worst case: 64 simultaneous requests
- Average case (400 tokens/request): 640 simultaneous requests — **10× more**

### What lives in HBM
```
Total HBM = model weights (permanent)
          + CUDA runtime overhead (~1-2GB)
          + activations during forward pass (transient, small)
          + KV cache (dynamic, lives only during prefill and decode)
```

### KV cache lifetime per conversation turn
```
1. Full history sent to model
2. Prefill — reprocess everything, build KV cache from scratch
3. Decode — generate tokens one by one, extend KV cache
4. Done — discard KV cache, wait for next message
```
KV cache is rebuilt every single turn. As conversation grows longer, **prefill gets more expensive** each turn.

### Deadlock risk and prevention
If all pages are in use and no request is close to finishing, everyone waits → **deadlock**.

Prevention strategies:
- **Admission control** — never fill to 100%, maintain a buffer
- **Preemption** — proactively swap low-priority requests to CPU RAM before deadlock forms
- **Reservation** — each admitted request gets a soft page guarantee

---

## 6. Prefill vs decode

| | Prefill | Decode |
|---|---|---|
| What happens | Process entire input prompt at once | Generate tokens one at a time |
| Bound by | Compute | Memory bandwidth |
| KV cache | Built from scratch | Extended one token at a time |
| Cost grows with | Prompt length | Output length |

---

## 7. Tensor parallelism

Used when model weights **don't fit on a single GPU**.

### Core idea
Instead of splitting by layer, split **within each layer** — each weight matrix is divided column-wise across GPUs.

```
GPU 1: input × left half of weights → partial output 1
GPU 2: input × right half of weights → partial output 2
→ all-reduce combines partial results
→ both GPUs have full output
→ move to next layer
```

### Key properties
- Both GPUs work **in parallel** on the same input
- Each GPU holds **half the weights**
- Input is **identical on all GPUs** at the start of each layer
- Output is **identical on all GPUs** after all-reduce
- Split must be **equal** across GPUs (always power of 2: 2, 4, 8) to avoid load imbalance

### Communication frequency
Every major matrix multiplication requires an all-reduce:
```
~4-6 all-reduces per layer × 80 layers = 320-480 all-reduces per token
```

This is why **NVLink is mandatory** for tensor parallelism (900 GB/s vs PCIe's 64 GB/s — 14× difference).

### Tradeoff
High communication overhead — but GPUs work in parallel.

### Practical limit
Works great within a single node (8 GPUs via NVLink). Beyond that, communication dominates.

---

## 8. Pipeline parallelism

### Core idea
Split **by layers** across GPUs:
```
GPU 1: layers 1-20
GPU 2: layers 21-40
GPU 3: layers 41-60
GPU 4: layers 61-80
```

Each GPU owns its layers completely. GPU 1 processes, passes activations to GPU 2, and so on.

### Communication frequency
Only at handoff points between GPUs:
```
4 GPUs → 3 handoffs per token (N-1 communications)
```

Very low communication — can tolerate slow network connections between nodes.

### The pipeline bubble problem
At any given moment only one GPU is working:
```
GPU 1 working → GPU 2, 3, 4 idle
GPU 2 working → GPU 1, 3, 4 idle
...
```
75% GPU idle time with 4 GPUs.

### Solving the bubble with micro-batching
Send multiple micro-batches back to back without waiting:
```
Time →
GPU 1: [MB1][MB2][MB3][MB4]
GPU 2:      [MB1][MB2][MB3][MB4]
GPU 3:           [MB1][MB2][MB3][MB4]
GPU 4:                [MB1][MB2][MB3][MB4]
```
More micro-batches → smaller bubble as fraction of total time:
```
4 micro-batches  → ~25% bubble
8 micro-batches  → ~12.5% bubble
32 micro-batches → ~3% bubble
```

### Note: pipeline parallelism is inference-only concern here
Backward propagation and weight updates are a **training** concern. At inference, weights are frozen — no backward pass, no gradient computation, no weight updates.

---

## 9. Combining both parallelism strategies

| Strategy | Communication | GPU utilization | Best for |
|---|---|---|---|
| Tensor parallelism | High (320-480 all-reduces/token) | High (parallel) | Within a node, NVLink |
| Pipeline parallelism | Low (N-1 handoffs/token) | Low without micro-batching | Across nodes, slow network |

### Combined deployment (e.g. 16 GPUs, 2 nodes)
```
Within each node (8 GPUs, NVLink) → tensor parallelism
Between nodes (slow network)      → pipeline parallelism
```
Play to each strategy's strength:
- Tensor parallelism where communication is fast
- Pipeline parallelism where communication is slow

---

## 10. Quantization

### The core idea
Weights don't need full FP16 precision at inference time. Reducing precision reduces memory and speeds up memory bandwidth:

```
70B model:
FP16 → 140GB
INT8 → 70GB  (2× smaller)
INT4 → 35GB  (4× smaller)
```

Attacks two problems simultaneously:
- **Memory capacity** — fits larger models on fewer GPUs
- **Memory bandwidth** — streams weights faster → more tokens per second

### How numbers are stored

**INT8**: 8-bit integer, 256 distinct values (-128 to 127). To store a decimal weight, scale the entire layer's range to fit:
```
weight 0.003847, layer range [-0.8, +0.8]
→ 0.003847 × (127 / 0.8) = 0.61 → rounds to 1
→ stored as integer 1
→ recovered as 1 × (0.8 / 127) = 0.0063  (small error)
```

**FP16**: 16-bit float with three components:
```
1 bit  → sign
5 bits → exponent (scale/magnitude)
10 bits → mantissa (significant digits)
```
Can represent huge range of values with reasonable precision everywhere.

**BF16**: trades precision for range:
```
1 bit  → sign
8 bits → exponent (wider range than FP16)
7 bits → mantissa (less precise than FP16)
```
Better for deep learning — wide range avoids overflow, precision less critical.

### Accuracy vs precision tradeoff
```
FP16  → baseline, full accuracy
BF16  → nearly identical to FP16
FP8   → very close to FP16 (Hopper GPUs H100+)
INT8  → small accuracy loss, usually acceptable
INT4  → noticeable loss, but manageable with modern methods
INT2  → significant degradation
1bit  → model largely breaks down
```
Loss is **not linear** — FP16→INT8 loses very little, INT4→INT2 loses a lot.

### Why INT4 is the current sweet spot
```
FP16 → INT4:
- 4× memory reduction
- 4× bandwidth improvement
- Acceptable accuracy with modern quantization methods
```
A 70B model needing 140GB in FP16 fits in **35GB** in INT4.

---

## 11. GPTQ (smart quantization)

### Problem with naive quantization
Blindly rounding every weight to nearest INT4 value treats all weights equally — wastes precision on unimportant weights while being too aggressive on critical ones.

### The Hessian — sensitivity map
The Hessian matrix captures how sensitive the model output is to each weight:
```
High H[i][i] → weight i is very sensitive → rounding it will hurt accuracy
Low H[i][i]  → weight i is insensitive → rounding barely matters
```
Think of it as a **map of which weights matter most**.

### GPTQ algorithm
Instead of rounding all weights independently, GPTQ uses error compensation:

```
1. Look at Hessian → find least sensitive weight first
2. Round that weight to INT4
3. Measure the rounding error introduced
4. Adjust remaining (still FP16) neighboring weights to compensate
5. Repeat for next least sensitive weight
```

### Error compensation example
```
w1 = 0.386 rounds to 0.375 → error of +0.011
→ w2 adjusted by -0.004 (still FP16, has room)
→ w3 adjusted by -0.005
→ w4 adjusted by -0.002
→ net error ≈ 0
```

Errors are spread across neighbors and cancelled out **before they get quantized themselves**.

**Result**: same INT4 bits, much better accuracy than naive rounding.

---

## Key concepts to be able to explain out loud

1. Why is decode memory-bandwidth-bound and not compute-bound?
2. What is arithmetic intensity and how does batch size affect it?
3. What is the difference between static and continuous batching?
4. What is paged KV cache and what OS concept does it borrow from?
5. What is the difference between tensor parallelism and pipeline parallelism?
6. Why does tensor parallelism require NVLink?
7. How do micro-batches reduce the pipeline bubble?
8. What does quantization trade off and where is the practical limit?
9. What does GPTQ do differently from naive quantization?
10. Walk through an end-to-end LLM deployment story.

---

## Topics still to cover
- FP8 quantization (Hopper-specific)
- AWQ quantization
- Speculative decoding
- NeMo framework
- Triton inference server (dynamic batching, model ensembles, config.pbtxt)
- TensorRT build pipeline (ONNX → engine)

---

## 12. Speculative decoding

### The problem it solves
Decode generates one token per forward pass — the most expensive operation (streaming all weights from HBM) for the least output (one token). Tensor Cores are massively underutilized.

### The core idea
Use a small cheap **draft model** to guess multiple tokens, then verify them all in one forward pass of the large **target model**.

### The two models
- **Draft model** — small, fast (e.g. 1B parameters). Generates K candidate tokens cheaply.
- **Target model** — large, accurate (e.g. 70B parameters). Verifies all K candidates in one pass.

### The flow
```
Step 1: Draft model generates K=4 candidate tokens cheaply
Input: "The capital of France is"
Draft guesses: ["Paris", ",", "which", "is"]

Step 2: Target model verifies all K tokens in ONE forward pass
Input to target: "The capital of France is Paris , which is"
→ processes like a prefill (parallel, compute-bound, efficient)
→ produces probability distribution at each position

Step 3: Accept or reject each token
"Paris" → target also predicted Paris → ✓ accept
","     → target also predicted ","   → ✓ accept
"which" → target predicted "the"      → ✗ reject, take target's suggestion
```

### The speedup
```
K separate decode steps → K × HBM bandwidth cost
1 verification pass of K tokens → ~1× HBM bandwidth cost (prefill, not decode)
```
Up to K+1 tokens from just 1 target model forward pass.

### Why it works for RAG chatbots specifically
FAQ answers are predictable and formulaic:
```
"Based on our policy, you should..."
"According to the FAQ, the answer is..."
```
Predictable text = high draft model acceptance rate = maximum speedup.

### Relationship to other optimizations
```
Quantization       → reduce bandwidth cost itself (140GB → 35GB per pass)
Continuous batching → maximize utilization across requests
Speculative decoding → get more tokens per bandwidth consumed per request
```
Three complementary levers attacking the same bottleneck.

---

## 13. TensorRT build pipeline

### The problem with raw PyTorch at inference
- Python overhead on every operation
- No kernel fusion — each operation is a separate GPU kernel launch
- No hardware-specific tuning
- Dynamic graph — hard to optimize ahead of time

### The four-stage pipeline
```
Trained model (PyTorch/TF)
    ↓
Export to ONNX (intermediate format)
    ↓
TensorRT optimization
    ↓
Optimized engine (.plan file)
    ↓
Deploy via Triton
```

### What is ONNX?
Open Neural Network Exchange — a universal intermediate format. Like PDF for documents:
```
PyTorch    ──→ ┐
TensorFlow ──→ ┤ ONNX → TensorRT
JAX        ──→ ┘
```
Contains: all operations, how they connect, input/output shapes, weight values.

### TensorRT optimizations

**Layer fusion**: merge sequential operations into one kernel
```
Conv + BatchNorm + ReLU → one single kernel
→ data loads once, all ops happen in SRAM, writes back once
```
For LLMs: LayerNorm + matmul, matmul + bias + activation, QKV projections all fused.

**Kernel auto-tuning**: benchmark multiple CUDA kernel implementations for each operation on your specific GPU, bake the fastest one into the engine. This is why building takes time and why an H100 engine won't be optimal on an A100.

**Precision calibration**: for INT8, run calibration dataset through network, record activation ranges at each layer, compute optimal scale factors, bake into engine.

**Memory planning**: pre-allocate all memory needed at build time → zero dynamic allocation at inference time → removes latency variance.

### Output: the .plan file
- Specific to your GPU model
- Specific to chosen precision (FP16/INT8)
- Specific to input shapes
- Not portable — must rebuild for different hardware

---

## 14. TensorRT-LLM full stack

### Why standard TensorRT isn't enough for LLMs
- Dynamic sequence lengths
- Autoregressive generation (output feeds back as input)
- Massive KV cache memory requirements
- Need for continuous batching
- Multi-GPU tensor/pipeline parallelism

### TensorRT-LLM workflow
```
Pretrained weights (HuggingFace, etc.)
    ↓
TensorRT-LLM model definition (hand-optimized per architecture)
    ↓
trtllm-build (engine builder)
    ↓
TensorRT-LLM engine (one file per GPU)
    ↓
TensorRT-LLM runtime
    ↓
Triton inference server
```

### trtllm-build key parameters
```
--dtype        → fp16, bf16, int8, int4
--tp_size      → tensor parallelism degree
--pp_size      → pipeline parallelism degree
--max_batch_size
--max_input_len
--max_output_len
```

### TensorRT-LLM runtime manages
- Continuous batching scheduler
- Paged KV cache allocator
- Token sampling (greedy, top-p, top-k, beam search)
- Multi-GPU coordination (all-reduces, pipeline handoffs)
- Speculative decoding
- Flash Attention

### Full serving stack
```
Client request (HTTP/gRPC)
    ↓
Triton (queuing, request-level batching)
    ↓
TensorRT-LLM runtime (continuous batching, KV cache, sampling)
    ↓
TensorRT-LLM engine (optimized kernels, fused ops, quantized weights)
    ↓
GPU hardware (Tensor Cores, HBM)
```

---

## 15. Flash Attention

### The problem with standard attention
For a sequence of N tokens, the attention score matrix is N×N. Standard attention:
1. Computes all scores → writes N×N matrix to HBM
2. Reads N×N matrix back from HBM for softmax
3. Writes normalized scores back to HBM
4. Reads back for multiplication with V

4 HBM round trips. O(N²) memory reads/writes — grows quadratically with sequence length.

### Why naive tiling doesn't work
Softmax requires seeing ALL scores before normalizing any of them:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```
Example with 4 tokens, scores = [2.0, 1.0, 3.0, 0.5]:
- Tile 1 sees [2.0, 1.0] → partial softmax = [0.73, 0.27] ← WRONG
- Tile 2 arrives with [3.0, ...] → score 3.0 dominates everything → real softmax completely different
- Each new tile can invalidate all previous partial results

### Flash Attention's solution: online softmax + tiling
Process attention in small tiles entirely within SRAM. Maintain two running statistics:
- Running maximum score seen so far
- Running sum of exp(scores) seen so far

As each new tile arrives, **rescale previous partial results** using updated statistics:
```
old partial sum = 10.11
new full sum = 31.85
rescale factor = 10.11 / 31.85
→ go back and rescale tile 1 output
→ add tile 2 contribution
→ mathematically identical to full softmax
```
All rescaling happens in SRAM — no HBM writes needed.

### Result
```
Standard attention: O(N²) HBM reads/writes
Flash Attention:    O(N)  HBM reads/writes
```
Full N×N matrix never written to HBM. Same mathematical result, dramatically less memory traffic.

### Why it matters for RAG chatbots
Long retrieved context = long sequences = standard attention becomes a bottleneck.
Flash Attention makes long context practical. Used by default in TensorRT-LLM.

---

## 16. Production RAG chatbot deployment

### Full pipeline
```
User question
    ↓
TensorRT INT8 embedding model (Triton, dynamic batching)
    ↓
cuVS CAGRA vector search (GPU-accelerated ANN)
    ↓
Retrieved FAQ chunks
    ↓
TensorRT-LLM (quantization + continuous batching + speculative decoding + prompt caching)
    ↓
Answer
```

### Stage 1: Offline document preparation
- Chunk FAQ documents carefully — chunk size directly affects KV cache usage at inference
- Good chunking = smaller context = cheaper prefill = less KV cache pressure
- Embed all chunks using embedding model → store in vector DB
- This is done **once offline** — never repeated at inference time

### Stage 2: Embedding model (query)
- Model type: encoder model (e.g. sentence-transformers, E5, BGE) — NOT an LLM
- Optimization: standard TensorRT (not TensorRT-LLM — no autoregressive generation)
- INT8 quantization, dynamic batching via Triton
- Fixed shape advantage: one forward pass per query, no decode loop, no KV cache needed

### Stage 3: Vector retrieval (cuVS / CAGRA)
- Exact nearest neighbor search doesn't scale to millions of vectors
- CAGRA: graph-based ANN search — build graph offline, traverse at query time
- Parallelized within a single GPU via thousands of CUDA threads
- Alternative: IVF (Inverted File Index) — faster index rebuild, better for dynamic corpora
- Library: NVIDIA cuVS (not TensorRT, not NeMo)

### Stage 4: LLM inference (TensorRT-LLM)
- Quantization: INT8 or INT4 (FAQ task tolerates it — answers are factual, grounded)
- Continuous batching: many employees sending questions simultaneously
- Speculative decoding: predictable FAQ answers = high acceptance rate = large speedup
- Prompt caching: system prompt is identical for every request → cache its KV cache permanently
- Flash Attention: long retrieved context becomes feasible

### Stage 5: Serving (Triton ensemble)
- Chain all steps into one endpoint — client sends question, gets answer
- No multiple API calls from client
- Metrics, monitoring, versioning via Triton

### Why NVIDIA's stack specifically
Every stage has a GPU-accelerated NVIDIA solution — cuVS for retrieval, TensorRT for embedding, TensorRT-LLM for the LLM, all served through Triton. The entire pipeline stays on GPU with minimal CPU bottlenecks and data transfers.

---

## 17. NeMo framework

### Where it fits
```
NeMo         → training and fine-tuning models
TensorRT-LLM → optimizing and running models at inference
Triton       → serving models in production
```

### What NeMo provides
- Distributed training across many GPUs out of the box
- Built-in support for popular LLM architectures
- Tensor/pipeline parallelism during training
- Fine-tuning methods: full fine-tuning, LoRA, P-tuning
- Data preprocessing pipelines
- Direct export to TensorRT-LLM format

### Why NeMo over plain PyTorch
Plain PyTorch requires manual implementation of tensor parallelism, pipeline parallelism, gradient checkpointing, mixed precision. NeMo provides all of this via config files.

---

## 18. LoRA (Low Rank Adaptation)

### Full fine-tuning problem
Update every weight in the model:
```
70B model → 70B gradients → 70B optimizer states → enormous GPU memory
```
Also risks catastrophic forgetting — model loses general knowledge.

### LoRA core idea
Add small trainable adapter matrices alongside frozen original weights:
```
Original weight W: 4096 × 4096 = 16M parameters (frozen)
LoRA matrices A + B: (4096×8) + (8×4096) = 65K parameters (trained)
→ 245× fewer trainable parameters
```

### How it works
```
During training:
Output = W×input + A×B×input
W frozen, only A and B updated

During inference:
Merge: W_final = W + A×B
→ no extra computation, same speed as original model
```

### Mathematical insight
Any weight update matrix ΔW can be approximated as product of two smaller matrices:
```
ΔW (large) ≈ A × B (two small matrices)
```
This works because weight updates during fine-tuning tend to be low-rank in practice.

### For the RAG chatbot
Fine-tuning goal: learn company answer format, tone, FAQ citation style.
```
Full fine-tuning → overkill, expensive, risks forgetting
LoRA → cheap, fast, preserves general knowledge, learns style well
```

### Full fine-tuning story
```
Company FAQ Q&A pairs
→ NeMo with LoRA (adapter_dim=8)
→ Train for few epochs on 1-2 GPUs
→ Merge LoRA weights into base model
→ Export to TensorRT-LLM
→ Deploy via Triton
```

---

## Updated: Key concepts to explain out loud

1. Why is decode memory-bandwidth-bound and not compute-bound?
2. What is arithmetic intensity and how does batch size affect it?
3. What is the difference between static and continuous batching?
4. What is paged KV cache and what OS concept does it borrow from?
5. What is the difference between tensor parallelism and pipeline parallelism?
6. Why does tensor parallelism require NVLink?
7. How do micro-batches reduce the pipeline bubble?
8. What does quantization trade off and where is the practical limit?
9. What does GPTQ do differently from naive quantization?
10. How does speculative decoding work and why is it good for FAQ chatbots?
11. What is Flash Attention and what problem does it solve?
12. Walk through the full RAG chatbot production deployment end to end.
13. What is LoRA and why use it instead of full fine-tuning?
14. What is the NeMo → TensorRT-LLM → Triton workflow?
