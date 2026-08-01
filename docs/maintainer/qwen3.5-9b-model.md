# Qwen3.5-9B Model Reference

This reference records the exact BF16 checkpoint's Text, MTP, Vision,
multimodal-position, numeric, and persistent-state semantics used by the
registered `qwen3_5_9b` target. It does not define artifact bytes, weight
loading, packing or quantization, execution-Op design, or qualify the advertised
extended million-token mode. The registered artifact and quantization
specification is [`qwen3.5-9b-artifact.md`](qwen3.5-9b-artifact.md).

## 1. Model identity

Qwen3.5-9B is a dense multimodal model in the Qwen3.6 family. It uses the
Hugging Face architecture identifiers `Qwen3_5ForConditionalGeneration`,
`qwen3_5`, and `qwen3_5_text`. Those identifiers name the implementation
family; they do not indicate a separate product line.

It is a post-trained dense model with three checkpoint components:

- a 32-layer hybrid Text decoder with a dense SwiGLU in every layer;
- a one-layer MTP draft model whose decoder block also contains a dense SwiGLU;
- a 27-layer Vision transformer and patch merger.

The fixed shape and tensor inventory below are checkpoint facts. A different
hidden size, layer count, head layout, or Vision tower is a different exact
model profile rather than a runtime configuration of this one.

## 2. Global dimensions

### 2.1 Text decoder

| Field | Value |
|---|---:|
| hidden size | 4096 |
| decoder layers | 32 |
| intermediate size | 12288 |
| vocabulary matrix rows | 248320 |
| tokenizer-addressable tokens | 248077 |
| full-attention interval | 4 |
| full-attention layers | 8 |
| Gated DeltaNet layers | 24 |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1e7` |
| checkpoint position capacity | 262144 |
| MTP hidden layers | 1 |

Layer `i` is full attention exactly when `(i + 1) % 4 == 0`, giving zero-based indices
`3, 7, 11, 15, 19, 23, 27, 31`; every other layer is GDN. All 32 layers apply the same
dense SwiGLU structure after their token mixer. There is no MoE.

The input embedding and output head are untied, independent matrices of shape `[248320,4096]`.
The 248320 rows are the padded model vocabulary width. Tokenizer ids end at 248076, so rows
248077 through 248319 are padded/reserved model rows without corresponding tokenizer tokens.

The source config has no RoPE-scaling block and directly defines 262144 positions.

### 2.2 Full gated attention

| Field | Value |
|---|---:|
| query heads | 16 |
| KV heads | 4 |
| head dimension | 256 |
| Q width | 4096 |
| output-gate width | 4096 |
| raw `q_proj` rows | 8192 |
| K/V width | 512 each |
| rotated dimensions per head | 64 |
| Q heads per KV head | 4 |
| attention scale | `1/sqrt(256) = 0.0625` |

The query projection is logically a per-head `[query_256 | output_gate_256]` projection. Query and
gate values are interleaved by head in the physical `[8192,4096]` checkpoint tensor; splitting the
first 4096 rows from the last 4096 rows is wrong. Q and K use zero-centered RMSNorm before partial
interleaved MRoPE. The causal-attention result is multiplied elementwise by `sigmoid(output_gate)`
before the output projection.

### 2.3 Gated DeltaNet

| Field | Value |
|---|---:|
| Q/K heads | 16 |
| Q/K head dimension | 128 |
| V heads | 32 |
| V head dimension | 128 |
| Q width | 2048 |
| K width | 2048 |
| V width | 4096 |
| Z output-gate width | 4096 |
| fused Q/K/V projection width | 8192 |
| A/B control width | 32 each |
| causal convolution width | 4 |
| recurrent-state dtype | FP32 |
| delta-rule scale | `1/sqrt(128)` |

Every two adjacent V heads share one Q/K head. Each GDN layer retains three preceding Q/K/V
projection columns and 32 recurrent matrices of shape `[128,128]`; its persistent state is bounded
with respect to context length.

### 2.4 Vision

| Field | Value |
|---|---:|
| transformer depth | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| head dimension | 72 |
| spatial patch | 16 x 16 |
| temporal patch | 2 frames |
| flattened patch width | `3 x 2 x 16 x 16 = 1536` |
| spatial merge | 2 x 2 patches |
| merger input width | 4608 |
| merger output width | 4096 |
| learned position table | 48 x 48 |
| Vision RoPE theta | 10000 |

### 2.5 Exact base-checkpoint parameter inventory

Header-only inspection of all safetensors shards gives BF16 tensors and the following exact
weight-element counts:

| Component | Parameters |
|---|---:|
| token embedding | 1016,670,720 |
| 32 Text decoder blocks | 12,304,609,280 |
| final Text norm and independent `lm_head` | 1016,670,720 |
| one-layer MTP module | 228,177,920 |
| Vision encoder and merger | 446,571,248 |
| **total checkpoint** | **15,012,699,888** |

## 3. Shared decoder layer skeleton and norms

Both Text layer types use the same pre-norm residual schedule:

```text
h = offset_rmsnorm(x, input_norm)
x = x + mixer(h)                         # GDN or gated full attention
h = offset_rmsnorm(x, post_attention_norm)
x = x + mlp(h)                           # dense SwiGLU
```

Text and MTP projections use bias-free linear layers. Attention dropout is zero.

## 4. Full-attention layer

For normalized input `h[4096,T]`:

```text
q_gate = q_projection(h)                  # [16,512,T]
q, gate = split_last(q_gate, [256,256])   # [16,256,T] each
k = k_projection(h)                       # [4,256,T]
v = v_projection(h)                       # [4,256,T]

q = offset_rmsnorm(q, q_norm)
k = offset_rmsnorm(k, k_norm)
q, k = partial_interleaved_mrope(q, k, rotary_dims=64)

a = causal_gqa(q, k, v, scale=1/sqrt(256), kv_cache)
a = a * sigmoid(gate)
y = o_projection(a)                       # [4096,T]
```

Each KV head serves four query heads. Prefill appends all K/V columns and evaluates causal
attention over the chunk. Decode appends one column and attends over the resident prefix.

## 5. Gated DeltaNet layer

The normalized input first produces logical Q/K/V, the output gate Z, and per-V-head controls:

```text
qkv = in_qkv(h)    # [8192,T] = q[16,128,T] | k[16,128,T] | v[32,128,T]
z   = in_z(h)      # [32,128,T]
a   = in_a(h)      # [32,T]
b   = in_b(h)      # [32,T]
```

Only concatenated Q/K/V passes through a depthwise causal width-4 convolution followed by SiLU.
Z, A, and B do not pass through the convolution.

## 6. Text prefill and decode

### Prefill

1. gather token embeddings into `[4096,T]` without an embedding scale;
2. replace image/video placeholder columns with Vision merger output when input is multimodal;
3. run the 32 decoder layers in `[GDN, GDN, GDN, full attention] x 8` order;
4. carry full-attention KV and GDN convolution/recurrent state across chunks without resetting
   logical positions;
5. apply final offset RMSNorm;
6. project required hidden columns through the independent `lm_head[248320,4096]`;
7. process logits and select the next token according to the generation policy.

### Ordinary decode

1. embed the current token;
2. run all 32 layers for one new position;
3. append four K/V entries and update twenty-four GDN state sets;
4. apply final norm and the full `lm_head`;
5. process logits, select the next token, and commit the new logical position.

## 7. MTP draft model

The checkpoint contains one MTP decoder layer. It is conditioned on both the target Text hidden
state and the token following that state.

For main-model final-normalized hidden state `h_t` and token `x_(t+1)`:

```text
e = offset_rmsnorm(embed(x_(t+1)), pre_fc_norm_embedding)
h = offset_rmsnorm(h_t,             pre_fc_norm_hidden)
u = fc(concat(e, h))                                      # [8192] -> [4096]

u = one_full_attention_dense_decoder_layer(u)
draft_hidden = offset_rmsnorm(u, mtp.norm)
draft_logits = shared_target_lm_head(draft_hidden)        # predicts x_(t+2)
```

The MTP decoder layer has its own full gated-attention and dense SwiGLU weights with the same
geometry as a Text full-attention layer: 16 Q heads, 4 KV heads, 256 head dimension, gated
attention output, and a 12288-wide SwiGLU MLP.

## 8. Evidence and implementation map

| Model concern | Source |
|---|---|
| exact 4096-wide dimensions, 32-layer topology, limits, scales | `src/targets/qwen3_5_9b/impl/config.h` |
| exact tensor binding, immutable Text/MTP/Vision views | `src/targets/qwen3_5_9b/impl/load/` |
| fused attention projection, staged GDN projection, dense post-mixer leaves | `src/targets/qwen3_5_9b/impl/variant.h`, `impl/variant.cpp` |
| fixed Text/MTP/Vision execution, planning, Program lifecycle | `src/targets/qwen3_6/impl/runtime/` |
| tokenizer, template, multimodal processing, output decoding | `src/targets/qwen3_6/impl/frontend/` |
| mathematical and explicit local-state Op contracts/implementations | `include/ninfer/ops/`, `src/ops/` |
| exact artifact and converter | [`qwen3.5-9b-artifact.md`](qwen3.5-9b-artifact.md), `tools/convert/qwen3_5_9b/` |