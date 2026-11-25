# GLM-5.3-Flash FP4-Local on TensorFold, four GB10s on a switchless ring

Serves NVIDIA's own mixed-precision checkpoint `nvidia/GLM-5.3-Flash-FP4-Local` (revision `76348ddb`, 165 GiB) with
TensorFold's native engine at tensor parallel 4, one rank on each of four DGX Spark class machines (GB10) cabled as
a ring with no switch.

Two ways to run it, kept side by side for comparison. Both produce the same container on every rank (checked by
creating rank 0 both ways and comparing image, command, environment, mounts, devices, limits and GPU request).

| | Compose | Shell scripts |
|---|---|---|
| Files | `compose.yaml` (one rank, declarative) + `Makefile` (order and waiting) | `start.sh`, `stop.sh`, `prepare.sh` |
| Prepare | `make prepare` | `./prepare.sh` |
| Start / stop | `make up` / `make down` | `./start.sh` / `./stop.sh` |
| See what will run | `make config` (rank 0's file, every value filled in) | `DRY_RUN=1 ./start.sh` |
| Drafter | always DFlash2 (`DRAFTER_DIR` required) | DFlash2, or the model's MTP drafts with `DRAFTER_DIR` empty |
| Extra environment | an optional `extra.env` file | `EXTRA_ENV` in `site.env` |
| Needs | Docker Compose 2.30 or later (`gpus:`) | bash, ssh |

Shared by both:

- `config.sh`: pinned inputs. TensorFold 1.3.4's release archive is fetched from GitHub and checked by sha256.
- `site.env` (git-ignored, from `site.env.example`): addresses, the model and drafter folders, port and context.

Run one way at a time: both name the container `glm53-fp4-tf` on every node.

The server is TensorFold's static binary. It runs in NVIDIA's PyTorch container only for the CUDA 13 user libraries,
NCCL 2.30.7 and rdma-core; nothing is built. Each rank mounts the release and the model read-only.

## What the ring needs

Each node reaches only its two neighbours, so (per TensorFold's GLM CUDA recipe): `TF_GLM_PREFILL_REDUCE=gather`,
`NCCL_ALGO=Ring` with a per-rank graph file, `NCCL_CUMEM_ENABLE=1`, `NCCL_RUNTIME_CONNECT=1`, and subnet-aware routing
for RoCE links on separate /24 subnets. `start.sh` sets these. The per-rank graph files (`nccl-graph-rank<N>.xml`)
are read from `GRAPH_DIR`; this recipe does not generate them yet.

`TF_ROCE_HCA` takes at most two device names, so the RoCE gather path is off here; gathers go through NCCL.

## Measured

2026-11-25, TensorFold 1.3.4, four GB10s, BF16 cache, 245,760-token context, 8 streams admitted, temperature 0, thinking off. Loaded in 101 s; 46.8 GiB of model a rank.

| tokens per second | MTP + copies (default) | DFlash2 + copies |
|---|---|---|
| one stream: structured / code / prose | 96.6 / 79.4 / 64.5 | 131.4 / 91.6 / 64.9 |
| four streams, aggregate: prose / code | 156.1 / 184.6 | 177.2 / 228.2 |
| cold prompt, 8K to 128K tokens | about 1,780 | about 1,480 |

### FP8 context cache and a 1M window (the default since 2026-11-25)

`EXTRA_ARGS="--kv-cache fp8"` and `CONTEXT=1048576` in `site.env`, matching the EXL3 lane's FP8 cache and window. The
engine then admits 7 streams at once, each with the whole 1,048,576-token window (6.96 GiB a stream a node), plus a
5.4 GiB prompt cache. Same benchmark, DFlash2:

| tokens per second | BF16 cache, 245,760 window | FP8 cache, 1,048,576 window |
|---|---|---|
| one stream: structured / code / prose | 131.4 / 91.6 / 64.9 | 131.6 / 92.6 / 66.2 |
| four streams, aggregate: prose / code | 177.2 / 228.2 | 145.2 / 152.1 |
| cold prompt, 8K to 224K tokens | about 1,480 | about 1,410 |

A 673,777-token prompt with a six-digit code at 60% depth came back with the code (476 s, thinking off). Upstream
marks the FP8 cache lossy (95.5% top-1 agreement with the BF16 cache in its two-node diagnostic). Four streams lose
about a third of their throughput with it; for more parallel throughput and a shorter window, set
`CONTEXT=245760` and `EXTRA_ARGS=""`.

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e`, `DRAFTER_DIR` in `site.env`, passed as `--drafter`) is the
default here: faster replies on structured text and code, the same on prose, and cold prompts about 17% slower
because the drafter reads the prompt too. Leave `DRAFTER_DIR` empty for the model's own MTP drafts.

A plain question came back without reasoning text when thinking was off, and a tool call returned well-formed
arguments. Not tried yet: `TF_GLM_DRAFT_CONFIDENCE`, the ring relay
(`TF_GLM_ROCE_MULTI` with `TF_TP_ROCE_RELAY`), `TF_GLM_CHUNK_SOLO=4096`, an API key, a longer context.
