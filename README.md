# GLM-5.3-Flash FP4-Local on TensorFold, four GB10s on a switchless ring

Serves NVIDIA's own mixed-precision checkpoint `nvidia/GLM-5.3-Flash-FP4-Local` (revision `76348ddb`, 165 GiB) with
TensorFold's native engine at tensor parallel 4, one rank on each of four DGX Spark class machines (GB10) cabled as
a ring with no switch.

- `config.sh`: pinned inputs. TensorFold 1.3.4's release archive is fetched from GitHub and checked by sha256.
- `site.env` (git-ignored, from `site.env.example`): addresses, the model folder, port and context.
- `./prepare.sh`: fetch and verify the release here, then copy this folder to every worker.
- `./start.sh`: start the workers' ranks, then rank 0 here, and wait for the API (`DRY_RUN=1` prints the commands).
- `./stop.sh`: remove the container on every node (exact name).

The server is TensorFold's static binary. It runs in NVIDIA's PyTorch container only for the CUDA 13 user libraries,
NCCL 2.30.7 and rdma-core; nothing is built. Each rank mounts the release and the model read-only.

## What the ring needs

Each node reaches only its two neighbours, so (per TensorFold's GLM CUDA recipe): `TF_GLM_PREFILL_REDUCE=gather`,
`NCCL_ALGO=Ring` with a per-rank graph file, `NCCL_CUMEM_ENABLE=1`, `NCCL_RUNTIME_CONNECT=1`, and subnet-aware routing
for RoCE links on separate /24 subnets. `start.sh` sets these. The per-rank graph files (`nccl-graph-rank<N>.xml`)
are read from `GRAPH_DIR`; this recipe does not generate them yet.

`TF_ROCE_HCA` takes at most two device names, so the RoCE gather path is off here; gathers go through NCCL.

## Measured

2026-11-25, TensorFold 1.3.4, four GB10s, 245,760-token context, 8 streams admitted, drafts MTP + copies (defaults),
temperature 0, thinking off. Loaded in 101 s; 46.8 GiB of model a rank.

| | tokens per second |
|---|---|
| one stream: structured / code / prose | 96.6 / 79.4 / 64.5 |
| four streams, aggregate: prose / code | 156.1 / 184.6 |
| cold prompt, 8K to 128K tokens | about 1,780 |

A plain question came back without reasoning text when thinking was off, and a tool call returned well-formed
arguments. Not tried yet: the DFlash2 drafter (`--drafter`), `TF_GLM_DRAFT_CONFIDENCE`, the ring relay
(`TF_GLM_ROCE_MULTI` with `TF_TP_ROCE_RELAY`), `TF_GLM_CHUNK_SOLO=4096`, an API key, a longer context.
