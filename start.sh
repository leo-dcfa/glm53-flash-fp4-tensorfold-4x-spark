#!/usr/bin/env bash
# Start GLM-5.3-Flash FP4-Local on TensorFold's native engine, one rank a node: workers first, then rank 0 here.
set -euo pipefail
source "$(dirname "$0")/config.sh"
WORLD=$(( $(wc -w <<<"$WORKERS") + 1 ))
run_cmd() {  # run_cmd <rank>: the docker command for that rank, one line
  local r=$1
  local drafter_mount="" drafter_arg=""
  [ -z "${DRAFTER_DIR:-}" ] || { drafter_mount="-v $DRAFTER_DIR:/drafter:ro"; drafter_arg="--drafter /drafter"; }
  echo docker run -d --name $CONTAINER --gpus all --ipc=host --network host --shm-size 16g --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$TF_DIR:/opt/tensorfold:ro" -v "$MODEL_DIR:/model:ro" -v "$GRAPH_DIR/nccl-graph-rank$r.xml:/nccl-graph.xml:ro" $drafter_mount \
    -e TF_GLM_RANK=$r -e TF_GLM_WORLD=$WORLD -e NCCL_COMM_ID=$HEAD_ADDR:29583 -e TF_GLM_PREFILL_REDUCE=gather \
    -e NCCL_CUMEM_ENABLE=1 -e NCCL_RUNTIME_CONNECT=1 -e NCCL_ALGO=Ring -e NCCL_GRAPH_FILE=/nccl-graph.xml \
    -e NCCL_SOCKET_IFNAME=$SOCKET_IFNAME -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1 \
    -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
    -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e HF_HUB_OFFLINE=1 $EXTRA_ENV \
    --entrypoint /opt/tensorfold/bin/tensorfold-native $BASE_IMAGE serve /model --name $SERVED_NAME \
    --host $HOST --port $PORT --context $CONTEXT $drafter_arg $EXTRA_ARGS
}
if [ "${DRY_RUN:-0}" = 1 ]; then for r in $(seq 0 $((WORLD-1))); do echo "rank $r: $(run_cmd $r)"; done; exit 0; fi
r=$WORLD; for w in $(tac -s' ' <<<"$WORKERS "); do r=$((r-1)); ssh -n -o BatchMode=yes "$w" "docker rm -f $CONTAINER >/dev/null 2>&1; $(run_cmd $r)" >/dev/null && echo "rank $r started on $w"; done
docker rm -f $CONTAINER >/dev/null 2>&1 || true; $(run_cmd 0) >/dev/null && echo "rank 0 started here"
echo "waiting for http://$HOST:$PORT/v1/models (docker logs -f $CONTAINER)"
for i in $(seq 1 120); do
  curl -fsS -m 3 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1 && { echo "serving after ~$((i*10)) s"; exit 0; }
  docker ps --format '{{.Names}}' | grep -qx $CONTAINER || { echo "rank 0 exited:"; docker logs --tail 30 $CONTAINER 2>&1; exit 1; }
  sleep 10
done
echo "not serving after 20 min"; exit 1
