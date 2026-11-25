# Pinned inputs. TensorFold's native release (Apache-2.0), checked by sha256 before it is unpacked.
TF_VERSION=1.3.4
TF_ARCHIVE=tensorfold-$TF_VERSION-linux-aarch64.tar.gz
TF_SHA256=02566a5297dd67db9b0b834737fe73e192907454896ee7d097d3a0dfca303937
TF_URL=https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION/$TF_ARCHIVE
# The container only supplies CUDA 13 user libraries, NCCL 2.30.7 and rdma-core; the server is the static binary above.
BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
CONTAINER=glm53-fp4-tf
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TF_DIR=$HERE/dist/tensorfold-$TF_VERSION-linux-aarch64
source "$HERE/site.env"
