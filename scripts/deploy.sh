#!/usr/bin/env bash
# Deploy the Slurm HPC cluster to Kubernetes.
# Run this once after `minikube start --driver=docker --gpus=all`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="$REPO_DIR/k8s"
MINIKUBE_IP=$(minikube ip)

echo "==> Kubernetes cluster: $(kubectl cluster-info | head -1)"
echo "==> minikube IP: $MINIKUBE_IP"
echo ""

# 1. Generate munge key and patch the Secret manifest
echo "[1/5] Generating munge key …"
MUNGE_B64=$(dd if=/dev/urandom bs=1 count=1024 2>/dev/null | base64 -w 0)
sed "s|PLACEHOLDER_REPLACE_WITH_BASE64_KEY|${MUNGE_B64}|" \
    "$K8S_DIR/02-munge-secret.yaml" | kubectl apply -f -

# 2. Apply remaining manifests
echo "[2/5] Applying namespace and ConfigMap …"
kubectl apply -f "$K8S_DIR/01-namespace.yaml"
kubectl apply -f "$K8S_DIR/03-slurm-configmap.yaml"

echo "[3/5] Deploying Slurm controller pod …"
kubectl apply -f "$K8S_DIR/04-slurm-controller.yaml"

echo "[4/5] Waiting for controller to be ready …"
kubectl rollout status deployment/slurm-controller -n hpc --timeout=120s

echo "[5/5] Starting GPU compute container on minikube network …"
# Pull the munge key so the compute container uses the same one
MUNGE_KEY_FILE="$REPO_DIR/.munge.key.tmp"
echo "$MUNGE_B64" | base64 -d > "$MUNGE_KEY_FILE"
chmod 400 "$MUNGE_KEY_FILE"

docker run -d \
  --name gpu-node-0 \
  --hostname gpu-node-0 \
  --network minikube \
  --ip 192.168.49.10 \
  --gpus all \
  --add-host "slurmctl:${MINIKUBE_IP}" \
  -v "${MUNGE_KEY_FILE}:/etc/munge/munge.key:ro" \
  -v "${REPO_DIR}/k8s/slurm.conf.tmp:/etc/slurm/slurm.conf:ro" \
  -v "${REPO_DIR}/scripts:/scripts:ro" \
  -v "${REPO_DIR}/jobs:/jobs" \
  -e SLURM_NODENAME=gpu-node-0 \
  slurm-gpu-compute:latest

rm -f "$MUNGE_KEY_FILE"

echo ""
echo "==> Cluster deployed. Verify with:"
echo "    kubectl get pods -n hpc"
echo "    kubectl exec -n hpc deploy/slurm-controller -- sinfo"
echo "    kubectl exec -n hpc deploy/slurm-controller -- sbatch /jobs/train_cifar10.sh"
