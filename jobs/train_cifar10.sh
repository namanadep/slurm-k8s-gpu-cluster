#!/bin/bash
# ============================================================
# Slurm job: ResNet-18 CIFAR-10 — PyTorch DDP
#
# Single-GPU demo (this cluster):
#   sbatch train_cifar10.sh
#
# Scale to 4 nodes × 2 GPUs: change the two lines below to
#   #SBATCH --nodes=4
#   #SBATCH --gres=gpu:2
# ============================================================
#SBATCH --job-name=resnet18-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        # one task per node (torchrun owns GPU fanout)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:rtx3070:1
#SBATCH --time=00:30:00
#SBATCH --output=/jobs/slurm-%j.out
#SBATCH --error=/jobs/slurm-%j.err

# ── Environment ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job : $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "  Node: $SLURMD_NODENAME"
echo "  CPUs: $SLURM_CPUS_PER_TASK  |  Tasks: $SLURM_NTASKS"
echo "  Nodes in job: $SLURM_JOB_NODELIST"
echo "  Started: $(date)"
echo "============================================================"

# ── Distributed rendezvous setup ─────────────────────────────────────────────
# MASTER_ADDR = hostname of the first allocated node (rank 0 rendezvous point)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo ""
echo "[ddp] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "[ddp] WORLD_SIZE = $SLURM_NNODES nodes × $SLURM_NTASKS_PER_NODE GPU(s)"
echo ""

# ── nvidia-smi snapshot ───────────────────────────────────────────────────────
echo "[gpu] GPU allocated to this job:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
           --format=csv,noheader,nounits
echo ""

# ── Launch with torchrun ──────────────────────────────────────────────────────
# srun distributes one task to each node; torchrun handles per-node GPU fanout.
#
# To scale:
#   --nnodes=$SLURM_NNODES             picks up however many nodes Slurm gave us
#   --nproc_per_node=$SLURM_GPUS_ON_NODE  one process per GPU on each node
#   --rdzv_backend=c10d                TCP-based rendezvous (no external store needed)
#
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    /scripts/train_cifar10.py

echo ""
echo "============================================================"
echo "  Finished: $(date)"
echo "============================================================"
