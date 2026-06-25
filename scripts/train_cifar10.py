#!/usr/bin/env python3
"""
ResNet-18 on CIFAR-10 - PyTorch Distributed Data Parallel (DDP) training.

Designed to scale across N nodes × G GPUs each.
Launch with torchrun (set by the Slurm job script):

  # 1 node, 1 GPU  (this demo)
  torchrun --nnodes=1 --nproc_per_node=1 train_cifar10.py

  # 4 nodes, 2 GPUs each  (production scale-out)
  srun torchrun --nnodes=4 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    train_cifar10.py

Environment variables set by torchrun:
  LOCAL_RANK  - GPU index on this node
  RANK        - global process rank across all nodes
  WORLD_SIZE  - total number of processes (nodes × GPUs-per-node)
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as T

# ── Distributed init ──────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")

LOCAL_RANK  = int(os.environ["LOCAL_RANK"])
RANK        = dist.get_rank()
WORLD_SIZE  = dist.get_world_size()

torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK)

def log(msg):
    """Print only from global rank 0."""
    if RANK == 0:
        print(msg, flush=True)

log(f"[init] PyTorch {torch.__version__}  |  WORLD_SIZE={WORLD_SIZE}")
log(f"[init] GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")
log(f"[init] VRAM: {torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1024**3:.1f} GB")
log(f"[init] Processes: {WORLD_SIZE}  (scale to more nodes to add ranks)")

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS   = 5
BATCH    = 256          # per-process; effective batch = BATCH × WORLD_SIZE
LR       = 0.05 * WORLD_SIZE   # linear LR scaling rule for DDP
WORKERS  = 4
DATA_DIR = "/tmp/cifar10"

# ── Data - DistributedSampler ensures each rank sees different shards ─────────
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

log("[data] Downloading / loading CIFAR-10 …")
train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=(RANK == 0), transform=train_tf)
val_ds   = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=(RANK == 0), transform=val_tf)

dist.barrier()   # wait until rank-0 download is done before other ranks read

train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
val_sampler   = DistributedSampler(val_ds,   num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)

train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler,
                      num_workers=WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, sampler=val_sampler,
                      num_workers=WORKERS, pin_memory=True)

# ── Model - wrapped in DDP after moving to the local GPU ──────────────────────
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model = model.to(device)
model = DDP(model, device_ids=[LOCAL_RANK])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ─────────────────────────────────────────────────────────────
log(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
    f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}  {'Throughput':>12}")
log("─" * 80)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_sampler.set_epoch(epoch)   # reshuffle each epoch for DDP

    # ---- train ----
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)

    # Aggregate metrics across all ranks
    stats = torch.tensor([train_loss, float(correct), float(total)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    train_loss = stats[0].item() / stats[2].item()
    train_acc  = stats[1].item() / stats[2].item() * 100

    # ---- validate ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            correct  += (out.argmax(1) == y).sum().item()
            total    += x.size(0)

    stats = torch.tensor([val_loss, float(correct), float(total)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    val_loss = stats[0].item() / stats[2].item()
    val_acc  = stats[1].item() / stats[2].item() * 100

    scheduler.step()
    elapsed = time.time() - t0
    # Effective throughput = samples processed per second across all ranks
    throughput = (len(train_ds) / elapsed)

    log(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
        f"{val_loss:>9.4f}  {val_acc:>7.2f}%  {elapsed:>5.1f}s  "
        f"{throughput:>9.0f} img/s")

log("\n[done] Training complete.")
log(f"[done] Effective batch size: {BATCH * WORLD_SIZE}  "
    f"(per-GPU: {BATCH} × {WORLD_SIZE} process{'es' if WORLD_SIZE > 1 else ''})")
log(f"[done] Final validation accuracy: {val_acc:.2f}%")
log(f"[done] To scale: add --nnodes=N to the torchrun command in train_cifar10.sh")

dist.destroy_process_group()
