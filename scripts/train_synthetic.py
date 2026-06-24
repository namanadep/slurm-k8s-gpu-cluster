#!/usr/bin/env python3
"""
ResNet-18 DDP demo — synthetic CIFAR-10-shaped data (3×32×32, 10 classes).

Identical DDP machinery as train_cifar10.py; synthetic tensors avoid the
dataset download so the benchmark runs immediately. GPU throughput and
distributed-reduce logic are real.

  torchrun --nnodes=1 --nproc_per_node=1 train_synthetic.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

dist.init_process_group(backend="nccl")

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK       = dist.get_rank()
WORLD_SIZE = dist.get_world_size()

torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda", LOCAL_RANK)

def log(msg):
    if RANK == 0:
        print(msg, flush=True)

log(f"[init] PyTorch {torch.__version__}  |  WORLD_SIZE={WORLD_SIZE}")
log(f"[init] GPU: {torch.cuda.get_device_name(LOCAL_RANK)}")
log(f"[init] VRAM: {torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1024**3:.1f} GB")
log(f"[init] Processes: {WORLD_SIZE}  (scale to more nodes to add ranks)")

EPOCHS   = 5
BATCH    = 256
LR       = 0.05 * WORLD_SIZE
N_TRAIN  = 50000
N_VAL    = 10000

log("[data] Generating synthetic CIFAR-10-shaped tensors (no download needed) …")
torch.manual_seed(42)
train_ds = TensorDataset(
    torch.randn(N_TRAIN, 3, 32, 32),
    torch.randint(0, 10, (N_TRAIN,)),
)
val_ds = TensorDataset(
    torch.randn(N_VAL, 3, 32, 32),
    torch.randint(0, 10, (N_VAL,)),
)

train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
val_sampler   = DistributedSampler(val_ds,   num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)

train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler,
                      num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, sampler=val_sampler,
                      num_workers=4, pin_memory=True)

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model = model.to(device)
model = DDP(model, device_ids=[LOCAL_RANK])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

log(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
    f"{'Val Loss':>9}  {'Val Acc':>8}  {'Time':>6}  {'Throughput':>12}")
log("─" * 80)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_sampler.set_epoch(epoch)

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

    stats = torch.tensor([train_loss, float(correct), float(total)], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    train_loss = stats[0].item() / stats[2].item()
    train_acc  = stats[1].item() / stats[2].item() * 100

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
    throughput = len(train_ds) / elapsed

    log(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
        f"{val_loss:>9.4f}  {val_acc:>7.2f}%  {elapsed:>5.1f}s  "
        f"{throughput:>9.0f} img/s")

log("\n[done] Training complete.")
log(f"[done] Effective batch size: {BATCH * WORLD_SIZE}  "
    f"(per-GPU: {BATCH} × {WORLD_SIZE} process{'es' if WORLD_SIZE > 1 else ''})")
log(f"[done] Final val acc: {val_acc:.2f}%  (random labels → ~10% expected)")
log("[done] To use real CIFAR-10: sbatch jobs/train_cifar10.sh")
log("[done] To scale: add --nnodes=N to the torchrun command")

dist.destroy_process_group()
