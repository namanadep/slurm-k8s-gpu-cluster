FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Match the slurm/munge UIDs from the giovtorres CentOS 7 controller image
# so that Slurm's munge-based UID checks pass across containers.
RUN groupadd -g 996 slurm && useradd -u 998 -g slurm -s /sbin/nologin slurm && \
    groupadd -g 997 munge && useradd -u 999 -g munge -s /sbin/nologin munge

# Install munge, Slurm compute daemon, Python
RUN apt-get update && apt-get install -y \
    munge \
    slurmd \
    slurm-client \
    python3 \
    python3-pip \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.1 with CUDA 12.1 wheels
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Slurm runtime directories
RUN mkdir -p /var/spool/slurmd /var/log/slurm /etc/slurm

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
