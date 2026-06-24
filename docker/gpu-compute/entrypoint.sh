#!/bin/bash
set -e

# Fix ownership for munge
chown munge:munge /var/log/munge /var/lib/munge /etc/munge
chmod 700 /var/log/munge /var/lib/munge
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

# Create runtime directories
mkdir -p /run/munge /var/run/slurm /var/spool/slurmd /var/log/slurm
chown munge:munge /run/munge

# Start munged as the munge user and wait for socket
gosu munge /usr/sbin/munged --foreground &
for i in $(seq 1 15); do
    [ -S /run/munge/munge.socket.2 ] && break
    sleep 1
done
if [ ! -S /run/munge/munge.socket.2 ]; then
    echo "[entrypoint] ERROR: munged socket not found after 15s" >&2
    exit 1
fi
echo "[entrypoint] munged ready (socket: /run/munge/munge.socket.2)"

# Confirm config
grep -E "SlurmctldHost|ClusterName" /etc/slurm/slurm.conf 2>/dev/null || true

echo "[entrypoint] Starting slurmd as node: $SLURM_NODENAME"
exec /usr/sbin/slurmd -D -N "$SLURM_NODENAME"
