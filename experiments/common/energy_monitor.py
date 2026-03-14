"""Energy monitoring via Cray pm_counters sysfs interface.

Reads full-node energy counters (CPU + GPU + NIC + memory + everything)
from /sys/cray/pm_counters/. Works on CSCS Alps GH200 nodes.

Usage:
    monitor = EnergyMonitor()
    monitor.mark_start()
    # ... do work ...
    sample = monitor.mark_end()
    print(f"Node energy: {sample['node_joules']:.1f} J")
"""

import os
import time

PM_COUNTERS_DIR = "/sys/cray/pm_counters"


def _read_counter(path):
    """Read a pm_counters file, return (value, timestamp_us)."""
    try:
        with open(path, "r") as f:
            parts = f.readline().split()
            return int(parts[0]), int(parts[2])
    except (OSError, IndexError, ValueError):
        return None, None


def _detect_accel_indices():
    """Find which accel indices exist (compute nodes have 1, login nodes up to 4)."""
    indices = []
    for i in range(8):
        if os.path.exists(os.path.join(PM_COUNTERS_DIR, f"accel{i}_energy")):
            indices.append(i)
    return indices


class EnergyMonitor:
    """Per-iteration energy monitor using Cray pm_counters.

    Captures full-node energy (includes CPU, GPU, NIC, memory, everything)
    by reading hardware counters at the power supply level.

    Parameters
    ----------
    rank : int
        MPI rank (for logging). Default 0.
    """

    def __init__(self, rank=0):
        self.rank = rank
        self.available = os.path.isdir(PM_COUNTERS_DIR)
        self.accel_indices = _detect_accel_indices() if self.available else []
        self._start = None
        self.samples = []

    def _snapshot(self):
        """Read all energy counters at once."""
        if not self.available:
            return None
        snap = {"wall_time": time.time()}
        snap["node_energy"], snap["node_ts"] = _read_counter(
            os.path.join(PM_COUNTERS_DIR, "energy")
        )
        snap["cpu_energy"], _ = _read_counter(
            os.path.join(PM_COUNTERS_DIR, "cpu_energy")
        )
        for i in self.accel_indices:
            snap[f"accel{i}_energy"], _ = _read_counter(
                os.path.join(PM_COUNTERS_DIR, f"accel{i}_energy")
            )
        return snap

    def mark_start(self):
        """Record energy counters at the start of an interval."""
        self._start = self._snapshot()

    def mark_end(self, label=""):
        """Record energy counters at the end, compute deltas, store sample."""
        if self._start is None or not self.available:
            return None
        end = self._snapshot()
        if end is None:
            return None
        # Skip if any counter is None (sysfs read failed)
        if (
            self._start.get("node_energy") is None
            or end.get("node_energy") is None
            or self._start.get("cpu_energy") is None
            or end.get("cpu_energy") is None
        ):
            self._start = end
            return None
        sample = {
            "label": label,
            "wall_seconds": end["wall_time"] - self._start["wall_time"],
            "node_joules": end["node_energy"] - self._start["node_energy"],
            "cpu_joules": end["cpu_energy"] - self._start["cpu_energy"],
        }
        gpu_total = 0
        for i in self.accel_indices:
            key = f"accel{i}_energy"
            s_val, e_val = self._start.get(key), end.get(key)
            if s_val is None or e_val is None:
                continue
            delta = e_val - s_val
            sample[f"accel{i}_joules"] = delta
            gpu_total += delta
        sample["gpu_joules"] = gpu_total
        sample["other_joules"] = (
            sample["node_joules"] - sample["cpu_joules"] - gpu_total
        )
        sample["node_watts_avg"] = (
            sample["node_joules"] / sample["wall_seconds"]
            if sample["wall_seconds"] > 0
            else 0
        )
        self.samples.append(sample)
        self._start = end  # chain for next interval
        return sample

    def summary(self):
        """Return summary statistics over all recorded samples."""
        if not self.samples:
            return {}
        import numpy as np

        node_j = [s["node_joules"] for s in self.samples]
        gpu_j = [s["gpu_joules"] for s in self.samples]
        cpu_j = [s["cpu_joules"] for s in self.samples]
        watts = [s["node_watts_avg"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "node_joules_mean": float(np.mean(node_j)),
            "node_joules_std": float(np.std(node_j)),
            "node_joules_total": float(np.sum(node_j)),
            "gpu_joules_mean": float(np.mean(gpu_j)),
            "cpu_joules_mean": float(np.mean(cpu_j)),
            "node_watts_mean": float(np.mean(watts)),
        }
