"""GPU power sampling via nvidia-smi for energy measurement."""

import shutil
import subprocess
import threading
import time

import numpy as np


class PowerSampler:
    """Background thread that polls GPU power draw via nvidia-smi.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds.
    gpu_index : int or None
        GPU index to monitor. If None, uses ``CUDA_VISIBLE_DEVICES`` or 0.

    Examples
    --------
    >>> with PowerSampler(interval=0.1) as sampler:
    ...     # run workload
    ...     pass
    >>> print(sampler.result())
    """

    def __init__(self, interval=0.1, gpu_index=None):
        self._interval = interval
        self._gpu_index = gpu_index
        self._samples = []
        self._thread = None
        self._stop_event = threading.Event()
        self._available = shutil.which("nvidia-smi") is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def start(self):
        """Start background power sampling."""
        if not self._available:
            return
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background power sampling."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def result(self):
        """Return power measurement summary.

        Returns
        -------
        dict
            ``power_mean_w``, ``power_std_w``, ``n_samples``.
            All values are 0.0 if nvidia-smi is unavailable.
        """
        if not self._samples:
            return {"power_mean_w": 0.0, "power_std_w": 0.0, "n_samples": 0}
        arr = np.array(self._samples)
        return {
            "power_mean_w": float(np.mean(arr)),
            "power_std_w": float(np.std(arr)),
            "n_samples": len(arr),
        }

    def _poll(self):
        """Sampling loop executed in background thread."""
        import os

        gpu_idx = self._gpu_index
        if gpu_idx is None:
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            gpu_idx = int(vis.split(",")[0])

        query = f"--id={gpu_idx}" if gpu_idx is not None else ""
        cmd = [
            "nvidia-smi",
            f"--query-gpu=power.draw",
            f"--format=csv,noheader,nounits",
        ]
        if query:
            cmd.insert(1, query)

        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=2, stderr=subprocess.DEVNULL)
                power = float(out.decode().strip().split("\n")[0])
                self._samples.append(power)
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop_event.wait(self._interval)


def compute_energy(power_result, time_seconds):
    """Compute energy from power measurement and elapsed time.

    Parameters
    ----------
    power_result : dict
        Output of :meth:`PowerSampler.result`.
    time_seconds : float
        Wall-clock time in seconds.

    Returns
    -------
    dict
        ``energy_mean_j`` and ``energy_std_j`` in joules.
    """
    return {
        "energy_mean_j": power_result["power_mean_w"] * time_seconds,
        "energy_std_j": power_result["power_std_w"] * time_seconds,
    }
