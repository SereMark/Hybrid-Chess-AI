import os

import psutil


def get_system_stats() -> dict[str, float]:
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0
        return {
            "cpu_percent": cpu_percent,
            "ram_used_gb": memory.used / (1024**3),
            "ram_total_gb": memory.total / (1024**3),
            "load_avg": load_avg,
        }
    except Exception:
        return {"cpu_percent": 0, "ram_used_gb": 0, "ram_total_gb": 0, "load_avg": 0}
