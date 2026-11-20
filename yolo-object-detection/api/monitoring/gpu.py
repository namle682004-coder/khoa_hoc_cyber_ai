import pynvml

try:
    pynvml.nvmlInit()
    HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    HANDLE = None


def gpu_stats():
    if not HANDLE:
        return {
            "gpu": None,
            "memory": None,
            "available": False,
        }

    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(HANDLE)
        return {
            "gpu": util.gpu,
            "memory": util.memory,
            "available": True,
        }
    except Exception:
        return {
            "gpu": None,
            "memory": None,
            "available": False,
        }
