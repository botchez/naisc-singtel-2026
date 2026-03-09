import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("naisc")


def safe_n_splits(n_samples: int, desired: int = 5, min_per_fold: int = 20) -> int:
    """Compute a safe number of CV splits given sample size."""
    return max(2, min(desired, n_samples // min_per_fold))


class Timer:
    """Simple context-manager timer that prints elapsed seconds."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
