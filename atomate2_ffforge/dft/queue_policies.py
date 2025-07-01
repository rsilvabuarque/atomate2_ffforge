"""
Queue-routing helpers (local vs SLURM) for FFForge.

• Detect a compatible FireWorks queue adapter (FW 1.x → 2.1+)
• Fall back to a stub when FireWorks is missing (CI / unit-test mode)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
import importlib
import warnings

import yaml  # type: ignore

# -----------------------------------------------------------------------------
# 0  Typing helpers – visible to MyPy only
# -----------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from typing import TypeAlias

    FWQueueAdapter: TypeAlias = Any  # concrete at runtime, opaque to MyPy

# -----------------------------------------------------------------------------
# 1  Locate a FireWorks adapter class (or supply a stub)
# -----------------------------------------------------------------------------
_SEARCH: list[tuple[str, list[str]]] = [
    ("fireworks.queue.adapters.common_adapter", ["CommonAdapter"]),  # FW ≥ 2.1
    (
        "fireworks.queue.queue_adapter",
        ["CommonAdapter", "QueueAdapter", "QueueAdapterBase"],
    ),  # FW 2.0
    ("fireworks.utilities.queue_adapter", ["QueueAdapter"]),  # FW ≤ 1.x
]

_adapter_cls: type[Any] | None = None
for mod, names in _SEARCH:
    try:
        module = importlib.import_module(mod)
    except ImportError:
        continue
    for name in names:
        if hasattr(module, name):
            _adapter_cls = getattr(module, name)
            break
    if _adapter_cls:
        break

if _adapter_cls is None:  # FireWorks absent → stub
    warnings.warn(
        "FireWorks not detected; using dummy QueueAdapter stub (unit-test mode)."
    )

    class _StubQueueAdapter:  # noqa: D101
        def __init__(self, *_: Any, **__: Any) -> None: ...
        @staticmethod
        def from_dict(_: dict[str, Any]) -> "_StubQueueAdapter":
            return _StubQueueAdapter()

        def __repr__(self) -> str:  # noqa: D401
            return "QueueAdapter<stub>"

    _adapter_cls = _StubQueueAdapter  # type: ignore[assignment]

QueueAdapter: type[Any] = _adapter_cls  # runtime public alias

# -----------------------------------------------------------------------------
# 2  Helpers
# -----------------------------------------------------------------------------
_LAUNCH_DIR = Path(__file__).resolve().parent.parent / "launch"
_LOCAL_LABELS = {"DIRECT", "COMPARE", "METRICS"}


def _build_adapter(cfg: dict[str, Any]) -> "FWQueueAdapter | Any":  # noqa: D401
    """Instantiate *QueueAdapter* from YAML *cfg*."""
    if hasattr(QueueAdapter, "from_dict"):  # type: ignore[attr-defined]
        try:
            return QueueAdapter.from_dict(cfg)  # type: ignore[attr-defined]
        except TypeError:
            pass

    try:
        return QueueAdapter(**cfg)  # type: ignore[arg-type,operator]
    except TypeError:
        return QueueAdapter(cfg)  # type: ignore[call-arg]


@lru_cache(maxsize=None)
def _load_adapter(kind: Literal["local", "slurm"]) -> "FWQueueAdapter | Any":
    yaml_cfg = yaml.safe_load((_LAUNCH_DIR / f"{kind}.yaml").read_text())
    return _build_adapter(yaml_cfg["queue_adapter"])


def queue_adapter(
    label: str, override: Literal["local", "slurm", None] = None
) -> "FWQueueAdapter | Any":
    """
    Return a QueueAdapter according to *label* or explicit *override*.
    """
    if override in {"local", "slurm"}:
        route: Literal["local", "slurm"] = override
    else:
        route = "local" if label in _LOCAL_LABELS else "slurm"
    return _load_adapter(route)


__all__ = ["queue_adapter", "QueueAdapter"]
