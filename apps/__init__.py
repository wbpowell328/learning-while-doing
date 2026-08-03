"""
Application registry.

Add new applications by:
  1. Create apps/<your_app>/ with its own config, dynamics, simulate().
  2. Import it below and register it in REGISTRY under a short key.

Every app must satisfy the Simulator protocol in apps/base.py.
"""
from .base import Simulator

# App implementations.
from . import cash_balance
from . import cash_balance_2d

REGISTRY: dict[str, object] = {
    "cash_balance":    cash_balance,
    "cash_balance_2d": cash_balance_2d,
}


def get_app(name: str):
    """Look up an application module by its registry key."""
    if name not in REGISTRY:
        raise KeyError(f"unknown app {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["Simulator", "REGISTRY", "get_app"]
