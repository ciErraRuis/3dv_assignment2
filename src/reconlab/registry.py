"""Typed registry and optional module plugins (loaded only on explicit request)."""

from __future__ import annotations

import importlib
from typing import TypeVar

from .errors import ValidationError

T = TypeVar("T")


class ComponentRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[type, str], type] = {}

    def register(self, interface: type[T], name: str, implementation: type[T]) -> None:
        if not issubclass(implementation, interface):
            raise ValidationError(f"{implementation} does not implement {interface}")
        key = (interface, name)
        if key in self._items:
            raise ValidationError(f"Component already registered: {interface.__name__}:{name}")
        self._items[key] = implementation

    def create(self, interface: type[T], name: str, **options) -> T:
        try:
            cls = self._items[(interface, name)]
        except KeyError as exc:
            raise ValidationError(f"Unknown {interface.__name__}: {name}") from exc
        try:
            return cls(**options)
        except TypeError as exc:
            raise ValidationError(f"Invalid options for {name}: {exc}") from exc

    def load_plugin(self, module: str) -> None:
        """The module must expose register(registry); it is trusted Python code."""
        importlib.import_module(module).register(self)
