"""Actionable failures; missing backends never silently become mock results."""


class ReconlabError(Exception):
    """Base class for recoverable user-facing errors."""


class ValidationError(ReconlabError):
    """An input, configuration, or artifact violates its contract."""


class DependencyError(ReconlabError):
    """An optional dependency is unavailable or has an incompatible version."""
