"""Common utilities."""

from uuid import UUID

from uuid_utils.compat import uuid7


def create_id() -> UUID:
    """Create an unique if for a model."""
    return uuid7()
