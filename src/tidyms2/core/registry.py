"""Registry for concrete classes."""

from typing import Generic, TypeVar

from .exceptions import RegistryError, RepeatedIdError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maintains a registry of related classes."""

    def __init__(self, name: str):
        self._name = name
        self._records: dict[str, type[T]] = dict()

    def get(self, id_: str) -> type[T]:
        """Retrieve a class from the registry."""
        if id_ not in self._records:
            raise RegistryError(f"Entry {id_} not found in {self._name} registry.")
        return self._records[id_]

    def register(self, *ids: str):
        """Add a class to the registry.

        Use as a decorator.

        """
        for id_ in ids:
            if id_ in self._records:
                raise RepeatedIdError(id_)

        def wrapper(entry: type[T]) -> type[T]:
            if not ids:
                self._records[entry.__name__] = entry
            else:
                for id_ in ids:
                    self._records[id_] = entry
            return entry

        return wrapper


operator_registry = Registry("operator")
