"""Base class for data format handlers."""

from abc import ABC, abstractmethod


class BaseFormat(ABC):
    """Abstract base for data format handlers.

    Subclasses implement :meth:`extensions`, :meth:`load`, and :meth:`save`,
    then register via ``@FormatRegistry.register``.

    Example::

        @FormatRegistry.register
        class MyFormat(BaseFormat):
            @classmethod
            def extensions(cls) -> list:
                return ['.myext']
            def load(self, file_path: str) -> list: ...
            def save(self, data: list, file_path: str, **kwargs) -> None: ...
    """

    @classmethod
    @abstractmethod
    def extensions(cls) -> list:
        """Return list of supported file extensions.

        Returns:
            list: List of file extensions (e.g., ['.json', '.JSON']).
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> list:
        """Load data from file.

        Args:
            file_path: Path to the data file.

        Returns:
            list: List of data dictionaries.
        """
        pass

    @abstractmethod
    def save(self, data: list, file_path: str, **kwargs) -> None:
        """Save data to file.

        Args:
            data: List of data dictionaries.
            file_path: Output file path.
            **kwargs: Format-specific options.
        """
        pass
