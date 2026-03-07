"""Component registry and config-based instantiation."""

from typing import Any, Dict, Optional, Type, Union


class Registry:
    """A registry for managing and instantiating registered modules.

    Example:
        FILTERS = Registry()

        @FILTERS.register_module()
        class MyFilter:
            pass

        # Get registered class
        cls = FILTERS.get('MyFilter')
    """

    def __init__(self, name: str = ""):
        """Initialize the registry.

        Args:
            name: Optional name for the registry (for debugging).
        """
        self._name = name
        self._modules: Dict[str, Type] = {}

    def register_module(
        self, name: Optional[str] = None, module: Optional[Type] = None
    ):
        """Register a module class.

        Can be used as a decorator or called directly:
            @REGISTRY.register_module()
            class MyClass: pass

            # Or directly:
            REGISTRY.register_module('MyClass', MyClass)

        Args:
            name: Optional name for the module. If None, uses class __name__.
            module: Optional module class to register directly.

        Returns:
            Decorator function or the registered module.
        """

        def _register_module(module: Type) -> Type:
            key = name if name is not None else module.__name__
            if key in self._modules:
                raise KeyError(
                    f"{key} is already registered in {self._name or 'registry'}"
                )
            self._modules[key] = module
            return module

        if module is not None:
            return _register_module(module)
        return _register_module

    def get(self, name: str) -> Type:
        """Get a registered module by name.

        Args:
            name: Name of the registered module.

        Returns:
            The registered module class.

        Raises:
            KeyError: If the module is not registered.
        """
        if name not in self._modules:
            raise KeyError(f"{name} is not registered in {self._name or 'registry'}")
        return self._modules[name]

    def __contains__(self, name: str) -> bool:
        """Check if a module is registered."""
        return name in self._modules

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, modules={list(self._modules.keys())})"


def build_from_cfg(cfg: Union[str, Dict, None], registry: Registry, **kwargs) -> Any:
    """Build a module from configuration.

    Args:
        cfg: Configuration dict with 'type' key, or string name, or None.
        registry: Registry containing the module class.
        **kwargs: Additional arguments passed to the module constructor.

    Returns:
        Instantiated module, or None if cfg is None.

    Raises:
        TypeError: If cfg type is not supported or instantiation fails.
    """
    if cfg is None:
        return None

    if isinstance(cfg, str):
        return registry.get(cfg)(**kwargs)

    if isinstance(cfg, dict):
        args = cfg.copy()
        args.update(kwargs)
        obj_type = args.pop("type", None)
        if obj_type is None:
            raise TypeError("Config dict must contain 'type' key")
        obj_cls = registry.get(obj_type)
        try:
            return obj_cls(**args)
        except TypeError as e:
            raise TypeError(f"{obj_cls.__name__}: {e}")


# Global registries for different component types
DATALOADER = Registry("dataloader")
PIPELINES = Registry("pipelines")
OPERATORS = Registry("operators")  # Unified registry for all operators
MODELS = Registry("models")
DATASAVER = Registry("datasaver")
HOOKS = Registry("hooks")
REQUESTS = Registry("requests")
