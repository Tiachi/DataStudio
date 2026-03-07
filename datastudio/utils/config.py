"""Configuration loading and management utilities.

Supports hierarchical configuration with inheritance (``_base_`` key),
recursive dict merging, and YAML export.

Example:
    >>> init_cfg('configs/my_config.py')
    >>> cfg = get_cfg()
    >>> print(cfg.model.name)
"""

import copy
import inspect
import logging
import os
import sys
from collections import OrderedDict
from importlib import import_module

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

__all__ = ["get_cfg", "init_cfg", "save_cfg", "print_cfg"]
BASE_KEY = "_base_"
COVER_KEY = "_cover_"
IGNORE_KEY = "_ignore_"
# Special prefix for paths relative to the configs root directory
CONFIGS_ROOT_PREFIX = "@/"


class Config(OrderedDict):
    """Configuration class based on OrderedDict.

    Supports attribute-style access, hierarchical configuration inheritance,
    and automatic merging of base configurations.

    Attributes:
        name: Configuration name (derived from filename).
        work_dir: Working directory for outputs.

    Example:
        >>> cfg = Config('config.py')
        >>> logger.info(cfg.model_name)
    """

    def __init__(self, *args):
        """Initialize the Config.

        Args:
            *args: Optional single argument - path to config file.
        """
        super().__init__()
        if len(args) == 1:
            self.load_from_file(args[0])
        else:
            assert len(args) == 0

    def __getattr__(self, name):
        """Get attribute by name, returning None if not found."""
        if name in self:
            return self[name]
        return None

    def __setattr__(self, name, value):
        """Set attribute as dictionary item."""
        self[name] = value

    @staticmethod
    def _load_dict_from_file_no_base(filename):
        """Load a Python config file as a dict without resolving _base_ inheritance.

        Imports the file as a Python module and extracts all non-dunder attributes.

        Args:
            filename: Path to the Python config file.

        Returns:
            dict: Raw configuration dictionary.
        """
        f_dir = os.path.dirname(filename)
        f_name = os.path.basename(filename)
        module_name = f_name[:-3]

        sys.path.insert(0, f_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }
        del sys.modules[module_name]
        return cfg

    @staticmethod
    def _find_configs_root(start_path):
        """Find the configs root directory by looking for 'configs' folder.

        Args:
            start_path: Starting path to search from.

        Returns:
            str: Path to configs root directory, or None if not found.
        """
        current = os.path.abspath(start_path)
        # Walk up to find configs directory
        while current != os.path.dirname(current):  # until filesystem root
            if os.path.basename(current) == "configs":
                return current
            # Check if current directory has a configs subdirectory
            configs_path = os.path.join(current, "configs")
            if os.path.isdir(configs_path):
                return configs_path
            current = os.path.dirname(current)
        return None

    @staticmethod
    def _resolve_base_path(base_path, cfg_dir, configs_root):
        """Resolve base config path.

        Supports:
        - "@/path" : relative to configs root directory
        - "path"   : relative to current config file directory

        Args:
            base_path: Base config path (may start with @/).
            cfg_dir: Directory of current config file.
            configs_root: Root configs directory.

        Returns:
            str: Resolved absolute path.
        """
        if base_path.startswith(CONFIGS_ROOT_PREFIX):
            # Relative to configs root directory
            if configs_root is None:
                raise ValueError(
                    f"Cannot resolve '{base_path}': configs root directory not found. "
                    "Make sure your config file is under a 'configs' directory."
                )
            relative_path = base_path[len(CONFIGS_ROOT_PREFIX) :]
            return os.path.join(configs_root, relative_path)
        else:
            # Relative to current config file directory
            return os.path.join(cfg_dir, base_path)

    @staticmethod
    def _load_dict_from_file(filename, configs_root=None):
        """Load config file with _base_ inheritance support.

        Recursively loads and merges base configurations specified
        by the _base_ key.

        Supports special path prefix:
        - "@/path" : relative to configs root directory (e.g., "@/_base_/models/gpt.py")
        - "path"   : relative to current config file directory (original behavior)

        Args:
            filename: Path to Python config file.
            configs_root: Optional configs root directory for @ prefix resolution.

        Returns:
            dict: Merged configuration dictionary.
        """
        cfg = Config._load_dict_from_file_no_base(filename)
        cfg_dir = os.path.dirname(os.path.abspath(filename))

        # Auto-detect configs root directory
        if configs_root is None:
            configs_root = Config._find_configs_root(cfg_dir)

        if BASE_KEY in cfg:
            if isinstance(cfg[BASE_KEY], list):
                base_filenames = cfg[BASE_KEY]
            else:
                assert isinstance(cfg[BASE_KEY], str)
                base_filenames = [cfg[BASE_KEY]]
            cfg_base = {}
            for bfn in base_filenames:
                resolved_path = Config._resolve_base_path(bfn, cfg_dir, configs_root)
                Config.merge_dict_b2a(
                    cfg_base, Config._load_dict_from_file(resolved_path, configs_root)
                )
            cfg.pop(BASE_KEY)
            Config.merge_dict_b2a(cfg_base, cfg)
            cfg = cfg_base
        return cfg

    @staticmethod
    def merge_dict_b2a(a, b):
        """Merge dict b into dict a recursively.

        Supports special keys:
        - _cover_: Replace entire dict instead of merging.

        Args:
            a: Target dict to merge into (modified in place).
            b: Source dict to merge from.
        """

        def clear_cover_key(a):
            if not isinstance(a, dict):
                return a
            out = copy.deepcopy(a)
            if COVER_KEY in out:
                out.pop(COVER_KEY)
            for k, v in out.items():
                out[k] = clear_cover_key(v)
            return out

        assert isinstance(a, dict)
        assert isinstance(b, dict)
        if COVER_KEY in b:
            a.clear()
            temp = clear_cover_key(copy.deepcopy(b))
            a.update(temp)
            return
        for k, v in b.items():
            if (
                (not (k in a))
                or (isinstance(v, dict) and v.get(COVER_KEY, False))
                or (not isinstance(v, dict))
                or (not isinstance(a[k], dict))
            ):
                a[k] = clear_cover_key(copy.deepcopy(v))
            else:
                Config.merge_dict_b2a(a[k], v)

    def load_from_file(self, filename):
        """Load configuration from a Python file.

        Args:
            filename: Path to the config file.
        """
        cfg = Config._load_dict_from_file(filename)
        self.clear()
        self.update(self.dfs(cfg))
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]
        self.work_dir = f"{self.work_dir}/{self.name}"

    def dfs(self, cfg_other):
        """Deep-first search conversion of nested dicts to Config objects.

        Args:
            cfg_other: Input configuration (dict, list, or value).

        Returns:
            Converted Config object or value.
        """
        if isinstance(cfg_other, dict):
            now_cfg = Config()
            for k, d in cfg_other.items():
                if inspect.ismodule(d):
                    continue
                now_cfg[k] = self.dfs(d)
        elif isinstance(cfg_other, list):
            now_cfg = [self.dfs(d) for d in cfg_other if (not inspect.ismodule(d))]
        else:
            now_cfg = copy.deepcopy(cfg_other)
        return now_cfg

    def dump(self):
        """Convert Config to plain dict recursively.

        Returns:
            dict: Plain dictionary representation.
        """
        now = dict()
        for k, d in self.items():
            if isinstance(d, Config):
                d = d.dump()
            if isinstance(d, list):
                d = [dd.dump() if isinstance(dd, Config) else dd for dd in d]
            now[k] = d
        return now


_cfg = Config()


def init_cfg(filename):
    """Initialize global config from file.

    Args:
        filename: Path to the config file.
    """
    logger.info("Loading config from: %s", filename)
    _cfg.load_from_file(filename)


def get_cfg():
    """Get the global config instance.

    Returns:
        Config: Global configuration object.
    """
    return _cfg


def save_cfg(save_file):
    """Save current config to YAML file.

    Args:
        save_file: Output file path.
    """
    with open(save_file, "w") as f:
        f.write(yaml.dump(_cfg.dump()))


def print_cfg():
    """Print current config in YAML format."""
    data = yaml.dump(_cfg.dump(), sort_keys=True)
    logger.info(data)
