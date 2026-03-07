"""
Unit tests for datastudio.utils.config module.

Tests cover:
- Config class
- Configuration loading and merging
- Global config functions
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from datastudio.utils.config import Config, get_cfg, init_cfg, print_cfg, save_cfg


class TestConfigInit:
    """Tests for Config initialization."""

    def test_init_empty(self):
        """Test initialization without arguments."""
        config = Config()
        assert config is not None
        assert len(config) == 0

    def test_init_with_py_file(self, temp_dir):
        """Test initialization with Python config file."""
        # Create a Python config file
        config_path = os.path.join(temp_dir, "test_config.py")
        with open(config_path, "w") as f:
            f.write("name = 'test'\nvalue = 42\n")

        config = Config(config_path)
        assert config.name == "test"
        assert config.value == 42


class TestConfigLoadFromFile:
    """Tests for Config.load_from_file method."""

    def test_load_python_file(self, temp_dir):
        """Test loading Python config file."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("model_name = 'gpt-4'\nbatch_size = 32\n")

        config = Config()
        config.load_from_file(config_path)
        assert config.model_name == "gpt-4"
        assert config.batch_size == 32

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file raises error."""
        config = Config()
        with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
            config.load_from_file(os.path.join(temp_dir, "nonexistent.py"))

    def test_load_with_base_inheritance(self, temp_dir):
        """Test loading config with _base_ inheritance."""
        # Create base config
        base_path = os.path.join(temp_dir, "base.py")
        with open(base_path, "w") as f:
            f.write("base_value = 'from_base'\nshared = 1\n")

        # Create child config
        child_path = os.path.join(temp_dir, "child.py")
        with open(child_path, "w") as f:
            f.write("_base_ = 'base.py'\nchild_value = 'from_child'\nshared = 2\n")

        config = Config(child_path)
        assert config.base_value == "from_base"
        assert config.child_value == "from_child"
        assert config.shared == 2  # Child overrides base


class TestConfigMergeDict:
    """Tests for Config.merge_dict_b2a method."""

    def test_merge_simple(self):
        """Test simple dict merge."""
        a = {"x": 1}
        b = {"y": 2}
        Config.merge_dict_b2a(a, b)
        assert a == {"x": 1, "y": 2}

    def test_merge_override(self):
        """Test merge with override."""
        a = {"x": 1}
        b = {"x": 2}
        Config.merge_dict_b2a(a, b)
        assert a == {"x": 2}

    def test_merge_nested(self):
        """Test nested dict merge."""
        a = {"outer": {"inner": 1}}
        b = {"outer": {"inner2": 2}}
        Config.merge_dict_b2a(a, b)
        assert a == {"outer": {"inner": 1, "inner2": 2}}

    def test_merge_empty_b(self):
        """Test merge with empty b."""
        a = {"x": 1}
        b = {}
        Config.merge_dict_b2a(a, b)
        assert a == {"x": 1}


class TestConfigDfs:
    """Tests for Config.dfs method."""

    def test_dfs_simple(self):
        """Test dfs with simple dict."""
        config = Config()
        result = config.dfs({"key": "value"})
        assert isinstance(result, Config)
        assert result["key"] == "value"

    def test_dfs_nested(self):
        """Test dfs with nested dict."""
        config = Config()
        result = config.dfs({"outer": {"inner": "value"}})
        assert isinstance(result, Config)
        assert isinstance(result["outer"], Config)

    def test_dfs_with_list(self):
        """Test dfs with list."""
        config = Config()
        result = config.dfs({"items": [1, 2, 3]})
        assert result["items"] == [1, 2, 3]


class TestConfigDump:
    """Tests for Config.dump method."""

    def test_dump_simple(self, temp_dir):
        """Test dumping simple config."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key = 'value'\n")

        config = Config(config_path)
        dumped = config.dump()
        assert isinstance(dumped, dict)
        assert dumped["key"] == "value"

    def test_dump_nested(self, temp_dir):
        """Test dumping nested config."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("outer = dict(inner='value')\n")

        config = Config(config_path)
        dumped = config.dump()
        assert isinstance(dumped, dict)
        assert dumped["outer"]["inner"] == "value"

    def test_dump_preserves_types(self, temp_dir):
        """Test dump preserves data types."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("int_val = 42\nfloat_val = 3.14\nbool_val = True\n")

        config = Config(config_path)
        dumped = config.dump()
        assert isinstance(dumped["int_val"], int)
        assert isinstance(dumped["float_val"], float)
        assert isinstance(dumped["bool_val"], bool)


class TestInitCfg:
    """Tests for init_cfg function."""

    def test_init_cfg_py(self, temp_dir):
        """Test init_cfg with Python file."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("test_key = 'test_value'\n")

        init_cfg(config_path)
        cfg = get_cfg()
        assert cfg.test_key == "test_value"

    def test_init_cfg_nonexistent(self, temp_dir):
        """Test init_cfg with nonexistent file."""
        with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
            init_cfg(os.path.join(temp_dir, "nonexistent.py"))


class TestGetCfg:
    """Tests for get_cfg function."""

    def test_get_cfg_returns_config(self, temp_dir):
        """Test get_cfg returns Config instance."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key = 'value'\n")

        init_cfg(config_path)
        cfg = get_cfg()
        assert isinstance(cfg, Config)

    def test_get_cfg_singleton(self, temp_dir):
        """Test get_cfg returns same instance."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key = 'value'\n")

        init_cfg(config_path)
        cfg1 = get_cfg()
        cfg2 = get_cfg()
        assert cfg1 is cfg2


class TestSaveCfg:
    """Tests for save_cfg function."""

    def test_save_cfg(self, temp_dir):
        """Test save_cfg saves to YAML."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key = 'value'\n")

        init_cfg(config_path)

        save_path = os.path.join(temp_dir, "saved.yaml")
        save_cfg(save_path)

        assert os.path.exists(save_path)
        with open(save_path, "r") as f:
            content = f.read()
        assert "key" in content


class TestPrintCfg:
    """Tests for print_cfg function."""

    def test_print_cfg(self, temp_dir, caplog):
        """Test print_cfg outputs config via logging."""
        import logging

        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key = 'value'\n")

        init_cfg(config_path)
        with caplog.at_level(logging.INFO, logger="datastudio.utils.config"):
            print_cfg()

        assert "key" in caplog.text


class TestConfigAttributeAccess:
    """Tests for Config attribute access."""

    def test_attribute_get(self, temp_dir):
        """Test getting attribute."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("my_attr = 'my_value'\n")

        config = Config(config_path)
        assert config.my_attr == "my_value"

    def test_attribute_set(self):
        """Test setting attribute."""
        config = Config()
        config.new_attr = "new_value"
        assert config.new_attr == "new_value"

    def test_attribute_none_for_missing(self):
        """Test missing attribute returns None."""
        config = Config()
        assert config.nonexistent is None


class TestConfigEdgeCases:
    """Edge case tests for Config."""

    def test_empty_py_file(self, temp_dir):
        """Test loading empty Python file."""
        config_path = os.path.join(temp_dir, "empty.py")
        with open(config_path, "w") as f:
            f.write("")

        config = Config(config_path)
        assert config is not None

    def test_py_with_none_values(self, temp_dir):
        """Test Python file with None values."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("value = None\n")

        config = Config(config_path)
        assert config.value is None

    def test_deep_nesting(self, temp_dir):
        """Test deeply nested config."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("a = dict(b=dict(c=dict(d='deep')))\n")

        config = Config(config_path)
        assert config.a.b.c.d == "deep"

    def test_special_characters_in_keys(self, temp_dir):
        """Test config with special characters in keys."""
        config_path = os.path.join(temp_dir, "config.py")
        with open(config_path, "w") as f:
            f.write("key_with_underscore = 'value'\n")

        config = Config(config_path)
        assert config.key_with_underscore == "value"
