"""
Unit tests for datastudio.utils.registry module.

Tests cover:
- Registry class
- build_from_cfg function
- Global registries (DATALOADER, PIPELINES, OPERATORS, etc.)
"""

import pytest

from datastudio.utils.registry import (
    DATALOADER,
    DATASAVER,
    HOOKS,
    MODELS,
    OPERATORS,
    PIPELINES,
    REQUESTS,
    Registry,
    build_from_cfg,
)


class TestRegistryInit:
    """Tests for Registry initialization."""

    def test_init_default(self):
        """Test initialization with default name."""
        registry = Registry()
        assert registry is not None

    def test_init_with_name(self):
        """Test initialization with custom name."""
        registry = Registry(name="test_registry")
        assert registry._name == "test_registry"


class TestRegistryRegisterModule:
    """Tests for Registry.register_module method."""

    @pytest.fixture
    def registry(self):
        """Create clean registry for testing."""
        return Registry(name="test")

    def test_register_with_decorator(self, registry):
        """Test registering module with decorator."""

        @registry.register_module()
        class TestClass:
            pass

        assert "TestClass" in registry

    def test_register_with_custom_name(self, registry):
        """Test registering module with custom name."""

        @registry.register_module(name="CustomName")
        class TestClass:
            pass

        assert "CustomName" in registry
        assert "TestClass" not in registry

    def test_register_directly(self, registry):
        """Test registering module directly."""

        class TestClass:
            pass

        registry.register_module(module=TestClass)
        assert "TestClass" in registry

    def test_register_function(self, registry):
        """Test registering function."""

        @registry.register_module()
        def test_function():
            pass

        assert "test_function" in registry

    def test_register_duplicate_raises_error(self, registry):
        """Test registering duplicate name raises error."""

        @registry.register_module()
        class TestClass:
            pass

        with pytest.raises((KeyError, ValueError)):

            @registry.register_module()
            class TestClass:
                pass


class TestRegistryGet:
    """Tests for Registry.get method."""

    @pytest.fixture
    def registry_with_modules(self):
        """Create registry with registered modules."""
        registry = Registry(name="test")

        @registry.register_module()
        class ModuleA:
            pass

        @registry.register_module()
        class ModuleB:
            pass

        return registry

    def test_get_existing_module(self, registry_with_modules):
        """Test getting existing module."""
        module = registry_with_modules.get("ModuleA")
        assert module is not None
        assert module.__name__ == "ModuleA"

    def test_get_nonexistent_module(self, registry_with_modules):
        """Test getting nonexistent module raises error."""
        with pytest.raises(KeyError):
            registry_with_modules.get("NonExistent")

    def test_get_case_sensitive(self, registry_with_modules):
        """Test get is case sensitive."""
        with pytest.raises(KeyError):
            registry_with_modules.get("modulea")


class TestRegistryContains:
    """Tests for Registry.__contains__ method."""

    @pytest.fixture
    def registry_with_modules(self):
        """Create registry with registered modules."""
        registry = Registry(name="test")

        @registry.register_module()
        class TestModule:
            pass

        return registry

    def test_contains_existing(self, registry_with_modules):
        """Test __contains__ for existing module."""
        assert "TestModule" in registry_with_modules

    def test_contains_nonexistent(self, registry_with_modules):
        """Test __contains__ for nonexistent module."""
        assert "NonExistent" not in registry_with_modules


class TestBuildFromCfg:
    """Tests for build_from_cfg function."""

    @pytest.fixture
    def registry(self):
        """Create registry with test module."""
        registry = Registry(name="test")

        @registry.register_module()
        class TestClass:
            def __init__(self, param1=None, param2=None):
                self.param1 = param1
                self.param2 = param2

        return registry

    def test_build_from_dict_config(self, registry):
        """Test building from dict config."""
        cfg = {"type": "TestClass", "param1": "value1", "param2": "value2"}

        instance = build_from_cfg(cfg, registry)

        assert instance is not None
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"

    def test_build_from_string_config(self, registry):
        """Test building from string config (type name only)."""
        cfg = "TestClass"

        instance = build_from_cfg(cfg, registry)

        assert instance is not None

    def test_build_with_extra_kwargs(self, registry):
        """Test building with extra kwargs."""
        cfg = {"type": "TestClass", "param1": "from_cfg"}

        instance = build_from_cfg(cfg, registry, param2="from_kwargs")

        assert instance.param1 == "from_cfg"
        assert instance.param2 == "from_kwargs"

    def test_build_none_config(self, registry):
        """Test building with None config."""
        result = build_from_cfg(None, registry)
        assert result is None

    def test_build_missing_type(self, registry):
        """Test building without type field raises error."""
        cfg = {"param1": "value"}

        with pytest.raises((KeyError, TypeError)):
            build_from_cfg(cfg, registry)

    def test_build_nonexistent_type(self, registry):
        """Test building nonexistent type raises error."""
        cfg = {"type": "NonExistent"}

        with pytest.raises(KeyError):
            build_from_cfg(cfg, registry)


class TestGlobalRegistries:
    """Tests for global registry instances."""

    def test_dataloader_registry_exists(self):
        """Test DATALOADER registry exists."""
        assert DATALOADER is not None
        assert isinstance(DATALOADER, Registry)

    def test_pipelines_registry_exists(self):
        """Test PIPELINES registry exists."""
        assert PIPELINES is not None
        assert isinstance(PIPELINES, Registry)

    def test_operators_registry_exists(self):
        """Test OPERATORS registry exists."""
        assert OPERATORS is not None
        assert isinstance(OPERATORS, Registry)

    def test_models_registry_exists(self):
        """Test MODELS registry exists."""
        assert MODELS is not None
        assert isinstance(MODELS, Registry)

    def test_datasaver_registry_exists(self):
        """Test DATASAVER registry exists."""
        assert DATASAVER is not None
        assert isinstance(DATASAVER, Registry)

    def test_hooks_registry_exists(self):
        """Test HOOKS registry exists."""
        assert HOOKS is not None
        assert isinstance(HOOKS, Registry)

    def test_requests_registry_exists(self):
        """Test REQUESTS registry exists."""
        assert REQUESTS is not None
        assert isinstance(REQUESTS, Registry)


class TestGlobalRegistriesContent:
    """Tests for content of global registries."""

    def test_operators_has_filters(self):
        """Test OPERATORS registry has filter classes."""
        assert "ConvLengthFilter" in OPERATORS
        assert "LengthAnomalyFilter" in OPERATORS

    def test_operators_has_rewriters(self):
        """Test OPERATORS registry has rewriter classes."""
        assert "RemoveThinkRewriter" in OPERATORS
        assert "SplitRewriter" in OPERATORS

    def test_pipelines_has_pipeline(self):
        """Test PIPELINES registry has Pipeline."""
        assert "Pipeline" in PIPELINES
        assert "SubPipeline" in PIPELINES

    def test_dataloader_has_standard(self):
        """Test DATALOADER registry has StandardDataLoader."""
        assert "StandardDataLoader" in DATALOADER

    def test_datasaver_has_standard(self):
        """Test DATASAVER registry has StandardDataSaver."""
        assert "StandardDataSaver" in DATASAVER


class TestRegistryEdgeCases:
    """Edge case tests for Registry."""

    def test_register_lambda(self):
        """Test registering lambda function."""
        registry = Registry(name="test")

        # Lambda registration may require explicit name
        registry.register_module(name="my_lambda", module=lambda x: x * 2)

        assert "my_lambda" in registry

    def test_register_builtin_type(self):
        """Test registering builtin type."""
        registry = Registry(name="test")

        registry.register_module(name="my_list", module=list)

        assert "my_list" in registry

    def test_empty_registry_iteration(self):
        """Test iterating empty registry."""
        registry = Registry(name="test")

        # Should be iterable even when empty
        modules = (
            list(registry._module_dict.keys())
            if hasattr(registry, "_module_dict")
            else []
        )
        assert modules == []

    def test_registry_repr(self):
        """Test registry string representation."""
        registry = Registry(name="test_registry")

        repr_str = repr(registry) if hasattr(registry, "__repr__") else str(registry)
        assert isinstance(repr_str, str)


class TestRegistryBuildFromCfg:
    """Tests for build_from_cfg function."""

    @pytest.fixture
    def registry_with_class(self):
        """Create a registry with a simple test class registered."""
        reg = Registry(name="build_test")

        @reg.register_module()
        class SimpleClass:
            def __init__(self, x=0, y=0):
                self.x = x
                self.y = y

        return reg

    def test_build_from_cfg_valid_config_and_registry(self, registry_with_class):
        """Test build_from_cfg with valid config and registry."""
        cfg = {"type": "SimpleClass", "x": 10, "y": 20}
        obj = build_from_cfg(cfg, registry_with_class)

        assert obj is not None
        assert obj.x == 10
        assert obj.y == 20

    def test_build_from_cfg_none_config_returns_none(self, registry_with_class):
        """Test build_from_cfg with None config returns None."""
        result = build_from_cfg(None, registry_with_class)
        assert result is None

    def test_build_from_cfg_missing_type_key_raises_error(self, registry_with_class):
        """Test build_from_cfg with missing 'type' key raises error."""
        cfg = {"x": 5, "y": 10}  # no 'type' key
        with pytest.raises(TypeError, match="must contain 'type'"):
            build_from_cfg(cfg, registry_with_class)

    def test_build_from_cfg_unregistered_type_raises_keyerror(
        self, registry_with_class
    ):
        """Test build_from_cfg with unregistered type raises KeyError."""
        cfg = {"type": "DoesNotExist"}
        with pytest.raises(KeyError):
            build_from_cfg(cfg, registry_with_class)

    def test_build_from_cfg_passes_extra_kwargs_to_constructor(
        self, registry_with_class
    ):
        """Test build_from_cfg passes extra kwargs to constructor."""
        cfg = {"type": "SimpleClass", "x": 1}
        obj = build_from_cfg(cfg, registry_with_class, y=99)

        assert obj.x == 1
        assert obj.y == 99


class TestGlobalRegistriesContents:
    """Tests that global registries contain expected modules."""

    def test_operators_contains_all_filter_types(self):
        """Test OPERATORS contains all filter types."""
        expected_filters = [
            "ConvLengthFilter",
            "LengthAnomalyFilter",
            "TextRepeatFilter",
            "ImageSizeFilter",
            "ImageExtFilter",
            "ImageAspectRatioFilter",
            "ResponseTagFilter",
        ]
        for name in expected_filters:
            assert name in OPERATORS, f"Expected filter '{name}' not found in OPERATORS"

    def test_operators_contains_all_rewriter_types(self):
        """Test OPERATORS contains all rewriter types."""
        expected_rewriters = [
            "RemoveThinkRewriter",
            "SplitRewriter",
            "NormPromptRewriter",
            "NormImageTagRewriter",
            "NormThinkRewriter",
            "RemoveAnswerRewriter",
            "RemoveReasonRewriter",
            "AddNoThinkRewriter",
            "NormMultiTurnPromptRewriter",
        ]
        for name in expected_rewriters:
            assert (
                name in OPERATORS
            ), f"Expected rewriter '{name}' not found in OPERATORS"

    def test_operators_contains_mllm_operators(self):
        """Test OPERATORS contains MLLM operators."""
        expected_mllm = [
            "MLLMFilter",
            "MLLMRewriter",
            "SelectiveMLLMRewriter",
        ]
        for name in expected_mllm:
            assert (
                name in OPERATORS
            ), f"Expected MLLM operator '{name}' not found in OPERATORS"

    def test_pipelines_contains_pipeline_and_subpipeline(self):
        """Test PIPELINES contains Pipeline and SubPipeline."""
        assert "Pipeline" in PIPELINES
        assert "SubPipeline" in PIPELINES

    def test_dataloader_contains_standard_data_loader(self):
        """Test DATALOADER contains StandardDataLoader."""
        assert "StandardDataLoader" in DATALOADER

    def test_datasaver_contains_standard_data_saver(self):
        """Test DATASAVER contains StandardDataSaver."""
        assert "StandardDataSaver" in DATASAVER

    def test_requests_contains_request_builder(self):
        """Test REQUESTS contains RequestBuilder."""
        assert "RequestBuilder" in REQUESTS
