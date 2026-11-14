"""
Test cases for dna_factory/utils/colorize_args.py

This module tests the argument parsing and colorization functionality.
"""

import dataclasses
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.utils.colorize_args import (
    UserSpecifiedArgs,
    colorize_user_args,
    find_redundant_args,
    format_args_with_colors,
    parse_user_args,
    parse_yaml_config,
    values_equal,
)


class TestParseUserArgs:
    """Test cases for parse_user_args function"""

    def test_parse_long_options(self):
        """Test parsing long options with -- prefix"""
        args = ["--learning_rate", "0.001", "--batch_size", "32"]
        result = parse_user_args(args)
        assert "learning_rate" in result
        assert "batch_size" in result

    def test_parse_options_with_dashes(self):
        """Test that dashes in option names are converted to underscores"""
        args = ["--learning-rate", "0.001", "--per-device-batch-size", "32"]
        result = parse_user_args(args)
        assert "learning_rate" in result
        assert "per_device_batch_size" in result

    def test_parse_short_options(self):
        """Test parsing short options with - prefix"""
        args = ["-n", "100", "-v"]
        result = parse_user_args(args)
        assert "n" in result
        assert "v" in result

    def test_parse_boolean_flags(self):
        """Test parsing boolean flags without values"""
        args = ["--verbose", "--debug"]
        result = parse_user_args(args)
        assert "verbose" in result
        assert "debug" in result

    def test_parse_mixed_arguments(self):
        """Test parsing mixed short and long options"""
        args = ["--epochs", "10", "-v", "--learning-rate", "0.001"]
        result = parse_user_args(args)
        assert "epochs" in result
        assert "v" in result
        assert "learning_rate" in result

    def test_parse_config_file_option(self):
        """Test parsing config file option and extracting its keys"""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model_name: test\n")
            f.write("learning_rate: 0.001\n")
            config_path = f.name

        try:
            args = ["--config", config_path, "--batch_size", "32"]
            result = parse_user_args(args)

            # Should include config option itself
            assert "config" in result
            # Should include keys from the config file
            assert "model_name" in result
            assert "learning_rate" in result
            # Should include command line args
            assert "batch_size" in result
        finally:
            os.unlink(config_path)

    def test_parse_long_options_with_equals(self):
        """Test that --name=value yields the same field names as --name value"""
        args = ["--learning_rate=0.001", "--batch_size=32"]
        result = parse_user_args(args)
        assert "learning_rate" in result
        assert "batch_size" in result

    def test_parse_options_with_dashes_and_equals(self):
        """Test that only the option name is dash-converted, never the value"""
        args = ["--per-device-batch-size=32", "--learning-rate=3e-5"]
        result = parse_user_args(args)
        assert "per_device_batch_size" in result
        assert "learning_rate" in result
        assert not any("=" in name for name in result)
        assert "3e_5" not in str(result)

    def test_parse_equals_and_space_forms_are_equivalent(self):
        """Test that both CLI forms produce identical results"""
        spaced = parse_user_args(["--learning_rate", "0.001", "--packing", "true"])
        equals = parse_user_args(["--learning_rate=0.001", "--packing=true"])
        assert spaced == equals

    def test_parse_config_file_option_with_equals(self):
        """Test that --config=path also pulls in the config file's keys"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model_name: test\n")
            config_path = f.name

        try:
            result = parse_user_args([f"--config={config_path}", "--batch_size=32"])
            assert "config" in result
            assert "model_name" in result
            assert "batch_size" in result
        finally:
            os.unlink(config_path)

    def test_parse_config_without_value(self):
        """Test that a valueless --config does not crash"""
        assert parse_user_args(["--config"]) == {"config"}
        assert parse_user_args(["--config="]) == {"config"}

    def test_empty_args(self):
        """Test with empty arguments list"""
        args = []
        result = parse_user_args(args)
        assert len(result) == 0


class TestParseYamlConfig:
    """Test cases for parse_yaml_config function"""

    def test_parse_flat_yaml(self):
        """Test parsing flat YAML structure"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key1: value1\n")
            f.write("key2: value2\n")
            f.write("key3: 123\n")
            config_path = f.name

        try:
            result = parse_yaml_config(config_path)
            assert "key1" in result
            assert "key2" in result
            assert "key3" in result
        finally:
            os.unlink(config_path)

    def test_parse_nested_yaml(self):
        """Test parsing nested YAML structure"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n")
            f.write("  name: test_model\n")
            f.write("  size: 100\n")
            f.write("training:\n")
            f.write("  epochs: 10\n")
            f.write("  learning_rate: 0.001\n")
            config_path = f.name

        try:
            result = parse_yaml_config(config_path)
            # Should include top-level keys
            assert "model" in result
            assert "training" in result
            # Should include nested keys with prefix
            assert "model_name" in result
            assert "model_size" in result
            assert "training_epochs" in result
            assert "training_learning_rate" in result
        finally:
            os.unlink(config_path)

    def test_parse_deeply_nested_yaml(self):
        """Test parsing deeply nested YAML structure"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("level1:\n")
            f.write("  level2:\n")
            f.write("    level3:\n")
            f.write("      key: value\n")
            config_path = f.name

        try:
            result = parse_yaml_config(config_path)
            assert "level1" in result
            assert "level1_level2" in result
            assert "level1_level2_level3" in result
            assert "level1_level2_level3_key" in result
        finally:
            os.unlink(config_path)

    def test_parse_empty_yaml(self):
        """Test parsing empty YAML file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = f.name

        try:
            result = parse_yaml_config(config_path)
            assert len(result) == 0
        finally:
            os.unlink(config_path)

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file returns empty set"""
        result = parse_yaml_config("/nonexistent/path/to/config.yaml")
        assert len(result) == 0

    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML returns empty set with warning"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: structure: :\n")
            config_path = f.name

        try:
            # Should not raise exception, just print warning
            result = parse_yaml_config(config_path)
            # Result may be empty or partial depending on YAML parser
            assert isinstance(result, set)
        finally:
            os.unlink(config_path)


class TestColorizeUserArgs:
    """Test cases for colorize_user_args function"""

    def test_colorize_single_line(self):
        """Test colorizing a single line with user argument"""
        text = "'learning_rate': 0.001"
        user_args = {"learning_rate"}
        result = colorize_user_args(text, user_args)

        # Should contain ANSI color codes
        assert "\033[1;95m" in result  # PINK color
        assert "\033[0m" in result  # RESET

    def test_colorize_multiple_lines(self):
        """Test colorizing multiple lines with different arguments"""
        text = "'learning_rate': 0.001\n'batch_size': 32\n'epochs': 10"
        user_args = {"learning_rate", "epochs"}
        result = colorize_user_args(text, user_args)

        lines = result.split("\n")
        # First and third lines should be colored
        assert "\033[1;95m" in lines[0]
        assert "\033[1;95m" in lines[2]
        # Second line should not be colored
        assert "\033[1;95m" not in lines[1]

    def test_colorize_no_matching_args(self):
        """Test text with no matching user arguments"""
        text = "'other_param': value"
        user_args = {"learning_rate"}
        result = colorize_user_args(text, user_args)

        # Should not contain color codes
        assert "\033[1;95m" not in result

    def test_colorize_unquoted_keys(self):
        """Test colorizing unquoted field names"""
        text = "learning_rate: 0.001"
        user_args = {"learning_rate"}
        result = colorize_user_args(text, user_args)

        # Should contain color codes
        assert "\033[1;95m" in result

    def test_colorize_empty_text(self):
        """Test colorizing empty text"""
        text = ""
        user_args = {"learning_rate"}
        result = colorize_user_args(text, user_args)
        assert result == ""

    def test_colorize_empty_user_args(self):
        """Test with empty user arguments set"""
        text = "'learning_rate': 0.001"
        user_args = set()
        result = colorize_user_args(text, user_args)

        # Should not contain color codes
        assert "\033[1;95m" not in result


class TestFormatArgsWithColors:
    """Test cases for format_args_with_colors function"""

    def test_format_simple_dict(self):
        """Test formatting simple dictionary with colors"""
        args_dict = {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
        user_args = {"learning_rate", "epochs"}

        result = format_args_with_colors(args_dict, user_args)

        # Should be formatted as string
        assert isinstance(result, str)
        # Should contain the values
        assert "0.001" in result
        assert "32" in result
        assert "10" in result
        # Should contain color codes for user args
        assert "\033[1;95m" in result

    def test_format_nested_dict(self):
        """Test formatting nested dictionary"""
        args_dict = {
            "model": {"name": "test", "size": 100},
            "training": {"epochs": 10, "lr": 0.001},
        }
        user_args = {"model", "epochs"}

        result = format_args_with_colors(args_dict, user_args)

        assert isinstance(result, str)
        assert "model" in result
        assert "training" in result

    def test_format_empty_dict(self):
        """Test formatting empty dictionary"""
        args_dict = {}
        user_args = set()

        result = format_args_with_colors(args_dict, user_args)
        assert isinstance(result, str)

    def test_format_with_various_types(self):
        """Test formatting dictionary with various value types"""
        args_dict = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "list_val": [1, 2, 3],
            "dict_val": {"nested": "value"},
        }
        user_args = {"string_val", "bool_val"}

        result = format_args_with_colors(args_dict, user_args)

        assert isinstance(result, str)
        assert "test" in result
        assert "42" in result
        assert "True" in result
        assert "None" in result


PINK = "\033[1;95m"
GRAY = "\033[90m"


class TestValuesEqual:
    """Test cases for values_equal (parsed-value, deep comparison)"""

    def test_numbers_compare_by_value(self):
        assert values_equal(1, 1.0)
        assert values_equal(1e-5, 1.0e-5)
        assert values_equal(float("1e-5"), 0.00001)
        assert not values_equal(1e-5, 2e-5)

    def test_none(self):
        assert values_equal(None, None)
        assert not values_equal(None, 0)
        assert not values_equal("none", None)

    def test_lists_and_tuples(self):
        assert values_equal(["a", "b"], ["a", "b"])
        assert values_equal(("a", "b"), ["a", "b"])
        assert not values_equal(["a", "b"], ["b", "a"])
        assert not values_equal(["a"], ["a", "b"])

    def test_nested_dicts_ignore_key_order(self):
        assert values_equal(
            {"min_lr_rate": 0.1, "opts": [1, {"x": 2}]},
            {"opts": [1, {"x": 2.0}], "min_lr_rate": 0.1},
        )
        assert not values_equal({"opts": [1, {"x": 2}]}, {"opts": [1, {"x": 3}]})

    def test_dataclass_items(self):
        @dataclasses.dataclass(eq=False)
        class Item:
            path: str
            weight: float = 1.0

        assert values_equal(
            [Item("org/a"), Item("org/b", 2)], [Item("org/a"), Item("org/b", 2.0)]
        )
        assert not values_equal([Item("org/a")], [Item("org/c")])

    def test_uncomparable_values_count_as_different(self):
        class Broken:
            def __eq__(self, other):
                raise RuntimeError("no comparison")

        assert not values_equal(Broken(), Broken())


class TestFindRedundantArgs:
    """Test cases for find_redundant_args"""

    def test_equal_to_default_is_redundant(self):
        parsed = [SimpleNamespace(learning_rate=1e-5, packing=True, max_length=2048)]
        default = [SimpleNamespace(learning_rate=1.0e-5, packing=True, max_length=1024)]
        result = find_redundant_args(
            {"learning_rate", "packing", "max_length"}, parsed, default
        )
        assert result == {"learning_rate", "packing"}

    def test_unspecified_args_are_never_reported(self):
        parsed = [SimpleNamespace(learning_rate=1e-5, packing=True)]
        default = [SimpleNamespace(learning_rate=1e-5, packing=True)]
        assert find_redundant_args({"packing"}, parsed, default) == {"packing"}
        assert find_redundant_args(set(), parsed, default) == set()

    def test_names_that_are_not_arguments_stay_user_specified(self):
        parsed = [SimpleNamespace(learning_rate=1e-5)]
        default = [SimpleNamespace(learning_rate=1e-5)]
        assert find_redundant_args({"config", "unknown_key"}, parsed, default) == set()

    def test_argument_missing_from_defaults_is_not_redundant(self):
        parsed = [SimpleNamespace(extra=1)]
        default = [SimpleNamespace()]
        assert find_redundant_args({"extra"}, parsed, default) == set()

    def test_nested_and_list_values(self):
        parsed = [
            SimpleNamespace(
                reward_funcs=["a.b", "c.d"],
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
                report_to=["none"],
            )
        ]
        default = [
            SimpleNamespace(
                reward_funcs=["a.b", "c.d"],
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
                report_to=["wandb"],
            )
        ]
        result = find_redundant_args(
            {"reward_funcs", "lr_scheduler_kwargs", "report_to"}, parsed, default
        )
        assert result == {"reward_funcs", "lr_scheduler_kwargs"}

    def test_same_name_in_several_objects_must_match_everywhere(self):
        parsed = [SimpleNamespace(seed=42), SimpleNamespace(seed=7)]
        default = [SimpleNamespace(seed=42), SimpleNamespace(seed=42)]
        assert find_redundant_args({"seed"}, parsed, default) == set()


class TestUserSpecifiedArgs:
    """Test cases for the UserSpecifiedArgs set"""

    def test_behaves_like_a_set_of_names(self):
        args = UserSpecifiedArgs({"learning_rate", "packing"}, {"packing"})
        assert args == {"learning_rate", "packing"}
        assert "packing" in args
        assert args.redundant == {"packing"}

    def test_redundant_is_limited_to_specified_names(self):
        args = UserSpecifiedArgs({"learning_rate"}, {"packing"})
        assert args.redundant == frozenset()

    def test_defaults_to_no_redundant_args(self):
        assert UserSpecifiedArgs({"learning_rate"}).redundant == frozenset()


class TestColorizeThreeStates:
    """Overriding args are pink, redundant ones gray, unspecified ones uncolored"""

    TEXT = "{ 'learning_rate': 1e-05,\n  'max_length': 2048,\n  'seed': 42}"

    def test_three_states(self):
        user_args = UserSpecifiedArgs(
            {"learning_rate", "max_length"}, {"learning_rate"}
        )
        lines = colorize_user_args(self.TEXT, user_args).split("\n")

        assert lines[0] == f"{GRAY}{{ 'learning_rate': 1e-05,\033[0m"
        assert lines[1] == f"{PINK}  'max_length': 2048,\033[0m"
        assert lines[2] == "  'seed': 42}"

    def test_explicit_redundant_args_argument(self):
        result = colorize_user_args(
            self.TEXT, {"learning_rate", "max_length"}, redundant_args={"learning_rate"}
        )
        lines = result.split("\n")
        assert lines[0].startswith(GRAY)
        assert lines[1].startswith(PINK)

    def test_plain_set_keeps_everything_pink(self):
        result = colorize_user_args(self.TEXT, {"learning_rate", "max_length"})
        lines = result.split("\n")
        assert lines[0].startswith(PINK)
        assert lines[1].startswith(PINK)
        assert GRAY not in result

    def test_format_args_with_colors_three_states(self):
        user_args = UserSpecifiedArgs(
            {"learning_rate", "max_length"}, {"learning_rate"}
        )
        result = format_args_with_colors(
            {
                "learning_rate": 1e-05,
                "max_length": 2048,
                "seed": 42,
                "run_name": "a-name-long-enough-to-make-pprint-put-one-key-per-line",
            },
            user_args,
        )
        lines = result.split("\n")
        assert lines[0].startswith(GRAY)
        assert lines[1].startswith(PINK)
        assert "\033[" not in lines[2]
        assert "\033[" not in lines[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
