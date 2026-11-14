import dataclasses
import os
import pprint

import yaml


class UserSpecifiedArgs(set):
    """Names of the arguments the user specified (`--config` YAML keys and CLI args).

    `redundant` is the subset whose final parsed value equals the default, i.e. the
    run is identical to one that omits them. It rides along on the set so the entry
    points can keep passing a single `user_specified_args` object around.
    """

    def __init__(self, names=(), redundant=()):
        super().__init__(names)
        self.redundant = frozenset(redundant) & self


def _normalize(value):
    """Reduce a parsed value to plain containers so `==` is a deep comparison."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _normalize(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def values_equal(value, default):
    """Deep equality of two parsed values (so 1 == 1.0, and lists/dicts compare by content)."""
    try:
        return bool(_normalize(value) == _normalize(default))
    except Exception:  # noqa: BLE001 - cosmetic only; an odd value just counts as different
        return False


def find_redundant_args(user_specified_args, parsed_args, default_args):
    """Return the user-specified names whose parsed value equals the default.

    `parsed_args` and `default_args` are parallel sequences of argument objects (the
    parsed dataclasses): the former from the real run, the latter from the same entry
    point parsed with neither a user config nor CLI overrides, so "the default" covers
    both `configs/_defaults-*.yaml` and the dataclass defaults. Parsed values are
    compared, never raw YAML/CLI text, so `1e-5` vs `1.0e-5` or `true` vs `True` are
    equal. Names that are not an argument of any object (e.g. `config`) are never
    redundant.
    """
    redundant = set()
    for name in user_specified_args:
        pairs = [
            (vars(parsed)[name], vars(default).get(name, dataclasses.MISSING))
            for parsed, default in zip(parsed_args, default_args, strict=True)
            if name in vars(parsed)
        ]
        if pairs and all(
            default is not dataclasses.MISSING and values_equal(value, default)
            for value, default in pairs
        ):
            redundant.add(name)
    return redundant


def parse_user_args(args):
    """Parse command line arguments to identify user-specified options including config files"""
    user_specified_args = set()
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            # `--name=value` carries its value in the same token, `--name value` in the next one.
            name, inline, inline_value = arg[2:].partition("=")
            field_name = name.replace("-", "_")
            user_specified_args.add(field_name)

            if inline:
                value = inline_value
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 1
            else:
                value = None

            # Keys of a user-supplied config file count as user-specified too
            if field_name == "config" and value:
                user_specified_args.update(parse_yaml_config(value))
        elif arg.startswith("-") and len(arg) > 1:
            # Handle short options like -n, -v, etc.
            field_name = arg[1:]
            user_specified_args.add(field_name)
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
        i += 1
    return user_specified_args


def parse_yaml_config(config_file):
    """Parse YAML configuration file and return set of keys"""
    yaml_args = set()
    try:
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    # Recursively collect all keys from nested dictionaries
                    def collect_keys(data, prefix=""):
                        if isinstance(data, dict):
                            for key, value in data.items():
                                full_key = f"{prefix}_{key}" if prefix else key
                                yaml_args.add(full_key)
                                if isinstance(value, dict):
                                    collect_keys(value, full_key)

                    collect_keys(config_data)
    except Exception as e:  # noqa: BLE001 - cosmetic only; a bad config must not abort the run
        # If there's an error reading the config file, just continue without adding yaml args
        print(f"Warning: Could not parse config file {config_file}: {e}")

    return yaml_args


def colorize_user_args(text, user_specified_args, redundant_args=None):
    """Colorize entire lines containing user-specified arguments.

    Arguments that change the run are bold pink; those equal to the default
    (`redundant_args`, by default `user_specified_args.redundant`) are gray.
    """
    PINK = "\033[1;95m"  # Bold bright magenta/pink color
    GRAY = "\033[90m"  # Bright black: specified, but equal to the default
    RESET = "\033[0m"

    if redundant_args is None:
        redundant_args = getattr(user_specified_args, "redundant", ())
    redundant_args = set(redundant_args)
    overriding_args = set(user_specified_args) - redundant_args

    def contains_any(line, args):
        # Pattern for quoted field names or unquoted field names
        return any(
            f"'{arg}':" in line or (f"{arg}:" in line and f"'{arg}':" not in line)
            for arg in args
        )

    lines = text.split("\n")
    colored_lines = []

    for line in lines:
        # Color the entire line: pink wins when a line matches both kinds
        if contains_any(line, overriding_args):
            line = f"{PINK}{line}{RESET}"
        elif contains_any(line, redundant_args):
            line = f"{GRAY}{line}{RESET}"

        colored_lines.append(line)

    return "\n".join(colored_lines)


def format_args_with_colors(args_dict, user_specified_args, redundant_args=None):
    """Format arguments dictionary with colors for user-specified values"""
    formatted = pprint.pformat(args_dict, indent=2, width=80)
    return colorize_user_args(formatted, user_specified_args, redundant_args)
