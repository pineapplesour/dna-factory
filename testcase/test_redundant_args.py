"""
Test cases for telling overriding arguments from redundant ones in `cli_main`.

A user-specified argument (a `--config` YAML key or a CLI arg) whose parsed value
equals the default is redundant: it is still user-specified, but it is reported in
`user_specified_args.redundant` and left out of the `output_dir: auto` name.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import the entry points
sys.path.insert(0, str(Path(__file__).parent.parent))

import sft
from dna_factory import training_runner
from dna_factory.utils.output_dir_generator import generate_auto_output_dir

CPU_ARGS = ["--use_cpu", "true", "--bf16", "false"]
CPU_NAMES = {"use_cpu", "bf16"}


@pytest.fixture(autouse=True)
def _cpu_only_parsing(monkeypatch):
    """Parse from the repo root, skip the CUDA tf32 probe, and capture the parsed args."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        "transformers.training_args.is_torch_tf32_available", lambda: False
    )
    monkeypatch.setattr(training_runner, "run_training", lambda *args: args)


def parse(tmp_path, yaml_text=None, cli=()):
    argv = [*cli, *CPU_ARGS]
    if yaml_text is not None:
        config = tmp_path / "user.yaml"
        config.write_text(yaml_text)
        argv = ["--config", str(config), *argv]
    return training_runner.cli_main(sft.SPEC, argv)


def auto_name(parsed):
    _, script, training, model, mixture, dna, user_specified = parsed
    return generate_auto_output_dir(
        model.model_name_or_path,
        user_specified,
        script,
        training,
        model,
        mixture,
        dna,
        "SFT",
    )


def test_yaml_three_states(tmp_path):
    # _defaults-SFT.yaml: learning_rate 2e-5, max_length 1024, packing true
    parsed = parse(tmp_path, "learning_rate: 2.0e-5\nmax_length: 2048\npacking: True\n")
    user_specified = parsed[-1]

    assert {"learning_rate", "max_length", "packing"} <= user_specified
    assert "num_train_epochs" not in user_specified
    assert user_specified.redundant - CPU_NAMES == {"learning_rate", "packing"}
    assert "max_length" not in user_specified.redundant
    assert ".max-2048" in auto_name(parsed)
    assert "lr-" not in auto_name(parsed)
    assert "pack-" not in auto_name(parsed)


def test_cli_three_states(tmp_path):
    parsed = parse(tmp_path, cli=["--learning_rate=0.00002", "--max_length", "2048"])
    user_specified = parsed[-1]

    assert user_specified.redundant - CPU_NAMES == {"learning_rate"}
    assert ".max-2048" in auto_name(parsed)
    assert "lr-" not in auto_name(parsed)


def test_overriding_args_keep_their_name(tmp_path):
    parsed = parse(tmp_path, "learning_rate: 1.0e-5\n", cli=["--max_length", "2048"])

    assert parsed[-1].redundant - CPU_NAMES == set()
    assert ".max-2048" in auto_name(parsed)
    assert ".lr-1e-05" in auto_name(parsed)


def test_cli_overriding_yaml_back_to_the_default_is_redundant(tmp_path):
    parsed = parse(tmp_path, "learning_rate: 1.0e-5\n", cli=["--learning_rate", "2e-5"])

    assert "learning_rate" in parsed[-1].redundant
    assert "lr-" not in auto_name(parsed)


def test_cli_overriding_yaml_away_from_the_default_is_not_redundant(tmp_path):
    parsed = parse(tmp_path, "learning_rate: 2.0e-5\n", cli=["--learning_rate", "1e-5"])

    assert "learning_rate" not in parsed[-1].redundant
    assert ".lr-1e-05" in auto_name(parsed)


def test_dataclass_default_and_nested_values(tmp_path):
    # `weight_decay` is not in _defaults-SFT.yaml: its default is the dataclass default.
    # `lr_scheduler_kwargs` and `report_to` are nested/list values of the defaults YAML.
    parsed = parse(
        tmp_path,
        "weight_decay: 0\n"
        "lr_scheduler_kwargs:\n  min_lr_rate: 0.1\n"
        "report_to:\n  - none\n"
        "output_dir: auto\n",
    )
    redundant = parsed[-1].redundant

    assert {"weight_decay", "lr_scheduler_kwargs", "output_dir"} <= redundant
    assert "report_to" not in redundant


def test_no_user_args_means_nothing_is_redundant(monkeypatch):
    # The baseline parse is skipped entirely when nothing was specified.
    monkeypatch.setattr(
        training_runner,
        "_parse_default_args",
        lambda *args: pytest.fail("baseline parse should be skipped"),
    )
    user_specified = training_runner.cli_main(sft.SPEC, [])[-1]
    assert user_specified == set()


def test_unparseable_defaults_fall_back_to_all_pink(tmp_path, monkeypatch):
    monkeypatch.setattr(training_runner, "_parse_default_args", lambda *args: None)
    parsed = parse(tmp_path, "learning_rate: 2.0e-5\n")

    assert "learning_rate" in parsed[-1]
    assert not getattr(parsed[-1], "redundant", ())
    assert ".lr-2e-05" in auto_name(parsed)
