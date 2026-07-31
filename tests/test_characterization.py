"""Characterization tests for dataset_tools.sh.

These tests verify the documented public behavior of the existing
dataset_tools.sh without modifying it. They characterize:

- Available commands and their aliases
- Default behaviors (no-reencode, copy mode)
- Trim range computation expectations
- Merge validation rules
- Script existence and structure
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lerobot_dataset_editor.characterization import DATASET_TOOLS_SH, get_characterization

# Skip all tests in this module if the script doesn't exist
pytestmark = pytest.mark.skipif(
    not DATASET_TOOLS_SH.exists(),
    reason="dataset_tools.sh not found at expected location",
)


class TestDatasetToolsCommands:
    """Characterize available commands."""

    def test_script_exists(self):
        """dataset_tools.sh exists at expected location."""
        assert DATASET_TOOLS_SH.exists()
        assert DATASET_TOOLS_SH.stat().st_mode & 0o111  # executable

    def test_help_command(self):
        """help command prints usage."""
        result = subprocess.run(
            [str(DATASET_TOOLS_SH), "help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "trim" in result.stdout
        assert "merge" in result.stdout

    def test_no_args_shows_usage(self):
        """No arguments shows usage and exits with code 2."""
        result = subprocess.run(
            [str(DATASET_TOOLS_SH)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 2

    def test_unknown_command_exits_2(self):
        """Unknown command exits with code 2."""
        result = subprocess.run(
            [str(DATASET_TOOLS_SH), "nonexistent"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 2
        assert "Unknown command" in result.stderr


class TestDatasetToolsAliases:
    """Characterize command aliases."""

    def test_trim_stationary_alias_hyphen(self):
        """trim-stationary is an alias for trim."""
        char = get_characterization()
        assert char.aliases["trim-stationary"] == "trim"

    def test_trim_stationary_alias_underscore(self):
        """trim_stationary is an alias for trim."""
        char = get_characterization()
        assert char.aliases["trim_stationary"] == "trim"

    def test_aliases_resolve_in_script(self):
        """Verify aliases are actually in the case statement by reading the script."""
        content = DATASET_TOOLS_SH.read_text()
        assert "trim-stationary" in content
        assert "trim_stationary" in content


class TestDatasetToolsDefaults:
    """Characterize default behaviors."""

    def test_default_no_reencode(self):
        """Both trim and merge default to no re-encode."""
        char = get_characterization()
        assert char.trim_no_reencode is True
        assert char.merge_no_reencode is True

    def test_default_copy_mode(self):
        """Trim defaults to video copy mode."""
        char = get_characterization()
        assert char.defaults["trim_video_copy_mode"] == "copy"

    def test_default_merge_remux_policy(self):
        """Merge defaults to never remux."""
        char = get_characterization()
        assert char.defaults["merge_remux_policy"] == "never"

    def test_script_documents_no_reencode(self):
        """The script documents that trim doesn't re-encode."""
        content = DATASET_TOOLS_SH.read_text()
        assert "does not decode" in content or "no-reencode" in content or "not decode" in content

    def test_script_documents_copy_videos(self):
        """The script documents that merge copies videos."""
        content = DATASET_TOOLS_SH.read_text()
        assert "copied" in content or "copy" in content.lower()


class TestDatasetToolsTrimBehavior:
    """Characterize trim range computation."""

    def test_trim_range_description(self):
        """Verify trim range computation is documented."""
        char = get_characterization()
        assert "epsilon" in char.trim_range_computation
        assert "stationary" in char.trim_range_computation.lower()

    def test_trim_dispatches_to_python(self):
        """Verify trim dispatches to trim_stationary_dataset.py."""
        content = DATASET_TOOLS_SH.read_text()
        assert "trim_stationary_dataset.py" in content


class TestDatasetToolsMergeBehavior:
    """Characterize merge validation."""

    def test_merge_validation_rules(self):
        """Verify documented merge validation rules."""
        char = get_characterization()
        rules = char.merge_validation_rules
        assert any("fps" in r for r in rules)
        assert any("robot_type" in r for r in rules)
        assert any("feature" in r for r in rules)
        assert any("two" in r.lower() for r in rules)

    def test_merge_dispatches_to_python(self):
        """Verify merge dispatches to merge_datasets.py."""
        content = DATASET_TOOLS_SH.read_text()
        assert "merge_datasets.py" in content


class TestDatasetToolsReport:
    """Test that characterization generates a valid report."""

    def test_characterization_report_structure(self):
        """Verify report has all required fields."""
        char = get_characterization()
        report = char.to_report()
        assert "commands" in report
        assert "aliases" in report
        assert "defaults" in report
        assert "trim_no_reencode" in report
        assert "merge_no_reencode" in report
        assert "script_exists" in report
        assert report["script_exists"] is True
