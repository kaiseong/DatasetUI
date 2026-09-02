"""Tests for the CLI report command."""

from __future__ import annotations

import json

from lerobot_dataset_editor.cli import generate_report


class TestReportCommand:
    """Test the report generation."""

    def test_report_generates_successfully(self):
        """Report should generate without errors."""
        report = generate_report()
        assert report["operation"] == "report"
        assert report["status"] in ("ok", "error")

    def test_report_has_version(self):
        """Report should include package version."""
        report = generate_report()
        assert "dataset_editor_version" in report
        assert report["dataset_editor_version"] == "0.1.0"

    def test_report_has_timestamp(self):
        """Report should include a timestamp."""
        report = generate_report()
        assert "timestamp" in report

    def test_report_contracts_section(self):
        """Report should have contracts validation results."""
        report = generate_report()
        assert "contracts" in report
        contracts = report["contracts"]
        assert "v21_schema_valid" in contracts
        assert "v30_schema_valid" in contracts
        assert "language_v31_valid" in contracts
        assert "space_parity_commit" in contracts
        assert contracts["space_parity_commit"] == "d724744111cae6feb9a2194e607e71749813a97a"

    def test_report_fixtures_section(self):
        """Report should have fixtures status."""
        report = generate_report()
        assert "fixtures" in report
        fixtures = report["fixtures"]
        assert "v21_valid" in fixtures
        assert "v30_valid" in fixtures
        assert "v30_annotated" in fixtures
        assert "corrupt" in fixtures

    def test_report_dataset_tools_section(self):
        """Report should have dataset_tools characterization."""
        report = generate_report()
        assert "dataset_tools" in report
        dt = report["dataset_tools"]
        assert "commands" in dt
        assert "aliases" in dt
        assert "defaults" in dt
        assert "trim_no_reencode" in dt
        assert "merge_no_reencode" in dt

    def test_report_is_valid_json(self):
        """Report should be serializable to valid JSON."""
        report = generate_report()
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        assert parsed == report

    def test_report_status_ok_when_all_pass(self):
        """Report status should be 'ok' when all validations pass."""
        report = generate_report()
        # If schemas load and fixtures validate, status should be ok
        if (report["contracts"].get("v21_schema_valid") and
            report["contracts"].get("v30_schema_valid") and
            report["fixtures"].get("v21_valid") and
            report["fixtures"].get("v30_valid")):
            assert report["status"] == "ok"


def test_report_contains_executable_fixture_summaries():
    report = generate_report()
    v21 = report["fixtures"]["v21_valid"]
    v30 = report["fixtures"]["v30_valid"]
    annotated = report["fixtures"]["v30_annotated"]
    assert v21["valid"] is True
    assert v21["detected_version"] == "v2.1"
    assert v21["episodes"] == 2
    assert v21["frames"] == 20
    assert v30["valid"] is True
    assert v30["detected_version"] == "v3.0"
    assert v30["row_groups"] == 3
    assert annotated["valid"] is True
    assert set(annotated["annotation_styles"]) >= {
        "task_aug", "subtask", "plan", "memory", "interjection", "speech", "vqa"
    }
    assert set(annotated["vqa_answer_kinds"]) == {
        "bbox", "keypoint", "count", "attribute", "spatial"
    }
    assert report["fixtures"]["corrupt"]["rejected_as_expected"] is True


def test_report_contains_truthful_space_parity_progress():
    parity = generate_report()["space_parity"]
    assert parity["commit"] == "d724744111cae6feb9a2194e607e71749813a97a"
    assert parity["total"] == 37
    assert parity["planned"] == 0
    assert parity["implemented"] == parity["total"]
    assert parity["all_mapped"] is True


def test_report_verifies_characterized_sources_have_not_drifted():
    tools = generate_report()["dataset_tools"]
    assert tools["status"] == "current"
    assert tools["source_fingerprints_match"] is True


def test_report_conforms_to_rpc_schema():
    import jsonschema

    from lerobot_dataset_editor.schemas import rpc_schema

    jsonschema.Draft202012Validator(rpc_schema()).validate(generate_report())


def test_report_contains_verified_official_compatibility_evidence():
    report = generate_report()
    compatibility = report["official_compatibility"]
    assert compatibility["evidence_valid"] is True
    assert compatibility["baselines"] == {
        "v2.1": "lerobot==0.3.3",
        "v3.0": "lerobot==0.6.0",
    }
    assert compatibility["contract_fixtures"]["v21_valid"]["full_loader_passed"] is True
    assert compatibility["contract_fixtures"]["v30_valid"]["full_loader_passed"] is True
    assert compatibility["contract_fixtures"]["v30_annotated"]["full_loader_passed"] is True
    assert report["fixtures"]["v21_valid"]["official_loader_validated"] is True
    assert report["fixtures"]["v30_valid"]["official_loader_validated"] is True
    assert report["fixtures"]["v30_annotated"]["official_loader_validated"] is True
    assert report["fixtures"]["official_v21_v033"]["valid"] is True
    assert report["fixtures"]["official_v21_v033"]["global_stats_present"] is False
    assert report["fixtures"]["official_v30_v060"]["valid"] is True
