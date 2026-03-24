from pathlib import Path


README_TEXT = Path("README.md").read_text(encoding="utf-8")


def test_readme_mentions_cli_usage() -> None:
    assert "video-smoke-detection process" in README_TEXT


def test_readme_mentions_generated_output_policy() -> None:
    assert "generated outputs are ignored" in README_TEXT.lower()
