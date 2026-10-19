from pathlib import Path

from capacity_planner.report import render_readme

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_generated_sections_are_up_to_date():
    text = README.read_text()
    assert render_readme(text) == text, (
        "README tables are stale: run `python -m capacity_planner.report --update-readme`"
    )
