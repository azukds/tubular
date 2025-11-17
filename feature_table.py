"""Script to run in pre-commit/CI in order to keep the README feature table up to date."""

import pathlib
import re

import pandas as pd

from tubular.base import FEATURE_REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]


def get_feature_table() -> str:
    """Process FEATURE_REGISTRY into markdown table to README.

    Returns
    -------
        str: markdown table

    """
    df = pd.DataFrame.from_dict(FEATURE_REGISTRY, orient="index").sort_index()
    # replace bools with unicode tick/cross
    df = df.replace({True: ":heavy_check_mark:", False: ":x:"})

    return df.to_markdown(tablefmt="github").strip()


def insert_table_to_readme(table: str, readme_text: str) -> None:
    """Insert markdown table into repo README file.

    Returns
    -------
        str: updated markdown text

    """
    START = "<!-- AUTO-GENERATED feature table -->"
    END = "<!-- /AUTO-GENERATED feature table -->"
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", flags=re.DOTALL)
    replacement = f"{START}\n{table}\n{END}"

    return pattern.sub(replacement, readme_text)


if __name__ == "__main__":
    table = get_feature_table()

    readme_path = pathlib.Path("./README.md")

    readme_text = readme_path.read_text(encoding="utf8")

    updated_readme_text = insert_table_to_readme(table, readme_text)

    if readme_text != updated_readme_text:
        readme_path.write_text(updated_readme_text, encoding="utf8")
