"""Run in pre-commit/CI in order to keep the README feature table up to date."""

import pathlib
import re
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tubular.base import FEATURE_REGISTRY


def get_feature_table(feature_dict: dict[str, dict[str, bool]]) -> str:
    r"""Process provided feature_dict into markdown table for README.

    Parameters
    ----------
    feature_dict: dict[str, dict[str, bool]]
        nested dictionary containing info on features per class

    Returns
    -------
        str: markdown table

    Examples
    --------
    ```pycon
    >>> feature_dict = {
    ...     "class1": {"feature1": True, "feature2": False},
    ...     "class2": {"feature1": False, "feature2": False},
    ... }

    >>> print(get_feature_table(feature_dict))
    |        | feature1           | feature2   |
    |--------|--------------------|------------|
    | class1 | :heavy_check_mark: | :x:        |
    | class2 | :x:                | :x:        |

    ```

    """
    df = pd.DataFrame.from_dict(feature_dict, orient="index").sort_index()
    # replace bools with unicode tick/cross
    df = df.replace({True: ":heavy_check_mark:", False: ":x:"})

    return df.to_markdown(tablefmt="github").strip()


def insert_table_to_readme(table: str, readme_text: str) -> None:
    r"""Insert markdown table into repo README file.

    Returns
    -------
        str: updated markdown text

    Examples
    --------
    ```pycon
    >>> readme_text = (
    ...     "this is a fake readme\n"
    ...     "it contains info on stuff\n"
    ...     "<!-- AUTO-GENERATED feature table -->\n"
    ...     "placeholder\n"
    ...     "<!-- /AUTO-GENERATED feature table -->\n"
    ...     "and a conclusion"
    ... )

    >>> table = (
    ...     "|        | feature1           | feature2   |\n"
    ...     "|--------|--------------------|------------|\n"
    ...     "| class1 | :heavy_check_mark: | :x:        |\n"
    ...     "| class2 | :x:                | :x:        |"
    ... )

    >>> print(insert_table_to_readme(table, readme_text))
    this is a fake readme
    it contains info on stuff
    <!-- AUTO-GENERATED feature table -->
    |        | feature1           | feature2   |
    |--------|--------------------|------------|
    | class1 | :heavy_check_mark: | :x:        |
    | class2 | :x:                | :x:        |
    <!-- /AUTO-GENERATED feature table -->
    and a conclusion

    ```

    """
    START = "<!-- AUTO-GENERATED feature table -->"
    END = "<!-- /AUTO-GENERATED feature table -->"
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", flags=re.DOTALL)
    replacement = f"{START}\n{table}\n{END}"

    return pattern.sub(replacement, readme_text)


if __name__ == "__main__":
    table = get_feature_table(feature_dict=FEATURE_REGISTRY)

    readme_path = pathlib.Path("./README.md")

    readme_text = readme_path.read_text(encoding="utf8")

    updated_readme_text = insert_table_to_readme(table, readme_text)

    if readme_text != updated_readme_text:
        readme_path.write_text(updated_readme_text, encoding="utf8")
