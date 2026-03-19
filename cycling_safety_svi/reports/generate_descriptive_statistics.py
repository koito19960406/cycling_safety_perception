"""
Generate descriptive statistics table for the manuscript.
Outputs: reports/models/descriptive_statistics.tex

Usage:
    python cycling_safety_svi/reports/generate_descriptive_statistics.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJ_ROOT / "data" / "raw" / "database_2024_10_07_135133.db"
CHOICE_DATA_PATH = PROJ_ROOT / "data" / "processed" / "choice_data_with_demographics.csv"
OUTPUT_PATH = PROJ_ROOT / "reports" / "models" / "descriptive_statistics.tex"


def get_analysis_respondent_ids():
    """Get respondent IDs that appear in the analysis dataset."""
    choice = pd.read_csv(CHOICE_DATA_PATH)
    return choice["respondent_id"].unique()


def load_demographics(valid_ids):
    """Load demographics from SQLite, filtered to analysis respondents."""
    conn = sqlite3.connect(str(DB_PATH))
    resp = pd.read_sql("SELECT * FROM Response", conn)
    conn.close()

    resp = resp[resp["respondent_id"].isin(valid_ids)]
    resp = resp.dropna(
        subset=["age", "gender", "household_size", "income", "transportation", "cycler"]
    )
    return resp


def compute_distributions(resp):
    """Compute frequency distributions for key demographic variables."""
    n = len(resp)
    results = {}

    variables = {
        "age": {
            "label": "Age",
            "mapping": {1: "18--30", 2: "31--45", 3: "46--60", 4: "61--70", 5: "71+"},
        },
        "gender": {
            "label": "Gender",
            "mapping": {1: "Male", 2: "Female", 3: "Other"},
        },
        "household_size": {
            "label": "Household size",
            "mapping": {
                1: "1 person",
                2: "2 people",
                3: "3 people",
                4: "4 people",
                5: "5 people",
                6: "6+ people",
            },
        },
        "transportation": {
            "label": "Primary transport",
            "mapping": {
                2: "Bicycle",
                4: "Car",
                1: "Walking",
                3: "Public transport",
                5: "Other",
            },
        },
        "cycler": {
            "label": "Cycling frequency",
            "mapping": {
                7: "5+ days/week",
                5: "3 days/week",
                4: "2 days/week",
                6: "4 days/week",
                3: "1 day/week",
                2: "$<$1 day/week",
                1: "Never",
            },
        },
    }

    # Education: group into broader categories
    edu_groups = {
        "Primary/secondary": [2, 3, 4],
        "Vocational": [5, 6, 7, 8],
        "University (B.Sc./B.A.)": [9],
        "Postgraduate": [10, 11, 12],
        "Other/prefer not to say": [13, 14],
    }

    for col, info in variables.items():
        vc = resp[col].value_counts()
        rows = []
        for code, label in info["mapping"].items():
            cnt = vc.get(code, 0)
            rows.append({"category": label, "n": int(cnt), "pct": cnt / n * 100})
        results[info["label"]] = rows

    # Education grouped
    edu_rows = []
    for group_label, codes in edu_groups.items():
        cnt = resp["education"].isin(codes).sum()
        edu_rows.append({"category": group_label, "n": int(cnt), "pct": cnt / n * 100})
    results["Education (grouped)"] = edu_rows

    return results, n


def generate_latex(results, n):
    """Generate LaTeX table string."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Descriptive statistics of the survey sample ($n = {n}$).}}",
        r"\label{tab:descriptive_stats}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Variable & Category & $n$ & \% \\",
        r"\midrule",
    ]

    for i, (var_label, rows) in enumerate(results.items()):
        nrows = len(rows)
        for j, row in enumerate(rows):
            if j == 0:
                prefix = rf"\multirow{{{nrows}}}{{*}}{{{var_label}}}"
            else:
                prefix = ""
            lines.append(
                f"{prefix} & {row['category']} & {row['n']} & {row['pct']:.1f} \\\\"
            )
        if i < len(results) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main():
    valid_ids = get_analysis_respondent_ids()
    print(f"Respondents in analysis: {len(valid_ids)}")

    resp = load_demographics(valid_ids)
    print(f"Matched with valid demographics: {len(resp)}")

    results, n = compute_distributions(resp)

    # Print summary
    for var_label, rows in results.items():
        print(f"\n{var_label}:")
        for row in rows:
            print(f"  {row['category']}: {row['n']} ({row['pct']:.1f}%)")

    # Write LaTeX
    latex = generate_latex(results, n)
    OUTPUT_PATH.write_text(latex + "\n")
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
