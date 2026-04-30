"""Flatten the manuscript for journal resubmission.

Resolves \\input{...} calls in reports/main.tex against reports/<path>(.tex)
and rewrites \\includegraphics{figures/<sub>/<name>.<ext>} to
\\includegraphics{<name>.<ext>} (bare filename) so figures live alongside
the .tex in the output bundle.

Usage:
    python scripts/flatten_manuscript.py
    python scripts/flatten_manuscript.py --src reports/main.tex \\
        --out "reports/R&R round 1/Submission_R1/main.tex"

Single pass: confirmed there are no nested \\input{} calls inside any of the
inputted files (see the Phase 1 exploration in plans/the-manuscript-is-ready-quirky-quail.md).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")


def expand_inputs(src: Path) -> tuple[str, int, int]:
    """Read src, expand \\input{} relative to src.parent, strip figure paths.

    Returns (output_text, n_inputs_expanded, n_figures_flattened).
    """
    base = src.parent
    text = src.read_text()
    n_inputs = 0
    n_figures = 0

    def expand_input(match: re.Match[str]) -> str:
        nonlocal n_inputs
        target = match.group(1).strip()
        candidates = [base / target]
        if not target.endswith(".tex"):
            candidates.append(base / f"{target}.tex")
        for cand in candidates:
            if cand.is_file():
                n_inputs += 1
                content = cand.read_text().rstrip("\n")
                # Mark inlined block for traceability.
                return f"% --- begin: inlined {target} ---\n{content}\n% --- end: inlined {target} ---"
        raise FileNotFoundError(f"\\input target not found: {target} (looked in {candidates})")

    text = INPUT_RE.sub(expand_input, text)

    def strip_graphics_path(match: re.Match[str]) -> str:
        nonlocal n_figures
        prefix, path, suffix = match.group(1), match.group(2), match.group(3)
        bare = Path(path).name
        if bare != path:
            n_figures += 1
        return f"{prefix}{bare}{suffix}"

    text = GRAPHICS_RE.sub(strip_graphics_path, text)
    return text, n_inputs, n_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("reports/main.tex"),
        help="Source manuscript with \\input{} calls (default: reports/main.tex).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/R&R round 1/Submission_R1/main.tex"),
        help="Flattened output path (default: reports/R&R round 1/Submission_R1/main.tex).",
    )
    args = parser.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"Source not found: {args.src}")

    output, n_inputs, n_figures = expand_inputs(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    print(f"Wrote {args.out} ({len(output):,} chars).")
    print(f"Expanded {n_inputs} \\input{{}} call(s); flattened {n_figures} figure path(s).")


if __name__ == "__main__":
    main()
