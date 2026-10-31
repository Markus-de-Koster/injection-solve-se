"""Static equivalence of the vendored files with their source.

Every file listed in ``VENDORED.json`` was copied from the ``pgl`` source repository at one
commit and passed through the import rename below. This script reads each source file at
that commit (``git show <commit>:<path>`` in a local checkout of the source repository),
applies the rename, and compares the result with the file in this repository:

- ``renamed`` files must match exactly (after the rename);
- ``modified`` and ``partial`` files are listed with their unified diff, so every deviation
  from the source is visible; their manifest note says why they deviate.

Files under ``src/injection_hse`` and ``tests`` that the manifest does not list are reported
as well (a vendored file added without a manifest entry, or a file of this repository's
own).

usage::

    python tools/check_vendored.py --source <checkout of the source repository> [--diff]

Exit status 0 when every ``renamed`` file matches and every listed file exists, 1 otherwise.
Needs only the standard library and ``git``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "VENDORED.json"

#: the module ``pgl`` as a whole word (not inside ``pgl_x``, ``x/pgl`` or ``pgl-x``)
_MODULE = re.compile(r"(?<![\w/\-])pgl(?![\w/\-])")
#: repository paths that move with the rename (drivers and tests)
_PATHS = (("run/examples/pgl", "run/examples/injection_hse"), ("tests/pgl", "tests/injection_hse"))


def rename(text: str) -> str:
    """The import rename applied to every vendored file: the package ``pgl`` becomes
    ``injection_hse`` (imports, dotted references, quoted names) and the two repository
    paths of ``_PATHS`` follow it. Configuration files are vendored without a rename."""
    for old, new in _PATHS:
        text = text.replace(old, new)
    return _MODULE.sub("injection_hse", text)


def source_text(checkout: Path, commit: str, path: str) -> str:
    """The text of ``path`` at ``commit`` in the source checkout."""
    out = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    )
    return out.stdout.decode("utf-8")


def expected_text(checkout: Path, commit: str, entry: dict) -> str:
    """The source file as it reads after the rename (configuration files: unchanged)."""
    text = source_text(checkout, commit, entry["source"])
    return text if entry.get("rename", True) is False else rename(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source",
        required=True,
        type=Path,
        help="a git checkout of the source repository (any branch)",
    )
    ap.add_argument(
        "--diff",
        action="store_true",
        help="print the unified diff of every modified or partial file",
    )
    args = ap.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    commit = manifest["source"]["commit"]
    failures, deviations = [], []
    listed = set()
    for entry in manifest["files"]:
        listed.add(entry["path"])
        local = ROOT / entry["path"]
        if not local.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        want = expected_text(args.source, commit, entry)
        have = local.read_text()
        status = entry["status"]
        if status == "renamed":
            if have != want:
                failures.append(f"differs from its source: {entry['path']}")
            continue
        diff = list(
            difflib.unified_diff(
                want.splitlines(keepends=True),
                have.splitlines(keepends=True),
                fromfile=f"{entry['source']} (renamed)",
                tofile=entry["path"],
            )
        )
        added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        deviations.append((entry, added, removed, diff))
    unlisted = sorted(
        str(p.relative_to(ROOT))
        for base in ("src/injection_hse", "tests/injection_hse", "run/examples/injection_hse")
        for p in (ROOT / base).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(ROOT)) not in listed
    )
    n = len(manifest["files"])
    print(f"source {manifest['source']['repository']} @ {commit}")
    print(
        f"{n} vendored files: {n - len(deviations) - len(failures)} identical after the "
        f"rename, {len(deviations)} with deviations, {len(failures)} failing"
    )
    for entry, added, removed, diff in deviations:
        print(f"\n[{entry['status']}] {entry['path']}  (+{added} / -{removed} lines)")
        print(f"  why: {entry['note']}")
        if args.diff:
            sys.stdout.writelines("    " + d for d in diff)
    if unlisted:
        print("\nnot in the manifest (own files of this repository):")
        for p in unlisted:
            print(f"  {p}")
    for f in failures:
        print(f"FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
