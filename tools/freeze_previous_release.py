"""Keep the evaluation's headline numbers on the previous HMD release, for the article to give beside the new.

    uv run python tools/freeze_previous_release.py            # writes config/release-20260218.json

Decision HMD-Oct: the article states that the forms and the evaluation design were chosen on the HMD
release of 18 Feb 2026 and gives the headline shares on both releases.  The numbers on that release
are the companion's results at the commit that merged it (``SOURCE_REF``), all rendered on that
release.  This copies the families of numbers the headline draws on (the shares of first places,
the mean ranks and positions, the leave-one-origin-out shares) into a data file with its source and
the SHA-256 of the source, once; chapter 12 reads the file and reports each number as ``evFeb…``.
The file is data, like ``config/events.csv``: rerunning this tool reproduces it from git.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from mdmx import paths

SOURCE_REF = "c9de593"                     # T5h merged: the companion, every chapter on the release below
RELEASE = "20260218"
PREFIXES = ("evFirst", "evLoo", "evRank", "evPlace")
OUT = paths.root() / "config" / f"release-{RELEASE}.json"


def main() -> int:
    raw = subprocess.run(["git", "show", f"{SOURCE_REF}:companion/results/numbers.json"], cwd=paths.root(),
                         capture_output=True, check=True).stdout
    numbers = json.loads(raw)
    manifest = json.loads((paths.root() / "raw" / "MANIFEST.json").read_text(encoding="utf-8"))
    release = manifest["hmd"][RELEASE]
    keep = {k: {"value": v["value"], "text": v["text"], "desc": v["desc"]}
            for k, v in sorted(numbers.items()) if k.startswith(PREFIXES)}
    if not keep:
        raise SystemExit("no numbers selected: check SOURCE_REF and PREFIXES")
    out = {
        "about": "Headline numbers of the forecast evaluation on the HMD release of "
                 f"{release['last_modified_stamp'].split(';')[0]}, kept for the article (decision HMD-Oct). "
                 "Written by tools/freeze_previous_release.py; do not edit.",
        "release": {"id": RELEASE, "title": release["title"], "zip_sha256": release["zip_sha256"]},
        "source": {"ref": SOURCE_REF, "file": "companion/results/numbers.json",
                   "sha256": hashlib.sha256(raw).hexdigest()},
        "numbers": keep,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(paths.root())}: {len(keep)} numbers from {SOURCE_REF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
