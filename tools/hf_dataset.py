"""Registry of the Hugging Face dataset repos that ``fetch_from_hf.py`` restores.

Only the experiment outputs are hosted here. The source datasets (PARA, EVA,
LAPIS, Rapidata) are distributed by their authors under their own terms; see
docs/DATASETS.md for where to download them and how to lay them out.

An output tree is stored verbatim, file for file, with its directory layout
intact, plus a ``layout_manifest.json`` recording where each part belongs, so
``fetch_from_hf.py`` can rebuild the local tree exactly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this tools/ directory; datasets live under data/.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

HF_OWNER = os.getenv("HF_OWNER", "savoji").strip()

# Small JSON manifest written to each repo so `fetch` can rebuild data/<name>/.
MANIFEST_NAME = "layout_manifest.json"

# Verbatim files (CSVs, readmes) are stored under this repo prefix so they never
# collide with the auto-generated dataset card (README.md) or the parquet dirs.
RAW_PREFIX = "raw"


class DatasetSpec:
    """How one local tree maps to its Hub repo.

    Two shapes, both carrying a ``layout_manifest.json`` so
    ``fetch_from_hf.py`` can rebuild the local tree exactly:

    * **corpus** (``images_dir`` set) — a source dataset. Its images become a
      ``datasets`` ``images`` config (parquet with the original bytes embedded,
      browsable in the Hub viewer); everything else is stored verbatim. The
      parquet detour exists to dodge the Hub's 10k-files-per-directory and
      128-commits-per-hour limits, which flat image corpora blow through.
    * **verbatim** (``images_dir=None``) — an output tree such as the AutoPolish
      results, uploaded file-for-file with its directory layout intact. Those
      limits do not bind here (deepest directory holds ~100 files), and the
      layout *is* the artifact: the analysis scripts read it by path via
      ``--output-root``. Uploading from disk also avoids staging a parquet copy
      of every image locally.

    ``sources`` maps repo prefixes to project-relative local paths, so one repo
    can gather several trees (results/ and data/results/ for AutoPolish).
    """

    def __init__(self, name: str, images_dir: str | None = None,
                 sources: list[tuple[str, str]] | None = None,
                 description: str = ""):
        self.name = name
        self.repo_id = f"{HF_OWNER}/{name.upper()}"
        self.images_dir = images_dir
        # default: the single data/<name>/ tree, mounted at the repo root
        self.sources = sources or [("", f"data/{name}")]
        self.description = description

    @property
    def verbatim_only(self) -> bool:
        """True when there is no parquet images config, just the file tree."""
        return self.images_dir is None

    @property
    def local_dir(self) -> Path:
        """Primary local tree (the only one, for single-source datasets)."""
        return ROOT / self.sources[0][1]

    def roots(self) -> list[tuple[str, Path]]:
        """(repo prefix, absolute local path) for every tree in this dataset."""
        return [(prefix, ROOT / rel) for prefix, rel in self.sources]


DATASETS: dict[str, DatasetSpec] = {
    # Experiment outputs: cached analysis JSONs and figures, plus the raw
    # per-run logs and edited images the analysis scripts are pointed at.
    "autopolish": DatasetSpec(
        "autopolish",
        sources=[("analysis", "results"), ("runs", "data/results")],
        description=(
            "AutoPolish experiment outputs: cached analysis JSONs and figures, "
            "plus raw per-run judge logs and edited images."
        ),
    ),
}


def get_token() -> str | None:
    """The Hugging Face token from .env, or None (public repos need none)."""
    return os.getenv("HF_TOKEN", "").strip() or None


def resolve(name: str) -> DatasetSpec:
    """Return the DatasetSpec for a name, or exit with a clear error."""
    key = name.lower()
    if key not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)}.")
    return DATASETS[key]


def selected(names: list[str]) -> list[str]:
    """Expand CLI args into concrete dataset names; 'all' means every known one."""
    if names == ["all"]:
        return list(DATASETS)
    return [n.lower() for n in names]
