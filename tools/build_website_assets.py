"""Export the examples the project page shows, from the paper's editing run.

Reads the run tree ``data/results/c4_run2`` (fetch it with
``python tools/fetch_from_hf.py autopolish``) and writes, under ``website/static/``:

  * ``rounds/<image_id>/round_<k>.jpg`` and ``rounds.json`` — every round of the
    AutoPolish loop on one image: the best image so far, its held-out score and
    similarity to the original, the instruction tried, and each persona's score
    and complaint;
  * ``gallery/<image_id>/<critic>.jpg`` and ``gallery.json`` — the source and each
    critic's final best image, with scores and a one-line note per image.

Only EVA images are exported, because EVA is released under CC0.

Usage:
    python tools/build_website_assets.py
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "data" / "results" / "c4_run2"
OUT = ROOT / "website" / "static"
CRITICS = ["static", "blind", "society", "reward_only"]
MAX_SIDE = 960

ROUNDS_IMAGE = "eva__18116"
GALLERY = {  # image id -> what the AutoPolish edit does, as in the paper's figures
    "eva__365027": "A player and a court turn a lone ball into a scene, which no other critic tries.",
    "eva__18116": "The green bench is turned bright red, giving the gray wall a clear point of focus.",
    "eva__489134": "A bright Milky Way and many stars fill the empty sky, making it the subject of the photo.",
    "eva__742534": "The steel ovens behind the chef become a bright kitchen, giving the photo a warmer setting.",
    "eva__444303": "A truck is added to the empty snowy courtyard, giving the scene scale and a point of interest.",
    "eva__696581": "A brighter, textured road adds detail below the sunset.",
    "eva__698274": "The rabbit is made larger and sharper, so it stands out as the subject.",
    "eva__449": "The black night sky becomes a pale background, so the tower's structure is easier to see.",
}


def load_parts(critic: str) -> dict[str, list[dict]]:
    """Per-round records of one critic, grouped by image and sorted by round."""
    by_image: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(str(RUN / "logs" / f"c4_{critic}" / f"c4_{critic}*.part-*.json"))):
        shard = re.search(r"shard(\d+)of", p).group(1)
        for r in json.load(open(p, encoding="utf-8")):
            r["_shard"] = shard
            by_image.setdefault(r["image_id"], []).append(r)
    for rows in by_image.values():
        rows.sort(key=lambda r: r["step"])
    return by_image


def panel_for(critic: str, shard: str) -> list[str]:
    summary = RUN / "logs" / f"c4_{critic}" / f"c4_{critic}.shard{shard}of4.json"
    return json.load(open(summary, encoding="utf-8")).get("panel", [])


def as_list(v) -> list:
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v) if isinstance(v, str) and v else []
    except (ValueError, SyntaxError):
        return []


def image_file(critic: str, image_id: str, best_path: str) -> Path:
    """Local file for a logged best path (logs carry the path of the machine that ran them)."""
    name = os.path.basename(best_path)
    if name.startswith("step0"):
        # the source is not copied into the run; it is the dataset original
        ds, iid = image_id.split("__", 1)
        return ROOT / "data" / ds / "images" / (iid if Path(iid).suffix else f"{iid}.jpg")
    return RUN / "edits" / critic / image_id / name


def save_jpg(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    im.save(dest, "JPEG", quality=84, optimize=True, progressive=True)
    return str(dest.relative_to(OUT))


def persona_tag(card: str) -> dict:
    """Short label for a persona card, e.g. 'Woman, 26-29' with art and photo experience."""
    m = re.match(r"a (male|female) aged ([\d-]+\+?)", card)
    sex = {"male": "Man", "female": "Woman"}.get(m.group(1), "Viewer") if m else "Viewer"
    age = m.group(2) if m else ""
    art = re.search(r"art experience: (\w+)", card)
    photo = re.search(r"photography experience: (\w+)", card)
    return {"who": f"{sex}, {age}".strip(", "),
            "art": art.group(1) if art else "", "photo": photo.group(1) if photo else "",
            "card": card}


def num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 3)


def build_rounds(society: dict[str, list[dict]]) -> dict:
    rows = society[ROUNDS_IMAGE]
    panel = [persona_tag(c) for c in panel_for("society", rows[0]["_shard"])]
    out_rounds, seen = [], {}
    for r in rows:
        f = image_file("society", ROUNDS_IMAGE, r["best_path"])
        if f not in seen:
            seen[f] = save_jpg(f, OUT / "rounds" / ROUNDS_IMAGE / f"round_{r['step']:02d}.jpg")
        scores, complaints = as_list(r["panel_scores"]), as_list(r["complaints"])
        out_rounds.append({
            "round": r["step"], "image": seen[f], "score": num(r["best_obj"]),
            "similarity": num(r["drift_of_best"]),
            "committed": r["step"] == 0 or not (r.get("committed_candidate") is None
                                                or (isinstance(r["committed_candidate"], float)
                                                    and math.isnan(r["committed_candidate"]))),
            "instruction": r.get("instruction") or "",
            "reactions": [{"persona": i, "score": num(s), "complaint": c}
                          for i, (s, c) in enumerate(zip(scores, complaints))]
                         if len(scores) == len(complaints) == len(panel) else [],
        })
    return {"image_id": ROUNDS_IMAGE, "panel": panel, "rounds": out_rounds}


def build_gallery(logs: dict[str, dict[str, list[dict]]]) -> list[dict]:
    items = []
    for image_id, note in GALLERY.items():
        if not all(image_id in logs[c] for c in CRITICS):
            print(f"skip {image_id}: missing from a critic's log")
            continue
        first = logs["society"][image_id][0]
        entry = {"image_id": image_id, "note": note,
                 "source": {"image": save_jpg(image_file("society", image_id, first["best_path"]),
                                              OUT / "gallery" / image_id / "source.jpg"),
                            "score": num(first["best_obj"])},
                 "critics": {}}
        for c in CRITICS:
            last = logs[c][image_id][-1]
            entry["critics"][c] = {
                "image": save_jpg(image_file(c, image_id, last["best_path"]),
                                  OUT / "gallery" / image_id / f"{c}.jpg"),
                "score": num(last["best_obj"]), "similarity": num(last["drift_of_best"])}
        items.append(entry)
    return items


def main() -> None:
    if not RUN.exists():
        raise SystemExit(f"{RUN} not found. Run: python tools/fetch_from_hf.py autopolish")
    logs = {c: load_parts(c) for c in CRITICS}
    rounds = build_rounds(logs["society"])
    gallery = build_gallery(logs)
    (OUT / "rounds.json").write_text(json.dumps(rounds, indent=1, ensure_ascii=False))
    (OUT / "gallery.json").write_text(json.dumps(gallery, indent=1, ensure_ascii=False))
    print(f"rounds: {len(rounds['rounds'])} rounds of {ROUNDS_IMAGE}; gallery: {len(gallery)} images")


if __name__ == "__main__":
    main()
