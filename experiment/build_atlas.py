"""Build the static atlas and its shareable reply pages into an empty directory."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from shutil import copyfile

from .og_image import render_og
from .run import DIRECTIONS, parse_unit


ATLAS_DIR = Path(__file__).resolve().parents[1] / "atlas"
PUBLIC_ASSETS = ("index.html", "style.css", "app.js", "data.js", "share.css")
SITE_URL = "https://inkling.flowershop.dev"
SITE_NAME = "The Inkomplete Inkling Persona Atlas"
CATEGORY_NAMES = {
    "persona": "Persona",
    "changed": "Changed assistant",
    "off_topic": "Off-topic",
    "normal": "Normal",
    "incoherent": "Incoherent",
    "unlabeled": "Unlabeled",
}


def reply_slug(record: dict, direction: str) -> str:
    expert = record["expert"] if record["expert"] is not None else "none"
    return (f"{record['layer']}-{record['expert_type']}-{expert}-"
            f"{record['neuron']}-{direction}")


def public_records(rows: list) -> dict[str, dict]:
    records = {}
    for row in rows:
        expert = row["expert"]
        coordinate = f"{row['layer']}:{row['expert_type']}"
        if expert is not None:
            coordinate += f":{expert}"
        unit = parse_unit(f"{coordinate}:{row['neuron']}")
        direction = row["direction"]
        strength = row["strength"]
        if not isinstance(direction, str) or direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction for {coordinate}:{unit.neuron}.")
        if type(strength) not in (float, int) or not math.isfinite(strength) or strength <= 0:
            raise ValueError(f"Invalid strength for {coordinate}:{unit.neuron}.")

        sensitive = row.get("sensitive")
        if sensitive is not None and type(sensitive) is not bool:
            raise ValueError(f"Invalid sensitivity label for {coordinate}:{unit.neuron}.")
        reply = "" if sensitive else row["reply"]
        name = None if sensitive else row.get("name")
        if not isinstance(reply, str) or (name is not None and not isinstance(name, str)):
            raise ValueError(f"Invalid reply or name for {coordinate}:{unit.neuron}.")
        category = row.get("category") or "unlabeled"
        if not isinstance(category, str) or category not in CATEGORY_NAMES:
            raise ValueError(f"Unknown category for {coordinate}:{unit.neuron}.")

        record = {
            "layer": unit.layer, "expert_type": unit.expert_type, "expert": unit.expert,
            "neuron": unit.neuron, "direction": direction, "strength": strength,
            "category": category,
            "name": name if name and name.strip() else None, "reply": reply, "sensitive": bool(sensitive),
        }
        slug = reply_slug(record, direction)
        if slug in records:
            raise ValueError(f"Duplicate atlas reply: {slug}.")
        records[slug] = record
    return records


def render_page(record: dict, slug: str, opposite: str | None, prompt: str) -> str:
    layer, expert, neuron = record["layer"], record["expert"], record["neuron"]
    expert_type = record["expert_type"].title()
    expert_name = expert_type if expert is None else f"{expert_type} expert {expert}"
    expert_short = expert_type if expert is None else f"{expert_type} {expert}"
    coordinate = f"Layer {layer} · {expert_name} · Neuron {neuron}"
    direction = record["direction"]
    strength = f"{record['strength']:g}×"
    category_key = record["category"]
    category = CATEGORY_NAMES[category_key]
    title = record["name"] or category
    reply = record["reply"]
    if record["sensitive"]:
        title = "Sensitive reply"
        description = "This reply is marked sensitive. You can choose to view it in the atlas."
        image_alt = "Preview card for a sensitive Inkling-Small reply."
        content = f'<p class="share-note">{escape(description)}</p>'
    else:
        excerpt = " ".join(reply.split())
        description = excerpt if len(excerpt) <= 175 else excerpt[:174].rstrip() + "…"
        description = description or "Inkling-Small generated no visible reply for this push."
        image_alt = f"Inkling-Small reply card. {description}"
        quote = (f'<blockquote class="share-quote">{escape(reply)}</blockquote>' if excerpt else
                 f'<p class="share-note">{escape(description)}</p>')
        content = f'<p class="share-prompt">{escape(prompt)}</p>{quote}'
    page_url = f"{SITE_URL}/reply/{slug}/"
    image_url = f"{page_url}og.png"
    page_title = escape(f"{title} · L{layer} {expert_short} N{neuron} {direction.title()}")
    description = escape(description)
    image_alt = escape(image_alt)
    compare = ""
    if opposite:
        opposite_label = "down" if direction == "up" else "up"
        compare = f'<a class="button" href="../{opposite}/">Compare push {opposite_label}</a>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#10131b">
  <title>{page_title}</title>
  <meta name="description" content="{description}">
  <link rel="canonical" href="{page_url}">
  <meta property="og:type" content="website">
  <meta property="og:site_name" content="{SITE_NAME}">
  <meta property="og:title" content="{page_title}">
  <meta property="og:description" content="{description}">
  <meta property="og:url" content="{page_url}">
  <meta property="og:image" content="{image_url}">
  <meta property="og:image:type" content="image/png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="{image_alt}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{page_title}">
  <meta name="twitter:description" content="{description}">
  <meta name="twitter:url" content="{page_url}">
  <meta name="twitter:image" content="{image_url}">
  <meta name="twitter:image:alt" content="{image_alt}">
  <link rel="stylesheet" href="../../style.css">
  <link rel="stylesheet" href="../../share.css">
</head>
<body class="share-body">
  <header class="share-bar">
    <div class="shell share-bar-inner">
      <a class="brand" href="../../">
        <svg aria-hidden="true" viewBox="0 0 32 32" fill="none"><path d="M16 4v24M4 16h24M7.5 7.5l17 17m0-17-17 17" stroke="currentColor" stroke-width="1.5"/><circle cx="16" cy="16" r="4" fill="currentColor" stroke="#10131b" stroke-width="2"/></svg>
        <span>The <span class="ink-accent">Ink</span>omplete <b>Inkling Persona Atlas</b></span>
      </a>
    </div>
  </header>
  <main class="shell share-page">
    <header class="share-heading">
      <p class="share-coordinate">{coordinate}</p>
      <h1>{escape(title)}</h1>
      <dl class="share-facts">
        <div><dt>Category</dt><dd class="share-category" data-category="{escape(category_key)}">{category}</dd></div>
        <div><dt>Direction</dt><dd>Push {direction}</dd></div>
        <div><dt>Strength</dt><dd>{strength}</dd></div>
      </dl>
    </header>
    <article class="share-card" aria-labelledby="share-reply-title">
      <h2 id="share-reply-title">Inkling-Small reply</h2>
      {content}
    </article>
    <nav class="share-actions" aria-label="Browse replies">
      <a class="button button-accent" href="../../#library">Back to library</a>
      {compare}
    </nav>
  </main>
</body>
</html>
"""


def build(output: Path) -> int:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")

    run_bytes = (ATLAS_DIR / "run.json").read_bytes()
    run = json.loads(run_bytes)
    records = public_records(run["results"])
    output.mkdir(parents=True, exist_ok=True)
    for asset in PUBLIC_ASSETS:
        copyfile(ATLAS_DIR / asset, output / asset)
    (output / "run.json").write_bytes(run_bytes)

    for slug, record in records.items():
        direction = "down" if record["direction"] == "up" else "up"
        opposite = reply_slug(record, direction)
        reply_dir = output / "reply" / slug
        reply_dir.mkdir(parents=True)
        page = render_page(record, slug, opposite if opposite in records else None, run["prompt"])
        (reply_dir / "index.html").write_text(page, encoding="utf-8")
        render_og(record, reply_dir / "og.png")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New or empty directory for public files")
    args = parser.parse_args()
    try:
        count = build(args.output_dir)
    except (argparse.ArgumentTypeError, KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Built the atlas and {count} reply pages in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
