"""
bench_llm.py — Benchmark des modèles vision Ollama pour la passe 1.

Prend N photos du catalogue (résolues via la cascade), envoie chacune à
plusieurs modèles, et compare :
  - le temps moyen par photo (médiane + moyenne),
  - le nombre moyen de tags produits,
  - les tags eux-mêmes, côte à côte, pour juger la pertinence.

N'écrit RIEN dans le catalogue (lecture seule). Produit un rapport texte +
un CSV récapitulatif.

Usage :
    .venv/bin/python bench_llm.py "/Volumes/X10/LR-v15/LR-v15.lrcat" \
        --scope "2012-0160 Laos" --limit 5 \
        --models qwen3-vl:30b qwen3-vl:8b qwen2.5vl:7b
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from catalog_reader import CatalogReader
from image_source import ImageResolver
from log_panel import setup_logger
from pipeline import OllamaVision


def _pick_photos(cat: CatalogReader, scope: str | None, limit: int):
    """Sélectionne des photos réellement résolvables par la cascade."""
    resolver = ImageResolver(cat.previews_dir, cat.smart_previews_dir)
    out = []
    for rec in cat.iter_photos(folder_substring=scope):
        resolved = resolver.resolve(rec)
        if resolved is not None:
            out.append((rec, resolved))
            if len(out) >= limit:
                break
    resolver.close()
    return out


def run_bench(
    lrcat: str, scope: str | None, limit: int, models: list[str], out_dir: str | None
) -> None:
    out = Path(out_dir) if out_dir else Path.cwd()
    log = setup_logger(log_dir=out)
    log.info("=== BENCHMARK LLM (lecture seule) ===")
    log.info("Modèles : %s", ", ".join(models))

    with CatalogReader(lrcat) as cat:
        photos = _pick_photos(cat, scope, limit)
        log.info("%d photo(s) sélectionnée(s).", len(photos))

        # Pré-charge les images une seule fois (réutilisées par tous les modèles).
        images = [(rec, resolved.image) for rec, resolved in photos]

        per_model_times: dict[str, list[float]] = {m: [] for m in models}
        per_model_ntags: dict[str, list[int]] = {m: [] for m in models}
        txt_lines: list[str] = []

        # Réutilise un client par modèle (et non un par photo).
        clients = {m: OllamaVision(model=m) for m in models}

        # Warm-up : 1er appel = chargement du modèle en VRAM. On le fait hors
        # mesure pour que les temps reflètent l'inférence seule, pas le chargement.
        if images:
            warm_rec, warm_img = images[0]
            for model in models:
                log.info("Préchauffage %s…", model)
                clients[model].analyze(warm_img)

        for rec, img in images:
            txt_lines.append(f"\n### {rec.display_name}")
            for model in models:
                client = clients[model]
                t0 = time.time()
                tags, cats, _ = client.analyze(img)
                dt = time.time() - t0
                per_model_times[model].append(dt)
                per_model_ntags[model].append(len(tags))
                log.info("  %-18s %5.1fs  %2d tags", model, dt, len(tags))
                txt_lines.append(
                    f"  [{model}]  {dt:.1f}s  cat={cats or '—'}\n"
                    f"      {', '.join(tags) or '(aucun)'}"
                )

    # --- Récapitulatif ---
    txt_lines.append("\n\n=== RÉCAPITULATIF ===")
    summary_rows = []
    for model in models:
        times = per_model_times[model]
        ntags = per_model_ntags[model]
        if not times:
            continue
        med = statistics.median(times)
        avg = statistics.mean(times)
        avg_tags = statistics.mean(ntags)
        line = (
            f"{model:20s}  médiane {med:5.1f}s  moy {avg:5.1f}s  "
            f"tags/photo {avg_tags:4.1f}"
        )
        txt_lines.append("  " + line)
        log.info(line)
        summary_rows.append(
            {
                "model": model,
                "median_s": round(med, 2),
                "mean_s": round(avg, 2),
                "avg_tags": round(avg_tags, 1),
                "n_photos": len(times),
            }
        )

    txt_path = out / "bench_llm.txt"
    csv_path = out / "bench_llm.csv"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["model", "median_s", "mean_s", "avg_tags", "n_photos"]
        )
        w.writeheader()
        w.writerows(summary_rows)
    log.info("Rapport détaillé : %s", txt_path)
    log.info("Récapitulatif CSV : %s", csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark des modèles vision Ollama")
    ap.add_argument("lrcat", help="Chemin du catalogue .lrcat")
    ap.add_argument("--scope", help="Sous-chaîne de chemin de dossier (périmètre)")
    ap.add_argument("--limit", type=int, default=5, help="Nombre de photos (def: 5)")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["qwen3-vl:30b", "qwen3-vl:8b", "qwen2.5vl:7b"],
        help="Liste des modèles Ollama à comparer",
    )
    ap.add_argument("--out", help="Dossier de sortie (def: courant)")
    args = ap.parse_args()
    run_bench(args.lrcat, args.scope, args.limit, args.models, args.out)


if __name__ == "__main__":
    main()
