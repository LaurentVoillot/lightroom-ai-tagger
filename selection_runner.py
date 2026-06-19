"""
selection_runner.py — Tague une SÉLECTION de photos exportée depuis Lightroom.

Conçu pour être lancé par le plugin Lightroom Classic (voir lrc_plugin/).
Le plugin exporte les photos sélectionnées en JPEG dans un dossier temporaire
et y écrit un manifeste `manifest.json` :

    {
      "catalog": "/.../X.lrcat",
      "photos": [
        {
          "id": "uuid",
          "file": "0001.jpg",            # JPEG exporté (dans le même dossier)
          "name": "_DSC0001.NEF",        # nom du fichier original
          "folder": "/Volumes/.../2012-0160 Laos",
          "xmp": "/Volumes/.../_DSC0001.xmp",
          "lat": 13.43, "lon": 103.88,   # null si pas de GPS
          "has_gps": true
        }, ...
      ]
    }

Avantages :
  - Aucun accès au .lrcat (qui est verrouillé quand Lightroom est ouvert).
  - On travaille sur l'image telle que vue dans Lightroom (rendu/recadrage inclus).
  - Le script reste autonome et réutilise tout le pipeline existant.

Au lancement, le script PROPOSE INTERACTIVEMENT toutes les options
(tagging LLM, modèle, passe 2 espèces, GPS online, écriture XMP…), puis traite
la sélection et écrit un rapport. En MODE TEST il n'écrit aucun XMP.

Usage :
    python selection_runner.py /chemin/vers/manifest.json
    python selection_runner.py /chemin/vers/manifest.json --defaults   # sans questions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from catalog_reader import PhotoRecord
from image_source import ResolvedImage, SourceKind
from log_panel import RunStats, setup_logger


# --------------------------------------------------------------------------
# Petites aides d'interaction terminal
# --------------------------------------------------------------------------


def ask_yes_no(question: str, default: bool) -> bool:
    suffix = " [O/n] " if default else " [o/N] "
    try:
        # PYTHON — input(prompt) : lit une ligne au clavier (stdin), renvoie une
        # str. CHAÎNAGE de méthodes : .strip() enlève les espaces, .lower() met en
        # minuscules — chaque méthode renvoie une nouvelle str sur laquelle on
        # enchaîne. EOFError survient si stdin est fermé (entrée redirigée/vide).
        ans = input(question + suffix).strip().lower()
    except EOFError:
        return default
    if not ans:           # chaîne vide -> on garde le défaut
        return default
    # `x in (a, b, c)` : appartenance à un tuple. Vrai si ans vaut l'un d'eux.
    return ans in ("o", "oui", "y", "yes")


def ask_choice(question: str, options: list[str], default_idx: int = 0) -> str:
    print(question)
    for i, opt in enumerate(options):
        mark = " (défaut)" if i == default_idx else ""
        print(f"  {i + 1}. {opt}{mark}")
    try:
        ans = input("Choix : ").strip()
    except EOFError:
        return options[default_idx]
    if not ans:
        return options[default_idx]
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    return options[default_idx]


# --------------------------------------------------------------------------
# Construction des enregistrements depuis le manifeste
# --------------------------------------------------------------------------


def _record_from_entry(entry: dict) -> PhotoRecord:
    name = entry.get("name", entry.get("file", "?"))
    base, _, ext = name.rpartition(".")
    folder = entry.get("folder", "")
    return PhotoRecord(
        uuid=entry.get("id", name),
        file_uuid=entry.get("id", name),
        image_id=0,
        base_name=base or name,
        extension=ext,
        folder_abs=folder,
        file_format=None,
        has_gps=bool(entry.get("has_gps")),
        gps_lat=entry.get("lat"),
        gps_lon=entry.get("lon"),
        year=None,
        month=None,
        day=None,
    )


# --------------------------------------------------------------------------
# Options collectées (interactif ou défauts)
# --------------------------------------------------------------------------


def collect_options(defaults: bool) -> dict:
    if defaults:
        return {
            "tag": True,
            "model": "qwen3-vl:30b",
            "species": False,
            "online_species": True,
            "online_place": False,
            "write_xmp": False,
        }
    print("\n=== Options de taggage ===")
    tag = ask_yes_no("Générer les tags avec le LLM ?", True)
    model = "qwen3-vl:30b"
    species = False
    online_species = True
    online_place = False
    if tag:
        model = ask_choice(
            "Modèle de la passe 1 :",
            ["qwen3-vl:30b", "qwen2.5vl:7b", "qwen3-vl:8b"],
            0,
        )
        species = ask_yes_no(
            "Activer la passe 2 espèces (BioCLIP, expérimental) ?", False
        )
        online_species = ask_yes_no(
            "Filtrer les espèces par GPS via GBIF (réseau) ?", True
        )
        online_place = ask_yes_no(
            "Enrichir les lieux via Nominatim/OSM (réseau, POI) ?", False
        )
    write_xmp = ask_yes_no(
        "Écrire les tags dans des sidecars .xmp (sinon MODE TEST sans écriture) ?",
        False,
    )
    return {
        "tag": tag,
        "model": model,
        "species": species,
        "online_species": online_species,
        "online_place": online_place,
        "write_xmp": write_xmp,
    }


# --------------------------------------------------------------------------
# Traitement
# --------------------------------------------------------------------------


def run(manifest_path: str, opts: dict, interactive: bool) -> None:
    manifest_file = Path(manifest_path)
    work_dir = manifest_file.parent
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = data.get("photos", [])

    stats = RunStats()
    log = setup_logger(log_dir=work_dir, stats=stats)
    log.info("=== Taggage de la SÉLECTION Lightroom (%d photo(s)) ===", len(entries))

    mode = "ÉCRITURE XMP" if opts["write_xmp"] else "calcul des tags (pas de .xmp)"
    log.info("Mode : %s", mode)

    pipeline = None
    if opts["tag"]:
        from pipeline import OllamaVision, TaggingPipeline
        from gps_context import GpsContext

        gps = GpsContext(
            cache_dir=work_dir / "gps_cache",
            online_place=opts["online_place"],
            online_species=opts["online_species"],
        )
        pipeline = TaggingPipeline(
            ollama=OllamaVision(model=opts["model"]),
            gps=gps,
            use_species_pass=opts["species"],
        )
        log.info("Tagging activé (modèle %s, passe 2 espèces : %s)",
                 opts["model"], "oui" if opts["species"] else "non")

    xmp_writer = None
    if opts["write_xmp"]:
        from writers import XmpWriter

        xmp_writer = XmpWriter(suffix=opts.get("suffix", "_AI"))

    txt_lines: list[str] = []
    csv_rows: list[dict] = []
    results: dict[str, list[str]] = {}  # id photo -> tags (pour écriture Lua dans LrC)

    total = len(entries)
    progress_path = work_dir / "progress.json"
    results_path = work_dir / "results.json"

    def write_results() -> None:
        """Écrit results.json de façon ATOMIQUE (tmp + rename) après chaque photo.

        Ainsi, si le process est tué en cours (annulation depuis Lightroom), le
        plugin peut quand même taguer les photos déjà traitées, et ne lit jamais
        un fichier à moitié écrit.
        """
        tmp = results_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps({"tags_by_id": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(results_path)
        except Exception:
            pass

    def write_progress(done: int, current: str, finished: bool = False) -> None:
        """Écrit l'avancement pour la fenêtre de progression Lightroom (Lua)."""
        try:
            progress_path.write_text(
                json.dumps(
                    {"done": done, "total": total, "current": current,
                     "finished": finished},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass  # l'avancement ne doit jamais faire échouer le traitement

    write_progress(0, "Démarrage…")

    for i, entry in enumerate(entries, 1):
        rec = _record_from_entry(entry)
        jpeg = work_dir / entry["file"]
        ref = f"{i:04d}/{rec.display_name}"
        write_progress(i - 1, rec.display_name)
        if not jpeg.is_file():
            stats.bump("skipped")
            log.warning("%s : JPEG exporté introuvable — ignorée", ref)
            csv_rows.append({"n": i, "fichier": rec.display_name,
                             "statut": "MANQUANT", "gps": "", "tags": ""})
            continue

        try:
            img = Image.open(jpeg).convert("RGB")
            img.load()
        except Exception as e:
            stats.bump("skipped")
            log.warning("%s : JPEG illisible (%s) — ignorée", ref, e)
            continue

        stats.bump("processed")
        resolved = ResolvedImage(image=img, kind=SourceKind.ORIGINAL, path=str(jpeg))

        all_tags: list[str] = []
        write_tags = []
        place_tags: list[str] = []
        llm_tags: list[str] = []
        species_tags: list[str] = []
        if pipeline is not None:
            tr = pipeline.process(rec, resolved)
            place_tags, llm_tags, species_tags = tr.place_tags, tr.llm_tags, tr.species_tags
            all_tags = tr.merged()
            write_tags = tr.merged_hierarchical() if opts.get("hierarchical") else all_tags

        gps_str = (
            f"{rec.gps_lat:.5f},{rec.gps_lon:.5f}"
            if rec.has_gps and rec.gps_lat is not None
            else ""
        )
        detail = f"{ref}\n      gps    : {gps_str or '(pas de GPS)'}"
        if pipeline is not None:
            detail += (
                f"\n      lieu   : {', '.join(place_tags) or '—'}"
                f"\n      llm    : {', '.join(llm_tags) or '—'}"
                f"\n      espèces: {', '.join(species_tags) or '—'}"
            )
        txt_lines.append(detail)
        log.info("%s → %d tag(s)", ref, len(all_tags))
        if write_tags:
            # results.json pour l'écriture des mots-clés dans LrC par le Lua.
            # On normalise CHAQUE tag en chemin (liste de niveaux) pour que le
            # Lua ait un format uniforme : [["Lieu","Cambodge","Siem Reap"], ...].
            # Le suffixe n'est PAS appliqué ici (le Lua l'ajoute à la feuille).
            from writers import _as_path

            results[str(entry.get("id", rec.display_name))] = [
                _as_path(t) for t in write_tags
            ]
            write_results()  # sauvegarde incrémentale (résistante au kill)

        if xmp_writer is not None and write_tags:
            try:
                xmp_path = Path(entry.get("xmp") or rec.xmp_path)
                xmp_writer.write_tags(xmp_path, write_tags)
                stats.bump("xmp_written")
            except Exception as e:
                log.error("%s : écriture XMP échouée (%s)", ref, e)

        csv_rows.append({
            "n": i,
            "fichier": rec.display_name,
            "statut": "OK",
            "gps": gps_str,
            "tags": ", ".join(all_tags),
        })

    write_progress(total, "Terminé", finished=True)

    # Rapports
    txt_path = work_dir / "rapport_selection.txt"
    csv_path = work_dir / "rapport_selection.csv"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "fichier", "statut", "gps", "tags"])
        w.writeheader()
        w.writerows(csv_rows)

    # results.json a déjà été écrit de façon incrémentale après chaque photo
    # (write_results) ; on garantit ici une version finale complète.
    write_results()

    for line in stats.summary_lines():
        log.info(line)
    log.info("Rapport : %s", txt_path)
    log.info("CSV     : %s", csv_path)
    log.info("Résultats (pour LrC) : %s", results_path)
    if interactive:
        try:
            input("\nTerminé. Appuie sur Entrée pour fermer…")
        except EOFError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Tague une sélection Lightroom exportée")
    ap.add_argument("manifest", help="Chemin du manifest.json produit par le plugin")
    ap.add_argument("--defaults", action="store_true",
                    help="Utiliser les options par défaut sans poser de questions")
    # Options passées par le plugin Lua (mode non interactif piloté par le
    # dialogue natif Lightroom). Si --no-questions est absent, on reste interactif.
    ap.add_argument("--no-questions", action="store_true",
                    help="Ne pose aucune question : utilise les options des arguments")
    ap.add_argument("--no-tag", action="store_true", help="Ne pas générer de tags LLM")
    ap.add_argument("--model", default="qwen3-vl:30b", help="Modèle Ollama passe 1")
    ap.add_argument("--species", action="store_true", help="Activer la passe 2 BioCLIP")
    ap.add_argument("--no-online-species", action="store_true",
                    help="Désactiver le filtrage GBIF (online)")
    ap.add_argument("--online-place", action="store_true",
                    help="Enrichir les lieux via Nominatim (online)")
    ap.add_argument("--write-xmp", action="store_true",
                    help="Écrire aussi des sidecars .xmp")
    ap.add_argument("--hierarchical", action="store_true",
                    help="Produire des mots-clés hiérarchiques (lieu, espèces)")
    ap.add_argument("--suffix", default="_AI", help="Suffixe des tags (def: _AI)")
    args = ap.parse_args()

    if args.no_questions:
        opts = {
            "tag": not args.no_tag,
            "model": args.model,
            "species": args.species,
            "online_species": not args.no_online_species,
            "online_place": args.online_place,
            "write_xmp": args.write_xmp,
            "hierarchical": args.hierarchical,
            "suffix": args.suffix,
        }
        interactive = False
    else:
        opts = collect_options(args.defaults)
        interactive = not args.defaults

    run(args.manifest, opts, interactive)


if __name__ == "__main__":
    main()
