"""
test_report.py — MODE TEST (lecture seule) du Photo Tagger.

Objectif : valider toute la chaîne sans aucun risque pour la base Lightroom.
  - Le catalogue n'est ouvert qu'en lecture seule (mode=ro, immutable=1).
  - AUCUN fichier XMP n'est écrit, la base Lightroom n'est JAMAIS modifiée.
  - Sortie sous la forme demandée : « numéro_dossier/nom_de_fichier → tags ».
  - Produit aussi un rapport CSV (colonne `statut` pour repérer les
    photos indisponibles) et un fichier texte.

À ce stade, le pipeline de tags (LLM + espèces) n'est pas branché : le rapport
montre la SOURCE d'image effectivement résolue par la cascade et l'état GPS.
Les colonnes de tags seront remplies quand pipeline.py sera connecté.

Pré-vol volumes : on détecte une seule fois au démarrage si un volume requis
est absent (cf. log_panel.preflight_volumes). Comme un volume ne se montera pas
en cours de route, on ne répète jamais ce message.

Usage :
    .venv/bin/python test_report.py "/Volumes/X10/LR-v15/LR-v15.lrcat" \
        --scope "2020-0406 Orion Test1" --limit 20
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from catalog_reader import CatalogReader, PhotoRecord
from image_source import ImageResolver, SourceKind, ResolvedImage
from log_panel import RunStats, get_logger, preflight_volumes, setup_logger

# Étiquettes lisibles pour la source résolue.
_KIND_LABEL = {
    SourceKind.PREVIEW: "aperçu standard",
    SourceKind.SMART: "Smart Preview",
    SourceKind.ORIGINAL: "original",
}


def _flush_log(log) -> None:
    """Force l'écriture disque des handlers (progression visible en direct)."""
    for h in log.handlers:
        try:
            h.flush()
        except Exception:
            pass


def _folder_numbers(records: list[PhotoRecord]) -> dict[str, str]:
    """Attribue un numéro stable à chaque dossier rencontré (0001, 0002, ...)."""
    mapping: dict[str, str] = {}   # dict vide ; {} seul = dict (pas set)
    n = 0
    for r in records:
        if r.folder_abs not in mapping:   # `not in` sur un dict = clé absente
            n += 1
            # PYTHON — FORMAT SPEC dans une f-string : f"{n:04d}" = entier sur 4
            # chiffres avec zéros à gauche ("0001"). Équivaut à printf("%04d").
            mapping[r.folder_abs] = f"{n:04d}"
    return mapping


def run_test(
    lrcat: str,
    scope: str | None,
    limit: int | None,
    gps_only: bool,
    order: tuple[SourceKind, ...],
    out_dir: str | None,
    tag: bool = False,
    model: str = "qwen3-vl:30b",
    online_species: bool = True,
    species_pass: bool = False,
    test_mode: bool = True,
    write_xmp: bool = False,
    write_catalog: bool = False,
    suffix: str = "_AI",
    selected_only: bool = False,
    skip_tagged: bool = True,
    hierarchical: bool = False,
    resume: bool = False,
    progress_cb=None,
    stop_flag: str | None = None,
    num_gpu: int | None = None,
) -> None:
    out = Path(out_dir) if out_dir else Path.cwd()
    stats = RunStats()
    log = setup_logger(log_dir=out, stats=stats)

    # Garde-fou : le catalogue doit être accessible (volume monté). Évite le
    # cryptique « database disk image is malformed » si le disque est démonté.
    if not Path(lrcat).is_file():
        log.error(
            "Catalogue inaccessible : %s. Le volume est-il bien monté "
            "(ex. /Volumes/X10) ? Rebranche le disque puis relance.", lrcat,
        )
        return

    # En MODE TEST, toute écriture est désactivée quoi qu'il arrive.
    if test_mode:
        write_xmp = False
        write_catalog = False
        log.info("=== MODE TEST (lecture seule, aucune écriture) ===")
    else:
        log.info("=== MODE RÉEL — écriture : XMP=%s, catalogue=%s (suffixe '%s') ===",
                 "oui" if write_xmp else "non",
                 "oui" if write_catalog else "non", suffix)

    # Pipeline de tags (optionnel) : passe 1 LLM + GPS + passe 2 espèces.
    pipeline = None
    if tag:
        from pipeline import TaggingPipeline, OllamaVision
        from gps_context import GpsContext

        gps = GpsContext(
            cache_dir=out / "gps_cache", online_species=online_species
        )
        pipeline = TaggingPipeline(
            ollama=OllamaVision(model=model, num_gpu=num_gpu),
            gps=gps, use_species_pass=species_pass
        )
        log.info(
            "Tagging ACTIVÉ (modèle passe 1 : %s, passe 2 espèces : %s)",
            model,
            "oui" if species_pass else "non",
        )

    # Writers d'écriture réelle (hors mode test). Init défensive : si le
    # catalogue est verrouillé (Lightroom ouvert), on bascule sans planter.
    xmp_writer = None
    catalog_writer = None
    if write_xmp:
        from writers import XmpWriter
        xmp_writer = XmpWriter(suffix=suffix)
    if write_catalog:
        from writers import CatalogWriter
        try:
            catalog_writer = CatalogWriter(lrcat, suffix=suffix)
        except RuntimeError as e:
            log.error("Écriture catalogue impossible : %s", e)
            write_catalog = False

    session = None  # session de reprise (initialisée dans le with si resume)

    # Si on écrit dans le catalogue, le reader doit voir les écritures du writer
    # -> pas de mode immutable (B2 : éviter reader figé + writer RW concurrent).
    with CatalogReader(lrcat, immutable=not write_catalog) as cat:
        # Périmètre : sélection courante (persistée par LrC) ou filtre dossier.
        image_ids = None
        if selected_only:
            image_ids = cat.selected_image_ids()
            log.info("Sélection courante : %d photo(s).", len(image_ids))
            if not image_ids:
                log.warning(
                    "Aucune sélection trouvée dans le catalogue "
                    "(Adobe_selectedImages vide). Rien à traiter."
                )

        # Par défaut, on ne traite que les photos ayant un VRAI Smart Preview
        # (présentes dans AgDNGProxyInfo) — sauf si on cible une sélection
        # explicite. Évite que les photos cloud (Mobile Downloads, sans pixels
        # sur disque), placées en tête par le tri, saturent la limite.
        sp_only = not selected_only
        # En mode reprise, la limite (taille de lot) doit s'appliquer APRÈS le
        # filtre des photos déjà faites — sinon on re-sélectionne toujours les
        # mêmes premières photos. On ne limite donc pas la requête SQL ici.
        sql_limit = None if resume else limit
        records = list(
            cat.iter_photos(
                folder_substring=scope, gps_only=gps_only, limit=sql_limit,
                image_ids=image_ids, with_smart_preview_only=sp_only,
            )
        )
        log.info("%d photo(s) dans le périmètre.", len(records))

        # Skip des photos déjà taguées par l'IA (au moins un mot-clé finissant
        # par le suffixe). Le suffixe vide désactive ce skip (pas de marqueur).
        skipped_already = 0
        if skip_tagged and suffix:
            suf = suffix.lower()
            kept = []
            for rec in records:
                kws = cat.existing_keywords(rec.image_id)
                if any(k.lower().endswith(suf) for k in kws):
                    skipped_already += 1
                    stats.bump("skipped_already")
                else:
                    kept.append(rec)
            if skipped_already:
                log.info(
                    "%d photo(s) déjà taguée(s) IA (suffixe '%s') — ignorée(s).",
                    skipped_already, suffix,
                )
            records = kept

        # Reprise de session : ignore les photos déjà traitées par CE pipeline
        # (mémorisées dans out_dir/session.db). Permet de reprendre un gros run.
        if resume:
            from session_cache import SessionCache

            session = SessionCache(out / "session.db")
            before = len(records)
            records = [r for r in records if not session.is_done(r.file_uuid)]
            resumed = before - len(records)
            if resumed:
                log.info(
                    "Reprise : %d photo(s) déjà traitée(s) dans une session "
                    "précédente — ignorée(s) (%d déjà en base).",
                    resumed, session.count(),
                )
            # La limite (taille de lot) s'applique APRÈS le filtre de reprise.
            if limit:
                records = records[:limit]

        # Pré-vol : on ne vérifie les volumes des originaux que s'ils sont
        # réellement dans la cascade (sinon inutile de bloquer).
        if SourceKind.ORIGINAL in order:
            orig_paths = [r.original_path for r in records]
            # non fatal : si le volume des originaux manque, on continue avec
            # aperçus/smart previews et on le signale UNE seule fois.
            missing = preflight_volumes(orig_paths, logger=log, fatal=False)
            if missing:
                log.info(
                    "→ Ce n'est pas bloquant : les originaux sur ces volumes "
                    "sont ignorés, les aperçus/Smart Previews (sur le volume du "
                    "catalogue) prennent le relais."
                )

        resolver = ImageResolver(
            previews_dir=cat.previews_dir,
            smart_dir=cat.smart_previews_dir,
            order=order,
            cloud_smart_dir=cat.cloud_smart_previews_dir,
        )

        folder_num = _folder_numbers(records)
        txt_lines: list[str] = []
        csv_rows: list[dict] = []
        by_folder: dict[str, list[str]] = defaultdict(list)

        total_records = len(records)
        log.info("Début du traitement de %d photo(s)…", total_records)
        # PYTHON — EXPRESSION CONDITIONNELLE (ternaire) : `A if cond else B`
        # (l'ordre diffère de cond?A:B). Ici : Path(stop_flag) si fourni, sinon None.
        stop_path = Path(stop_flag) if stop_flag else None
        # PYTHON — enumerate(seq, 1) : itère en donnant (index, élément), index
        # démarrant à 1. Évite le classique `i = 0; i += 1`. Sans le `, 1`, l'index
        # commence à 0. On déballe directement en `idx, rec`.
        for idx, rec in enumerate(records, 1):
            # Arrêt propre demandé (bouton Stop) : on sort à la fin de la photo
            # précédente, tout est déjà commité/flushé. Le stop.flag est laissé
            # en place pour que le batch_runner s'arrête aussi entre deux lots.
            if stop_path is not None and stop_path.exists():
                log.info("Arrêt demandé — interruption propre après %d photo(s).",
                         idx - 1)
                break
            if progress_cb is not None:
                progress_cb(idx, total_records, rec.display_name)
            # Log de progression visible (toutes les photos) + flush, pour suivre
            # un gros run en direct dans le fichier log.
            # PYTHON — logging avec args SÉPARÉS (pas de f-string ici !) :
            # log.info("[%d/%d] %s", a, b, c) -> le formatage %d/%s n'est fait que
            # SI le message est réellement émis (niveau actif). Plus efficace que
            # f"[{a}]" qui formate toujours. C'est le style recommandé du module logging.
            log.info("[%d/%d] %s", idx, total_records, rec.display_name)
            _flush_log(log)
            resolved: ResolvedImage | None = resolver.resolve(rec)
            num = folder_num[rec.folder_abs]
            ref = f"{num}/{rec.display_name}"

            if resolved is None:
                stats.bump("skipped")
                gps = (
                    f"{rec.gps_lat:.5f},{rec.gps_lon:.5f}"
                    if rec.has_gps and rec.gps_lat is not None
                    else ""
                )
                txt_lines.append(f"{ref}  [INDISPONIBLE]")
                csv_rows.append(
                    {
                        "dossier": num,
                        "fichier": rec.display_name,
                        "statut": "INDISPONIBLE",
                        "source": "",
                        "gps": gps,
                        "tags": "",
                    }
                )
                continue

            stats.bump("processed")
            stats.bump(resolved.kind.value)
            src = _KIND_LABEL[resolved.kind]
            gps = (
                f"{rec.gps_lat:.5f},{rec.gps_lon:.5f}"
                if rec.has_gps and rec.gps_lat is not None
                else "(pas de GPS)"
            )
            # Tags : produits par le pipeline si activé, sinon vide.
            all_tags: list[str] = []
            write_tags = []  # ce qu'on écrit réellement (plat ou hiérarchique)
            place_tags: list[str] = []
            llm_tags: list[str] = []
            species_tags: list[str] = []
            if pipeline is not None:
                tr = pipeline.process(rec, resolved)
                place_tags, llm_tags, species_tags = (
                    tr.place_tags, tr.llm_tags, tr.species_tags
                )
                all_tags = tr.merged()
                write_tags = tr.merged_hierarchical() if hierarchical else all_tags

            detail = (
                f"{ref}\n      source : {src} ({resolved.image.size[0]}x{resolved.image.size[1]})"
                f"\n      gps    : {gps}"
            )
            if pipeline is not None:
                detail += (
                    f"\n      lieu   : {', '.join(place_tags) or '—'}"
                    f"\n      llm    : {', '.join(llm_tags) or '—'}"
                    f"\n      espèces: {', '.join(species_tags) or '—'}"
                )

            # Écriture réelle (hors mode test) : XMP et/ou catalogue, non destructif.
            if write_tags and xmp_writer is not None:
                added = xmp_writer.write_tags(Path(rec.xmp_path), write_tags)
                if added:
                    stats.bump("xmp_written", added)
                detail += f"\n      xmp    : +{added} tag(s)"
            if write_tags and catalog_writer is not None:
                # commit=False : écriture par lots, commit groupé tous les 10.
                added = catalog_writer.add_tags(rec.image_id, write_tags, commit=False)
                if added:
                    stats.bump("catalog_written", added)
                detail += f"\n      base   : +{added} tag(s)"
                if idx % 10 == 0:
                    catalog_writer.commit_batch()

            # Mémorise la photo comme traitée (reprise de session), commit tous
            # les 10 — assez fréquent pour une reprise fiable après un arrêt.
            # PYTHON — `idx % 10 == 0` (modulo) : vrai un tour sur 10. L'expression
            # booléenne est passée telle quelle au paramètre `commit=`.
            if session is not None:
                session.mark(rec.file_uuid, rec.display_name, len(all_tags),
                             commit=(idx % 10 == 0))

            # Log du résultat de la photo + flush, pour suivre en direct.
            log.info("    → %d tag(s)%s", len(all_tags),
                     "" if all_tags else " (aucun)")
            _flush_log(log)

            txt_lines.append(detail)
            by_folder[num].append(rec.display_name)
            csv_rows.append(
                {
                    "dossier": num,
                    "fichier": rec.display_name,
                    "statut": "OK",
                    "source": src,
                    "gps": gps if gps != "(pas de GPS)" else "",
                    "tags": ", ".join(all_tags),
                }
            )

        resolver.close()

    if catalog_writer is not None:
        catalog_writer.commit_batch()  # valide le dernier lot (< 50)
        catalog_writer.close()
    if session is not None:
        session.close()  # commit final de la session de reprise
    if pipeline is not None and pipeline.gps is not None:
        pipeline.gps.flush()  # B3 : écrit les caches GPS accumulés

    # --- Détection « mauvais catalogue / source introuvable » ---
    # Si des photos étaient dans le périmètre mais que TOUTES ont échoué à la
    # résolution d'image, c'est typiquement le mauvais catalogue (ex. cloud) ou
    # un volume démonté. On le signale clairement.
    n_in_scope = len(records)
    n_unavailable = stats.counter.get("skipped", 0)
    if n_in_scope > 0 and n_unavailable == n_in_scope:
        log.error(
            "AUCUNE image n'a pu être chargée (%d/%d). Catalogue inadapté "
            "(catalogue cloud/mobile ?) ou volume des originaux démonté.",
            n_unavailable, n_in_scope,
        )

    # --- Écriture des rapports (dans out_dir, jamais dans le catalogue) ---
    txt_path = out / "rapport_test.txt"
    csv_path = out / "rapport_test.csv"
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["dossier", "fichier", "statut", "source", "gps", "tags"]
        )
        w.writeheader()
        w.writerows(csv_rows)

    # --- Légende des numéros de dossier ---
    log.info("Correspondance numéros de dossier :")
    seen: set[str] = set()
    for rec in records:
        num = folder_num[rec.folder_abs]
        if num not in seen:
            seen.add(num)
            log.info("  %s = %s", num, rec.folder_abs)

    for line in stats.summary_lines():
        log.info(line)
    log.info("Rapport texte : %s", txt_path)
    log.info("Rapport CSV   : %s", csv_path)


def _parse_order(spec: str) -> tuple[SourceKind, ...]:
    alias = {
        "preview": SourceKind.PREVIEW,
        "previews": SourceKind.PREVIEW,
        "smart": SourceKind.SMART,
        "original": SourceKind.ORIGINAL,
        "originals": SourceKind.ORIGINAL,
    }
    out: list[SourceKind] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok not in alias:
            raise argparse.ArgumentTypeError(f"source inconnue : {tok}")
        out.append(alias[tok])
    return tuple(out)


# PYTHON — point d'entrée CLI. argparse = parseur d'arguments de ligne de commande
# de la stdlib. On déclare les arguments, il gère le parsing, --help, les erreurs.
def main() -> None:
    ap = argparse.ArgumentParser(description="Mode test (lecture seule) du Photo Tagger")
    # add_argument("lrcat", ...) = argument POSITIONNEL (obligatoire) ;
    # add_argument("--scope", ...) = argument OPTIONNEL (préfixe --). `type=int`
    # convertit automatiquement ; sans `type`, c'est une str.
    ap.add_argument("lrcat", help="Chemin du catalogue .lrcat")
    ap.add_argument("--scope", help="Sous-chaîne de chemin de dossier (périmètre)")
    ap.add_argument("--limit", type=int, help="Limiter le nombre de photos")
    # PYTHON — action="store_true" : argument-DRAPEAU. Présent -> True, absent ->
    # False. Pas de valeur à fournir (ex. `--gps-only`). argparse convertit aussi
    # `--gps-only` en attribut `args.gps_only` (tiret -> underscore).
    ap.add_argument("--gps-only", action="store_true", help="Photos géolocalisées uniquement")
    ap.add_argument(
        "--order",
        type=_parse_order,
        default=_parse_order("preview,smart,original"),
        help="Ordre de cascade des sources (def: preview,smart,original)",
    )
    ap.add_argument("--out", help="Dossier de sortie des rapports (def: courant)")
    ap.add_argument("--tag", action="store_true", help="Activer le pipeline de tags (LLM + espèces)")
    ap.add_argument("--model", default="qwen3-vl:30b", help="Modèle Ollama passe 1")
    ap.add_argument(
        "--no-online-species",
        action="store_true",
        help="Désactiver le filtrage d'espèces GBIF (online)",
    )
    ap.add_argument(
        "--species",
        action="store_true",
        help="Activer la passe 2 BioCLIP (identification d'espèces, expérimental)",
    )
    # Écriture réelle (par défaut : mode test, aucune écriture).
    ap.add_argument("--write", action="store_true",
                    help="Désactive le mode test et écrit réellement les tags")
    ap.add_argument("--xmp", action="store_true", help="Écrire des sidecars .xmp (avec --write)")
    ap.add_argument("--catalog", action="store_true",
                    help="Écrire dans la base LrC, Lightroom fermé (avec --write)")
    ap.add_argument("--suffix", default="_AI", help="Suffixe des tags (def: _AI ; vide possible)")
    ap.add_argument("--selected", action="store_true",
                    help="Traiter la sélection courante (persistée par LrC) au lieu d'un dossier")
    ap.add_argument("--no-skip-tagged", action="store_true",
                    help="Ne pas ignorer les photos déjà taguées par l'IA (même suffixe)")
    ap.add_argument("--hierarchical", action="store_true",
                    help="Écrire des mots-clés hiérarchiques (lieu, espèces) au lieu de plats")
    ap.add_argument("--resume", action="store_true",
                    help="Reprendre : ignorer les photos déjà traitées (out/session.db)")
    args = ap.parse_args()

    run_test(
        lrcat=args.lrcat,
        scope=args.scope,
        limit=args.limit,
        gps_only=args.gps_only,
        order=args.order,
        out_dir=args.out,
        tag=args.tag,
        model=args.model,
        online_species=not args.no_online_species,
        species_pass=args.species,
        test_mode=not args.write,
        write_xmp=args.xmp,
        write_catalog=args.catalog,
        suffix=args.suffix,
        selected_only=args.selected,
        skip_tagged=not args.no_skip_tagged,
        hierarchical=args.hierarchical,
        resume=args.resume,
    )


# PYTHON — IDIOME FONDAMENTAL : `if __name__ == "__main__":`. Chaque module a une
# variable `__name__`. Si le fichier est LANCÉ directement (python test_report.py),
# __name__ vaut "__main__" et on exécute main(). S'il est IMPORTÉ par un autre
# module, __name__ vaut "test_report" et ce bloc ne s'exécute PAS. C'est ce qui
# permet à un fichier d'être à la fois une bibliothèque ET un script exécutable.
if __name__ == "__main__":
    main()
