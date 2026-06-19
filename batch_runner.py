"""
batch_runner.py — Traite tout le catalogue (ou un périmètre) par LOTS, avec
reprise automatique. Conçu pour les très gros volumes (100k+ photos) sans
surveillance.

Principe :
  - On découpe le travail en lots de N photos.
  - Chaque lot est traité par run_test(resume=True) : les photos déjà faites
    (mémorisées dans out_dir/session.db) sont automatiquement sautées.
  - Entre deux lots : on vérifie que le volume du catalogue est toujours monté.
    S'il a décroché, on s'arrête proprement (la session est sauvegardée, on
    pourra relancer plus tard et reprendre où on en était).
  - Robustesse : une exception sur un lot est journalisée ; on tente le lot
    suivant (les photos du lot fautif ne sont pas marquées, donc retentées au
    prochain passage).

Usage :
    python batch_runner.py "/Volumes/X10/.../LR-v15.lrcat" \
        --out ~/phototagger_out --batch 200 \
        --tag --write --catalog --hierarchical [--scope "..."] [--species]

Reprendre après interruption : relancer EXACTEMENT la même commande. Grâce à
--out identique (donc la même session.db), seules les photos restantes sont
traitées.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog_reader import CatalogReader
from log_panel import get_logger, setup_logger
from test_report import run_test, _parse_order


def main() -> None:
    ap = argparse.ArgumentParser(description="Traitement par lots avec reprise")
    ap.add_argument("lrcat", help="Chemin du catalogue .lrcat")
    ap.add_argument("--out", required=True, help="Dossier de sortie (porte session.db)")
    ap.add_argument("--batch", type=int, default=200, help="Taille de lot (def: 200)")
    ap.add_argument("--scope", help="Restreindre à un dossier (sous-chaîne)")
    ap.add_argument("--gps-only", action="store_true")
    ap.add_argument("--tag", action="store_true", help="Générer les tags LLM")
    ap.add_argument("--model", default="qwen3-vl:30b")
    ap.add_argument("--species", action="store_true", help="Passe 2 BioCLIP")
    ap.add_argument("--no-online-species", action="store_true")
    ap.add_argument("--online-place", action="store_true")
    ap.add_argument("--write", action="store_true", help="Écriture réelle (pas test)")
    ap.add_argument("--xmp", action="store_true")
    ap.add_argument("--catalog", action="store_true")
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--suffix", default="_AI")
    ap.add_argument("--max-batches", type=int, default=0,
                    help="Limiter le nombre de lots (0 = illimité)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger(log_dir=out)
    log.info("=== TRAITEMENT PAR LOTS (lot=%d) ===", args.batch)

    # Combien de photos au total dans le périmètre (avec Smart Preview) ?
    try:
        with CatalogReader(args.lrcat) as cat:
            # PYTHON — COMPTAGE PARESSEUX : sum(1 for _ in generateur). On somme
            # un 1 par élément SANS construire de liste -> compte en O(1) mémoire,
            # idéal sur 200 000 photos. `_` = variable « jetable » (convention pour
            # « je n'utilise pas cette valeur »).
            total = sum(
                1 for _ in cat.iter_photos(
                    folder_substring=args.scope, gps_only=args.gps_only,
                    with_smart_preview_only=True,
                )
            )
    except Exception as e:
        log.error("Catalogue illisible (%s). Volume monté ?", e)
        return
    log.info("%d photo(s) à traiter au total (avant reprise).", total)

    order = _parse_order("preview,smart,original")
    lrcat_path = Path(args.lrcat)
    # Fichier sentinelle d'arrêt propre : sa présence demande l'arrêt À LA FIN
    # du lot courant (le lot en cours se termine et est commité avant l'arrêt).
    stop_flag = out / "stop.flag"
    if stop_flag.exists():
        stop_flag.unlink()  # nettoie un éventuel reliquat
    batch_no = 0
    while True:
        # Demande d'arrêt propre (bouton Stop) reçue entre deux lots ?
        if stop_flag.exists():
            log.info("Arrêt demandé (stop.flag) — arrêt propre, session sauvegardée.")
            stop_flag.unlink()
            break

        # Garde-fou : le volume du catalogue est-il toujours là ?
        if not lrcat_path.is_file():
            log.error("Le catalogue n'est plus accessible (volume démonté ?). "
                      "Arrêt propre — relance la même commande pour reprendre.")
            break

        batch_no += 1
        if args.max_batches and batch_no > args.max_batches:
            log.info("Nombre de lots max atteint (%d). Arrêt.", args.max_batches)
            break

        log.info("--- Lot %d (jusqu'à %d photos) ---", batch_no, args.batch)
        try:
            # run_test avec resume=True saute les photos déjà faites. La limite
            # = taille de lot ; comme la session grandit, chaque appel traite le
            # lot suivant de photos non encore traitées.
            run_test(
                lrcat=args.lrcat,
                scope=args.scope,
                limit=args.batch,
                gps_only=args.gps_only,
                order=order,
                out_dir=str(out),
                tag=args.tag,
                model=args.model,
                online_species=not args.no_online_species,
                species_pass=args.species,
                test_mode=not args.write,
                write_xmp=args.xmp,
                write_catalog=args.catalog,
                suffix=args.suffix,
                hierarchical=args.hierarchical,
                resume=True,
                stop_flag=str(stop_flag),
            )
        except Exception as e:
            log.error("Lot %d en échec (%s) — on tente le lot suivant.", batch_no, e)
            # PYTHON — `continue` saute à l'itération suivante de la boucle (le lot
            # suivant). `break` (ailleurs) sort complètement de la boucle.
            continue

        # Combien de photos restent à faire ? (session vs total)
        from session_cache import SessionCache

        # `with ... as sc:` ouvre et FERME proprement la session (context manager).
        with SessionCache(out / "session.db") as sc:
            done = sc.count()
        log.info("Progression : %d / %d photo(s) traitées.", done, total)
        if done >= total:
            log.info("=== TERMINÉ : toutes les photos du périmètre sont traitées. ===")
            break


if __name__ == "__main__":
    main()
