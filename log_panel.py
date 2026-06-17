"""
log_panel.py — Journalisation centralisée du Photo Tagger.

Objectifs :
  - Un logger unique réutilisable en CLI comme en GUI.
  - Sortie console + fichier .log horodaté.
  - Un handler Qt optionnel (branché par la GUI) pour afficher les messages
    dans une fenêtre de log avec colorisation par niveau.
  - Pré-vol des volumes : on vérifie UNE SEULE FOIS au démarrage que les
    volumes nécessaires sont montés. Un volume ne sera pas monté en cours de
    route, donc on n'émet jamais de warning répété « volume non monté » :
    soit on s'arrête proprement, soit on ignore en bloc la source concernée.

Ce module ne dépend pas de PyQt6 : le handler Qt n'est importé que si la GUI
l'appelle. La partie logique (pré-vol, comptage) reste utilisable sans interface.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "phototagger"

# ---------------------------------------------------------------------------
# Compteur de statistiques de run (sources utilisées, warnings, erreurs)
# ---------------------------------------------------------------------------


class RunStats:
    """Petit agrégateur de statistiques pour le récapitulatif de fin de run."""

    def __init__(self) -> None:
        self.counter: Counter[str] = Counter()

    def bump(self, key: str, n: int = 1) -> None:
        self.counter[key] += n

    def summary_lines(self) -> list[str]:
        c = self.counter
        lines = [
            "=== Terminé : %d photos traitées ===" % c.get("processed", 0),
            "  via aperçu standard : %d" % c.get("src_preview", 0),
            "  via Smart Preview   : %d" % c.get("src_smart", 0),
            "  via original        : %d" % c.get("src_original", 0),
            "  IGNORÉES (warnings) : %d" % c.get("skipped", 0),
        ]
        if c.get("xmp_written") or c.get("catalog_written"):
            lines.append(
                "  tags écrits : %d en .xmp · %d en base"
                % (c.get("xmp_written", 0), c.get("catalog_written", 0))
            )
        lines.append(
            "  warnings: %d · erreurs: %d" % (c.get("warning", 0), c.get("error", 0))
        )
        return lines


# ---------------------------------------------------------------------------
# Handler qui alimente RunStats à partir des niveaux de log émis
# ---------------------------------------------------------------------------


class _StatsHandler(logging.Handler):
    def __init__(self, stats: RunStats) -> None:
        super().__init__()
        self.stats = stats

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.stats.bump("error")
        elif record.levelno >= logging.WARNING:
            self.stats.bump("warning")


# ---------------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------------


def setup_logger(
    log_dir: str | os.PathLike | None = None,
    level: int = logging.INFO,
    stats: RunStats | None = None,
) -> logging.Logger:
    """Configure et renvoie le logger applicatif.

    - Écrit sur la console (stderr) : handler ajouté UNE seule fois.
    - Écrit dans un fichier `tagger_<horodatage>_<pid>.log` (nom unique) : un
      NOUVEAU fichier à chaque appel, pour que des runs successifs (notamment
      dans la GUI) n'écrivent pas dans le même fichier.
    - Branche le RunStats fourni : RE-câblé à chaque appel, sinon le résumé d'un
      2e run afficherait des zéros.

    Réentrant : appelable plusieurs fois sans dupliquer le handler console ni
    empiler des handlers fichier/stats obsolètes.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"
    )

    # Console : une seule fois (évite la duplication des lignes).
    if not getattr(logger, "_phototagger_console", False):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt)
        logger.addHandler(console)
        logger._phototagger_console = True  # type: ignore[attr-defined]

    # Fichier + stats : on retire les anciens (marqués) avant d'en remettre,
    # pour repartir proprement à chaque run.
    for h in list(logger.handlers):
        if getattr(h, "_phototagger_perrun", False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    base = Path(log_dir) if log_dir else Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Compteur de run (sur le logger) : garantit un nom unique même pour deux
    # runs dans la même seconde ET le même process (cas de la GUI).
    run_no = getattr(logger, "_phototagger_run_no", 0) + 1
    logger._phototagger_run_no = run_no  # type: ignore[attr-defined]
    logfile = base / f"tagger_{stamp}_{os.getpid()}_{run_no}.log"
    fileh = logging.FileHandler(logfile, encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    fileh._phototagger_perrun = True  # type: ignore[attr-defined]
    logger.addHandler(fileh)

    if stats is not None:
        sh = _StatsHandler(stats)
        sh._phototagger_perrun = True  # type: ignore[attr-defined]
        logger.addHandler(sh)

    logger._logfile = str(logfile)  # type: ignore[attr-defined]
    logger.info("Journal : %s", logfile)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------
# Pré-vol des volumes (vérification unique au démarrage)
# ---------------------------------------------------------------------------


class VolumeError(RuntimeError):
    """Levée quand un volume requis n'est pas monté au démarrage."""


def required_volumes(paths: list[str]) -> list[str]:
    """Extrait les points de montage /Volumes/<X> distincts d'une liste de chemins."""
    vols: set[str] = set()
    for p in paths:
        if not p:
            continue
        parts = Path(p).parts
        # /Volumes/<Nom>/...
        if len(parts) >= 3 and parts[1] == "Volumes":
            vols.add(os.path.join("/Volumes", parts[2]))
    return sorted(vols)


def preflight_volumes(
    paths: list[str],
    logger: logging.Logger | None = None,
    fatal: bool = True,
) -> list[str]:
    """Vérifie une seule fois que les volumes nécessaires sont montés.

    Renvoie la liste des volumes MANQUANTS. Si `fatal` et qu'il en manque,
    lève VolumeError après un unique message d'erreur. Sinon, émet un seul
    warning récapitulatif (pas de répétition par fichier).
    """
    log = logger or get_logger()
    missing = [v for v in required_volumes(paths) if not os.path.ismount(v) and not os.path.isdir(v)]
    if not missing:
        return []

    msg = "Volume(s) non monté(s) : %s — détecté au démarrage, vérifié une seule fois." % ", ".join(
        missing
    )
    if fatal:
        log.error(msg)
        raise VolumeError(msg)
    log.warning(msg + " Les sources sur ces volumes seront ignorées.")
    return missing
