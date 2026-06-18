"""
session_cache.py — Reprise de session pour les gros runs.

Mémorise dans une petite base SQLite les photos DÉJÀ TRAITÉES (par UUID de
fichier, stable), avec le nombre de tags produits et l'horodatage. Permet de
reprendre un run interrompu (plusieurs dizaines de milliers de photos) sans
re-soumettre au LLM celles qui sont faites.

Indépendant du catalogue Lightroom : la base de session vit dans le dossier de
sortie (`out_dir/session.db`), donc on peut avoir une session par run ou la
réutiliser en pointant le même dossier de sortie.

Différence avec le « skip déjà-tagué » : ce dernier regarde les mots-clés
présents dans le catalogue/XMP (état durable) ; la session, elle, mémorise ce
que CE pipeline a traité, y compris les photos sans tag produit (pour ne pas
les re-soumettre inutilement) et indépendamment de toute écriture.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class SessionCache:
    """Index SQLite des photos déjà traitées (reprise de run)."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed (
                file_uuid TEXT PRIMARY KEY,
                name      TEXT,
                n_tags    INTEGER,
                ts        REAL
            )
            """
        )
        self.conn.commit()
        # Cache mémoire des UUID déjà vus, pour un test d'appartenance rapide.
        self._done: set[str] = {
            r[0] for r in self.conn.execute("SELECT file_uuid FROM processed")
        }

    def is_done(self, file_uuid: str) -> bool:
        return file_uuid in self._done

    def mark(self, file_uuid: str, name: str, n_tags: int, commit: bool = True) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO processed (file_uuid, name, n_tags, ts) "
            "VALUES (?, ?, ?, ?)",
            (file_uuid, name, n_tags, time.time()),
        )
        self._done.add(file_uuid)
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def count(self) -> int:
        return len(self._done)

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "SessionCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
