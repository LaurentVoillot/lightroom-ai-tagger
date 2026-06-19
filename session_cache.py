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

# PYTHON — `from __future__ import annotations` : rend TOUTES les annotations de
# type « paresseuses » (évaluées comme des chaînes, jamais à l'exécution). Effet
# pratique : on peut écrire `str | Path` (syntaxe union, normalement Python 3.10+)
# ou référencer une classe avant sa définition, sur des versions plus anciennes.
# À mettre en 1re ligne de code. C'est purement cosmétique/typage : Python ne
# vérifie PAS les types au runtime (contrairement à un langage compilé typé).
from __future__ import annotations

# PYTHON — pas de `#include` / `using` : `import X` charge le module X (un autre
# fichier .py ou une lib standard) et expose son namespace sous le nom `X`.
# sqlite3, time, pathlib font partie de la bibliothèque standard (rien à installer).
import sqlite3
import time
from pathlib import Path  # `from M import N` : importe juste le nom N de M.


# PYTHON — `class` sans liste de membres déclarés : en Python les attributs
# d'instance ne sont PAS déclarés dans le corps de la classe (pas de champs comme
# en C++/Java). On les crée à la volée en les affectant sur `self`, typiquement
# dans __init__. Tout est public par convention ; un `_` en préfixe (ex. _done)
# signale « privé/interne » mais n'est pas réellement protégé par le langage.
class SessionCache:
    """Index SQLite des photos déjà traitées (reprise de run)."""

    # PYTHON — __init__ est le CONSTRUCTEUR. `self` = le pointeur sur l'instance
    # (équivalent de `this`), mais ici il est EXPLICITE : 1er paramètre de toute
    # méthode d'instance. L'annotation `db_path: str | Path` accepte une string
    # OU un objet Path (union de types) ; `-> None` plus bas = type de retour.
    def __init__(self, db_path: str | Path):
        # Path() = objet chemin orienté objet (comme java.nio.Path). Path(x) est
        # idempotent : si x est déjà un Path, on récupère le même type.
        self.path = Path(db_path)
        # .parent = dossier conteneur ; mkdir avec parents=True crée toute
        # l'arborescence manquante, exist_ok=True n'échoue pas si elle existe.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # sqlite3.connect veut une str -> conversion explicite str(Path).
        self.conn = sqlite3.connect(str(self.path))
        # PYTHON — chaîne triple-guillemets """...""" : littéral multiligne
        # (équivalent d'un here-doc / texte brut), pratique pour du SQL.
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
        # PYTHON — SET COMPREHENSION : { expr for x in iterable }. Construit un
        # `set` (collection de valeurs uniques, ~HashSet) en une expression.
        # Ici : pour chaque ligne r renvoyée par la requête, on prend r[0]
        # (1re colonne). L'annotation `: set[str]` est indicative seulement.
        # Le `self.conn.execute(...)` est directement itérable (il yield les lignes).
        self._done: set[str] = {
            r[0] for r in self.conn.execute("SELECT file_uuid FROM processed")
        }

    # PYTHON — `in` sur un set est O(1) (test d'appartenance par hachage). On
    # garde _done en mémoire pour éviter une requête SQL à chaque vérification.
    def is_done(self, file_uuid: str) -> bool:
        return file_uuid in self._done

    # PYTHON — `commit: bool = True` : paramètre avec VALEUR PAR DÉFAUT. L'appelant
    # peut l'omettre. On peut aussi le nommer à l'appel : mark(uuid, n, commit=False)
    # (arguments « par mot-clé », très courant en Python).
    def mark(self, file_uuid: str, name: str, n_tags: int, commit: bool = True) -> None:
        # PYTHON — requête PARAMÉTRÉE : les `?` sont remplacés par le tuple
        # (file_uuid, name, ...) -> protège des injections SQL. NE JAMAIS
        # concaténer des valeurs dans la requête. time.time() = timestamp Unix float.
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

    # len() sur un set = nombre d'éléments, en O(1).
    def count(self) -> int:
        return len(self._done)

    def close(self) -> None:
        # PYTHON — try/except : gestion d'erreurs (comme try/catch). `except
        # Exception` attrape toute erreur « normale ». Ici on ignore silencieusement
        # un échec de fermeture (best-effort) — acceptable pour un close().
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass  # `pass` = instruction vide (corps obligatoire mais on ne fait rien).

    # PYTHON — PROTOCOLE « CONTEXT MANAGER » : __enter__/__exit__ permettent
    # d'écrire `with SessionCache(path) as sc: ...`. À l'entrée du bloc, __enter__
    # est appelé (sa valeur de retour est liée à `sc`) ; à la sortie — même en cas
    # d'exception — __exit__ est appelé. C'est le RAII de Python : garantit la
    # fermeture des ressources. Idéal pour fichiers, connexions, locks.
    def __enter__(self) -> "SessionCache":
        return self

    # `*exc` capture les 3 args que Python passe à __exit__ (type, valeur,
    # traceback de l'exception, ou None x3 si sortie normale). `*` = « le reste
    # des arguments positionnels dans un tuple » (équivalent varargs).
    def __exit__(self, *exc) -> None:
        self.close()
