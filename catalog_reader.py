"""
catalog_reader.py — Lecture (STRICTEMENT lecture seule) d'un catalogue
Adobe Lightroom Classic (.lrcat) pour le Photo Tagger.

Le catalogue est ouvert via l'URI SQLite `file:...?mode=ro&immutable=1`, ce qui
garantit qu'aucune écriture n'est possible et évite tout verrou même si
Lightroom a laissé des fichiers -wal/-shm. On ne touche JAMAIS la base.

Schéma utile (Lightroom Classic v13-15) :
  Adobe_images            : id_local, id_global (UUID), rootFile, fileFormat,
                            captureTime, orientation
  AgLibraryFile           : id_local, baseName, extension, folder, originalFilename
  AgLibraryFolder         : id_local, pathFromRoot, rootFolder
  AgLibraryRootFolder     : id_local, absolutePath
  AgHarvestedExifMetadata : image, hasGPS, gpsLatitude, gpsLongitude,
                            dateYear, dateMonth, dateDay

L'UUID (id_global) sert de clé commune pour retrouver l'aperçu standard
(previews.db) ou le Smart Preview (.dng nommé d'après l'UUID).
"""

from __future__ import annotations

import sqlite3
# PYTHON — `dataclass` : décorateur qui génère automatiquement __init__,
# __repr__, __eq__... à partir des champs déclarés (voir PhotoRecord ci-dessous).
from dataclasses import dataclass
from pathlib import Path


# PYTHON — DÉCORATEUR `@dataclass` : un décorateur est une fonction qui « enveloppe »
# la classe/fonction suivante pour la transformer. @dataclass lit les annotations de
# champs du corps de la classe et génère le boilerplate. Ici, déclarer `uuid: str`
# etc. SUFFIT : @dataclass crée le constructeur PhotoRecord(uuid=..., file_uuid=...).
# C'est l'équivalent d'un `record`/`struct`/POJO. Contrairement à une classe
# normale, les champs SONT déclarés dans le corps (avec leur type).
@dataclass
class PhotoRecord:
    """Une photo du catalogue, telle que nécessaire pour le tagging."""

    # Chaque ligne = un champ + son type. `str | None` (= Optional[str]) signifie
    # « une chaîne OU None (null) ». Python n'impose pas le type, c'est documentaire.
    uuid: str                 # Adobe_images.id_global (clé des aperçus standard previews.db)
    file_uuid: str            # AgLibraryFile.id_global (clé des Smart Previews .dng)
    image_id: int             # Adobe_images.id_local
    base_name: str            # nom de fichier sans extension
    extension: str            # extension d'origine (NEF, CR2, jpg, ...)
    folder_abs: str           # chemin absolu du dossier (root + pathFromRoot)
    file_format: str | None   # RAW, JPG, ... (Adobe_images.fileFormat)
    has_gps: bool
    gps_lat: float | None
    gps_lon: float | None
    year: int | None
    month: int | None
    day: int | None

    # PYTHON — `@property` : transforme une méthode en attribut CALCULÉ en lecture
    # seule. On écrit `rec.original_path` (SANS parenthèses), pas
    # `rec.original_path()`. C'est un « getter » implicite (comme une propriété C#).
    @property
    def original_path(self) -> str:
        """Chemin attendu du fichier original sur disque."""
        # PYTHON — l'opérateur `/` est SURCHARGÉ sur les Path : Path("a") / "b.jpg"
        # construit le chemin "a/b.jpg" (séparateur géré selon l'OS). Bien plus sûr
        # qu'une concaténation de strings. f"...{x}..." = f-STRING : interpolation
        # de variables/expressions directement dans la chaîne (comme `${x}`).
        return str(Path(self.folder_abs) / f"{self.base_name}.{self.extension}")

    @property
    def display_name(self) -> str:
        return f"{self.base_name}.{self.extension}"

    @property
    def xmp_path(self) -> str:
        """Chemin du sidecar XMP à côté de l'original."""
        return str(Path(self.folder_abs) / f"{self.base_name}.xmp")


class CatalogReader:
    """Accès lecture seule au catalogue Lightroom."""

    def __init__(self, lrcat_path: str | Path, immutable: bool = True):
        self.lrcat_path = Path(lrcat_path)
        if not self.lrcat_path.is_file():
            # PYTHON — `raise` lève une exception (comme throw). FileNotFoundError
            # est une exception standard. Pas de checked exceptions en Python : on
            # ne déclare pas ce qu'une fonction peut lever.
            raise FileNotFoundError(f"Catalogue introuvable : {self.lrcat_path}")
        # immutable=1 => aucun verrou, lecture la plus rapide, MAIS le cache est
        # figé : à éviter si un CatalogWriter écrit le même fichier en parallèle.
        # Dans ce cas on ouvre en RO simple (immutable=False) pour voir les écritures.
        if immutable:
            uri = f"file:{self.lrcat_path}?mode=ro&immutable=1"
        else:
            uri = f"file:{self.lrcat_path}?mode=ro"
        # uri=True : interpréter le 1er argument comme une URI SQLite (et non un
        # simple chemin de fichier), ce qui permet les paramètres ?mode=ro...
        self.conn = sqlite3.connect(uri, uri=True)
        # row_factory = sqlite3.Row : chaque ligne devient un objet indexable par
        # NOM de colonne (row["uuid"]) en plus de l'index (row[0]) — comme un dict.
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "CatalogReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Chemins liés au catalogue -----------------------------------------

    @property
    def previews_dir(self) -> Path:
        """<Catalogue> Previews.lrdata (aperçus standard)."""
        # PYTHON — méthodes Path utiles : .stem = nom de fichier SANS extension
        # ("LR-v15" pour "LR-v15.lrcat"), .with_name(x) = remplace le dernier
        # segment du chemin par x (en gardant le dossier parent).
        return self.lrcat_path.with_name(self.lrcat_path.stem + " Previews.lrdata")

    @property
    def smart_previews_dir(self) -> Path:
        """<Catalogue> Smart Previews.lrdata (.dng)."""
        return self.lrcat_path.with_name(
            self.lrcat_path.stem + " Smart Previews.lrdata"
        )

    @property
    def cloud_smart_previews_dir(self) -> Path:
        """Smart Previews CLOUD (catalogue mobile) : Mobile Downloads.lrdata/
        downloaded-smart-previews/ (fichiers nommés d'après le nom original)."""
        return self.lrcat_path.with_name("Mobile Downloads.lrdata") / "downloaded-smart-previews"

    # -- Requêtes ----------------------------------------------------------

    # PYTHON — ATTRIBUT DE CLASSE : défini directement dans le corps de la classe
    # (pas dans __init__, pas de `self`). Il est PARTAGÉ par toutes les instances
    # (~static/const). Ici une constante SQL réutilisée par plusieurs méthodes.
    _BASE_QUERY = """
        SELECT
            i.id_global              AS uuid,
            fl.id_global             AS file_uuid,
            i.id_local               AS image_id,
            fl.baseName              AS base_name,
            fl.extension             AS extension,
            i.fileFormat             AS file_format,
            rf.absolutePath          AS root_abs,
            f.pathFromRoot           AS path_from_root,
            COALESCE(ex.hasGPS, 0)   AS has_gps,
            ex.gpsLatitude           AS gps_lat,
            ex.gpsLongitude          AS gps_lon,
            ex.dateYear              AS year,
            ex.dateMonth             AS month,
            ex.dateDay               AS day
        FROM Adobe_images i
        JOIN AgLibraryFile fl        ON fl.id_local = i.rootFile
        JOIN AgLibraryFolder f       ON f.id_local = fl.folder
        JOIN AgLibraryRootFolder rf  ON rf.id_local = f.rootFolder
        LEFT JOIN AgHarvestedExifMetadata ex ON ex.image = i.id_local
    """

    def _row_to_record(self, r: sqlite3.Row) -> PhotoRecord:
        # PYTHON — `a or b` ne renvoie PAS un booléen mais la 1re valeur
        # « vraie » : si r["root_abs"] est None/"" (falsy), on prend "". Idiome
        # courant pour fournir une valeur par défaut (équiv. `?? ""`).
        folder_abs = (r["root_abs"] or "") + (r["path_from_root"] or "")
        # pathFromRoot se termine par '/', on retire le slash final pour Path.
        folder_abs = folder_abs.rstrip("/")
        # PYTHON — appel avec ARGUMENTS NOMMÉS (uuid=..., file_uuid=...) : rend
        # l'appel auto-documenté et insensible à l'ordre. Possible sur tout
        # constructeur/fonction. bool(x) convertit explicitement (0/1 -> False/True).
        return PhotoRecord(
            uuid=r["uuid"],
            file_uuid=r["file_uuid"],
            image_id=r["image_id"],
            base_name=r["base_name"],
            extension=r["extension"] or "",
            folder_abs=folder_abs,
            file_format=r["file_format"],
            has_gps=bool(r["has_gps"]),
            gps_lat=r["gps_lat"],
            gps_lon=r["gps_lon"],
            year=r["year"],
            month=r["month"],
            day=r["day"],
        )

    def count_in_scope(self, folder_substring: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM (" + self._BASE_QUERY + ") t"
        params: tuple = ()
        if folder_substring:
            sql = (
                "SELECT COUNT(*) FROM Adobe_images i "
                "JOIN AgLibraryFile fl ON fl.id_local=i.rootFile "
                "JOIN AgLibraryFolder f ON f.id_local=fl.folder "
                "JOIN AgLibraryRootFolder rf ON rf.id_local=f.rootFolder "
                "WHERE (rf.absolutePath || f.pathFromRoot) LIKE ?"
            )
            params = (f"%{folder_substring}%",)
        return self.conn.execute(sql, params).fetchone()[0]

    def selected_image_ids(self) -> list[int]:
        """Renvoie les id_local de la sélection courante (persistée par Lightroom).

        Lightroom stocke la sélection active dans Adobe_variablesTable sous la clé
        Adobe_selectedImages (id_local séparés par des virgules). Disponible même
        Lightroom fermé, tant que le catalogue a été enregistré avec la sélection.
        """
        row = self.conn.execute(
            "SELECT value FROM Adobe_variablesTable WHERE name = 'Adobe_selectedImages'"
        ).fetchone()
        # PYTHON — `not row` est vrai si row est None (aucune ligne) ; `not row[0]`
        # si la valeur est vide. `[]` = liste vide littérale (retour anticipé).
        if not row or not row[0]:
            return []
        # PYTHON — import LOCAL (dans une fonction) : `re` (regex) n'est chargé que
        # si on arrive ici. Légitime pour un module lourd/rarement utilisé.
        import re

        # PYTHON — LIST COMPREHENSION : [ expr for x in iterable ]. Équivaut à une
        # boucle qui construit une liste. re.findall(r"\d+", s) renvoie toutes les
        # suites de chiffres ; on les convertit en int. `r"..."` = RAW STRING
        # (backslashes littéraux, indispensable pour les regex).
        return [int(x) for x in re.findall(r"\d+", str(row[0]))]

    def iter_photos(
        self,
        folder_substring: str | None = None,
        gps_only: bool = False,
        limit: int | None = None,
        image_ids: list[int] | None = None,
        with_smart_preview_only: bool = False,
    ):
        """Itère les PhotoRecord du catalogue.

        - folder_substring : restreint au sous-arbre dont le chemin absolu
          contient cette sous-chaîne (le « périmètre »).
        - gps_only : ne renvoie que les photos géolocalisées.
        - image_ids : restreint à ces Adobe_images.id_local (ex. sélection courante).
        - with_smart_preview_only : ne renvoie que les photos ayant réellement un
          Smart Preview (présentes dans AgDNGProxyInfo). Évite de traiter en
          priorité des photos cloud (Mobile Downloads) dont les pixels ne sont
          pas sur le disque et qui, par tri alphabétique, passent en tête.
        - limit : limite le nombre de résultats (utile pour les tests).
        """
        # On construit dynamiquement la clause WHERE : une liste de fragments SQL
        # et une liste parallèle de paramètres `?`.
        sql = self._BASE_QUERY
        where = []
        params: list = []
        if folder_substring:
            where.append("(rf.absolutePath || f.pathFromRoot) LIKE ?")
            params.append(f"%{folder_substring}%")
        if gps_only:
            where.append("COALESCE(ex.hasGPS, 0) = 1")
        if image_ids:
            # PYTHON — "?" * n RÉPÈTE la chaîne n fois ("???"), puis ",".join(...)
            # insère une virgule entre chaque caractère -> "?,?,?". On génère ainsi
            # autant de placeholders que d'ids. .extend() ajoute tous les éléments
            # d'une liste à une autre (vs .append() qui ajoute UN élément).
            placeholders = ",".join("?" * len(image_ids))
            where.append(f"i.id_local IN ({placeholders})")
            params.extend(image_ids)
        if with_smart_preview_only:
            where.append(
                "fl.id_global IN (SELECT fileUUID FROM AgDNGProxyInfo)"
            )
        # `if where:` est vrai si la liste est NON vide (une liste vide est falsy).
        if where:
            # " AND ".join(liste) assemble les fragments avec " AND " entre eux.
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY rf.absolutePath, f.pathFromRoot, fl.baseName"
        if limit:
            sql += f" LIMIT {int(limit)}"

        # PYTHON — GÉNÉRATEUR : cette fonction contient `yield`, donc l'appeler ne
        # l'EXÉCUTE PAS immédiatement — elle renvoie un itérateur paresseux. À
        # chaque tour de boucle de l'appelant, le code reprend jusqu'au prochain
        # `yield`, qui produit une valeur. Avantage ÉNORME ici : on ne charge
        # jamais 200 000 PhotoRecord en mémoire ; on en produit un à la fois,
        # à la demande. C'est l'équivalent de IEnumerable/yield return en C#.
        for r in self.conn.execute(sql, params):
            yield self._row_to_record(r)

    def existing_keywords(self, image_id: int) -> list[str]:
        """Mots-clés déjà associés à une image (pour détecter un déjà-taggué)."""
        cur = self.conn.execute(
            """
            SELECT k.name
            FROM AgLibraryKeyword k
            JOIN AgLibraryKeywordImage ki ON k.id_local = ki.tag
            WHERE ki.image = ?
            """,
            (image_id,),
        )
        return [r[0] for r in cur.fetchall() if r[0]]

    def list_folders(self, substring: str | None = None, limit: int = 50):
        """Liste (chemin_absolu, nb_photos) pour aider au choix du périmètre."""
        sql = """
            SELECT (rf.absolutePath || f.pathFromRoot) AS p, COUNT(*) AS n
            FROM AgLibraryFolder f
            JOIN AgLibraryRootFolder rf ON rf.id_local = f.rootFolder
            JOIN AgLibraryFile fl ON fl.folder = f.id_local
        """
        params: list = []
        if substring:
            sql += " WHERE (rf.absolutePath || f.pathFromRoot) LIKE ?"
            params.append(f"%{substring}%")
        sql += " GROUP BY f.id_local ORDER BY n DESC LIMIT ?"
        params.append(limit)
        return [(r["p"], r["n"]) for r in self.conn.execute(sql, params)]
