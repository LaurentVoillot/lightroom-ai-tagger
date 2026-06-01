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
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhotoRecord:
    """Une photo du catalogue, telle que nécessaire pour le tagging."""

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

    @property
    def original_path(self) -> str:
        """Chemin attendu du fichier original sur disque."""
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

    def __init__(self, lrcat_path: str | Path):
        self.lrcat_path = Path(lrcat_path)
        if not self.lrcat_path.is_file():
            raise FileNotFoundError(f"Catalogue introuvable : {self.lrcat_path}")
        uri = f"file:{self.lrcat_path}?mode=ro&immutable=1"
        # immutable=1 => aucun verrou, aucune écriture possible.
        self.conn = sqlite3.connect(uri, uri=True)
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
        return self.lrcat_path.with_name(self.lrcat_path.stem + " Previews.lrdata")

    @property
    def smart_previews_dir(self) -> Path:
        """<Catalogue> Smart Previews.lrdata (.dng)."""
        return self.lrcat_path.with_name(
            self.lrcat_path.stem + " Smart Previews.lrdata"
        )

    # -- Requêtes ----------------------------------------------------------

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
        folder_abs = (r["root_abs"] or "") + (r["path_from_root"] or "")
        # pathFromRoot se termine par '/', on retire le slash final pour Path.
        folder_abs = folder_abs.rstrip("/")
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

    def iter_photos(
        self,
        folder_substring: str | None = None,
        gps_only: bool = False,
        limit: int | None = None,
    ):
        """Itère les PhotoRecord du catalogue.

        - folder_substring : restreint au sous-arbre dont le chemin absolu
          contient cette sous-chaîne (le « périmètre »).
        - gps_only : ne renvoie que les photos géolocalisées.
        - limit : limite le nombre de résultats (utile pour les tests).
        """
        sql = self._BASE_QUERY
        where = []
        params: list = []
        if folder_substring:
            where.append("(rf.absolutePath || f.pathFromRoot) LIKE ?")
            params.append(f"%{folder_substring}%")
        if gps_only:
            where.append("COALESCE(ex.hasGPS, 0) = 1")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY rf.absolutePath, f.pathFromRoot, fl.baseName"
        if limit:
            sql += f" LIMIT {int(limit)}"

        for r in self.conn.execute(sql, params):
            yield self._row_to_record(r)

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
