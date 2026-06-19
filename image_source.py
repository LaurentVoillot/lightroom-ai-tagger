"""
image_source.py — Résolution de l'image à analyser pour une photo du catalogue,
selon une CASCADE de priorité choisie par l'utilisateur :

    1) Aperçu standard      (Previews.lrdata)        -> JPEG embarqué
    2) Smart Preview        (Smart Previews.lrdata)  -> .dng via rawpy
    3) Fichier original     (chemin catalogue)       -> rawpy / Pillow

On prend la PREMIÈRE source qui fonctionne. Chaque échec de bascule est
journalisé en INFO ; si AUCUNE source n'aboutit, un WARNING est émis et la
photo est ignorée.

Note volumes : les originaux peuvent être sur un volume non monté. Cette
situation est détectée une seule fois au démarrage (cf. log_panel.preflight_volumes) ;
ici on ne ré-émet donc pas de warning « volume non monté » par fichier — on se
contente de constater l'absence du fichier et de passer à la source suivante.

Format des aperçus standard
---------------------------
Les fichiers de niveau (`<uuid>-<digest>_<dim>`) sont des conteneurs Adobe
« AGBC » : un en-tête (~1 Ko) suivi d'un JPEG. On extrait le JPEG en cherchant
le marqueur de début (FFD8FF) et de fin (FFD9). La table `root-pixels.db`
(blob `jpegData`) sert de repli pour le niveau de base.
"""

from __future__ import annotations

import glob   # recherche de fichiers par motif (wildcards), comme un `ls *.txt`
import io     # flux en mémoire (BytesIO = « fichier » RAM, voir plus bas)
import os
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# PYTHON — `from PIL import Image` : PIL = Pillow, la lib d'images (à installer
# via pip, ce n'est pas la stdlib). `Image` est la classe centrale.
from PIL import Image

# Imports de NOS modules (autres fichiers du projet). Python les trouve car ils
# sont dans le même dossier (le « package » courant).
from catalog_reader import PhotoRecord
from log_panel import get_logger

# PYTHON — SET LITTÉRAL : { "a", "b", ... }. Des accolades AVEC des valeurs (et
# non des paires clé:valeur) = un set, pas un dict. Sert ici de table de
# correspondance pour un test d'appartenance O(1) (`ext in _RAW_EXT`).
_RAW_EXT = {
    "nef", "cr2", "cr3", "arw", "raf", "rw2", "dng", "orf", "pef",
    "srw", "raw", "rwl", "iiq", "3fr", "fff", "nrw", "kdc", "dcr",
}


# PYTHON — ENUM qui hérite AUSSI de `str` (héritage multiple : `(str, Enum)`).
# Résultat : chaque membre EST une vraie chaîne ("src_preview" == SourceKind.PREVIEW)
# tout en étant un membre d'énumération typé. Pratique pour sérialiser/loguer
# sans conversion. On y accède par SourceKind.PREVIEW et sa valeur par `.value`.
class SourceKind(str, Enum):
    PREVIEW = "src_preview"
    SMART = "src_smart"
    ORIGINAL = "src_original"


# Dataclass sans @property : juste un conteneur de 3 valeurs (struct).
@dataclass
class ResolvedImage:
    image: Image.Image   # le type est `Image.Image` (classe Image du module Image)
    kind: SourceKind
    path: str          # fichier/source d'où provient l'image (info/debug)


# PYTHON — FONCTION LIBRE (hors classe) : tout n'a pas à être dans une classe en
# Python (contrairement à Java). Une fonction au niveau module est parfaitement
# idiomatique. Le `_` initial = convention « privé au module ».
def _load_raw(path: str, max_long_edge: int = 2048) -> Image.Image:
    """Décode un RAW (ou DNG Smart Preview) via rawpy en RGB Pillow."""
    import rawpy  # import paresseux : lourd, pas requis pour les aperçus standard

    # PYTHON — `with ... as raw:` : context manager (cf. session_cache). rawpy
    # ferme proprement le fichier à la sortie du bloc, même en cas d'erreur.
    with rawpy.imread(path) as raw:
        # PYTHON — arguments NOMMÉS dans un appel multi-lignes : parfaitement
        # lisible et insensible à l'ordre. `rgb` est un tableau numpy (pixels).
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            half_size=True,  # demi-résolution : suffisant et bien plus rapide
        )
    # Image.fromarray = construit une image Pillow depuis le tableau numpy.
    im = Image.fromarray(rgb)
    # .thumbnail((w, h)) redimensionne EN PLACE en gardant le ratio (max w×h).
    # Le tuple (x, y) est un couple immuable, très utilisé pour les dimensions.
    im.thumbnail((max_long_edge, max_long_edge))
    return im


class StandardPreviewSource:
    """Lit les aperçus standard depuis <Catalogue> Previews.lrdata."""

    def __init__(self, previews_dir: Path):
        self.dir = Path(previews_dir)
        self._prev_db = self.dir / "previews.db"
        self._root_db = self.dir / "root-pixels.db"
        # PYTHON — on initialise les connexions à None : elles seront ouvertes
        # PARESSEUSEMENT (lazy) au 1er besoin, voir _pyramid(). Annotation
        # `sqlite3.Connection | None` = « une connexion ou None ».
        self._pconn: sqlite3.Connection | None = None
        self._rconn: sqlite3.Connection | None = None

    @property
    def available(self) -> bool:
        return self._prev_db.is_file()

    # PYTHON — LAZY INITIALIZATION : on n'ouvre la connexion qu'au 1er appel, puis
    # on la réutilise. Le `if self._pconn is None` est le test « pas encore créé ».
    # `is None` (et non `== None`) est l'idiome correct pour tester la nullité.
    def _pyramid(self) -> sqlite3.Connection:
        if self._pconn is None:
            self._pconn = sqlite3.connect(
                f"file:{self._prev_db}?mode=ro&immutable=1", uri=True
            )
        return self._pconn

    def _rootpixels(self) -> sqlite3.Connection | None:
        # `and` court-circuite : si _rconn n'est pas None, on n'évalue pas la suite.
        if self._rconn is None and self._root_db.is_file():
            self._rconn = sqlite3.connect(
                f"file:{self._root_db}?mode=ro&immutable=1", uri=True
            )
        return self._rconn

    # PYTHON — @staticmethod : méthode SANS `self` (ne dépend pas de l'instance).
    # Comme une fonction utilitaire, mais rangée dans la classe par cohérence.
    @staticmethod
    def _extract_jpeg(data: bytes) -> bytes | None:
        # PYTHON — `b"..."` = littéral BYTES (octets bruts), distinct d'une str
        # (texte). `\xff` = un octet en hexa. .find() renvoie l'index ou -1.
        # On cherche les marqueurs JPEG : début FFD8FF, fin FFD9.
        soi = data.find(b"\xff\xd8\xff")
        eoi = data.rfind(b"\xff\xd9")  # rfind = depuis la FIN
        if soi == -1 or eoi == -1 or eoi <= soi:
            return None
        # PYTHON — SLICING : data[debut:fin] extrait une sous-séquence (fin
        # exclue). Marche sur bytes, str, list... `data[soi:eoi+2]` = du début
        # jusqu'à inclure les 2 octets de fin FFD9.
        return data[soi : eoi + 2]

    def load(self, rec: PhotoRecord) -> Image.Image | None:
        if not self.available:
            return None
        # Les aperçus standard sont indexés par AgLibraryFile.id_global
        # (= rec.file_uuid), comme les Smart Previews — PAS par Adobe_images.id_global.
        key = rec.file_uuid
        row = self._pyramid().execute(
            "SELECT digest FROM Pyramid WHERE uuid = ? LIMIT 1", (key,)
        ).fetchone()
        if not row:
            return None
        digest = row[0]

        # Niveaux sur disque : on prend la plus grande dimension disponible.
        d = self.dir / key[0] / key[:4]
        pattern = str(d / f"{key}-{digest}_*")
        files = glob.glob(pattern)
        if files:
            def _dim(f: str) -> int:
                try:
                    return int(f.rsplit("_", 1)[1])
                except ValueError:
                    return 0

            for f in sorted(files, key=_dim, reverse=True):
                try:
                    jpg = self._extract_jpeg(Path(f).read_bytes())
                    if jpg:
                        im = Image.open(io.BytesIO(jpg))
                        im.load()
                        return im.convert("RGB")
                except Exception:
                    continue

        # Repli : blob jpegData du niveau de base.
        rp = self._rootpixels()
        if rp is not None:
            r = rp.execute(
                "SELECT jpegData FROM RootPixels WHERE uuid = ? LIMIT 1",
                (key,),
            ).fetchone()
            if r and r[0]:
                try:
                    im = Image.open(io.BytesIO(r[0]))
                    im.load()
                    return im.convert("RGB")
                except Exception:
                    return None
        return None

    def close(self) -> None:
        for c in (self._pconn, self._rconn):
            if c is not None:
                c.close()


class SmartPreviewSource:
    """Lit les Smart Previews (.dng) depuis <Catalogue> Smart Previews.lrdata.

    Les Smart Previews Lightroom sont des DNG « lossy » que LibRaw/rawpy ne sait
    PAS décoder (LibRawFileUnsupportedError). Le fichier est un TIFF multi-pages
    contenant : un petit thumbnail (~256 px) et l'aperçu pleine taille (~2560 px)
    compressé en JPEG XL (COMPRESSION.JPEGXL_DNG). On décode donc avec tifffile
    (+ imagecodecs pour JPEG XL) en prenant la série de plus grande surface.
    """

    def __init__(self, smart_dir: Path, cloud_dir: Path | None = None):
        self.dir = Path(smart_dir)
        # Dossier des Smart Previews CLOUD (catalogue mobile) : fichiers nommés
        # d'après le nom de fichier ORIGINAL (ex. _DSC6955.NEF), à plat, dans
        # Mobile Downloads.lrdata/downloaded-smart-previews/. Fallback optionnel.
        self.cloud_dir = Path(cloud_dir) if cloud_dir else None

    @property
    def available(self) -> bool:
        return self.dir.is_dir() or (self.cloud_dir is not None and self.cloud_dir.is_dir())

    def _dng_path(self, file_uuid: str) -> Path:
        return self.dir / file_uuid[0] / file_uuid[:4] / f"{file_uuid}.dng"

    def _cloud_path(self, rec: PhotoRecord) -> Path | None:
        """Smart Preview cloud : fichier nommé d'après le nom original."""
        if self.cloud_dir is None or not self.cloud_dir.is_dir():
            return None
        cand = self.cloud_dir / rec.display_name
        return cand if cand.is_file() else None

    @staticmethod
    def _decode_dng(path: str) -> Image.Image | None:
        import numpy as np
        import tifffile

        with tifffile.TiffFile(path) as tf:
            # Choisir la série de plus grande surface (l'aperçu pleine taille).
            best = max(
                tf.series,
                key=lambda s: (s.shape[0] * s.shape[1]) if len(s.shape) >= 2 else 0,
            )
            arr = best.asarray()
        if arr.dtype == np.uint16:
            arr = (arr / 256).astype("uint8")
        im = Image.fromarray(arr)
        return im.convert("RGB")

    def load(self, rec: PhotoRecord) -> Image.Image | None:
        if not self.available:
            return None
        # 1) Smart Preview desktop : .dng nommé d'après AgLibraryFile.id_global
        #    (file_uuid), PAS d'après l'UUID image. 2) sinon Smart Preview cloud.
        p = self._dng_path(rec.file_uuid)
        path = p if p.is_file() else self._cloud_path(rec)
        if path is None:
            return None
        try:
            im = self._decode_dng(str(path))
            if im is not None:
                im.thumbnail((2048, 2048))
                return im
        except Exception as e:
            get_logger().info("%s : Smart Preview illisible (%s)", rec.display_name, e)
        return None


class OriginalFileSource:
    """Lit le fichier original depuis le chemin reconstruit du catalogue."""

    def load(self, rec: PhotoRecord) -> Image.Image | None:
        path = rec.original_path
        if not os.path.isfile(path):
            return None  # absence (volume démonté traité au pré-vol) -> source suivante
        try:
            if rec.extension.lower() in _RAW_EXT:
                return _load_raw(path)
            im = Image.open(path)
            im.load()
            im.thumbnail((2048, 2048))
            return im.convert("RGB")
        except Exception as e:
            get_logger().warning("%s : décodage original échoué (%s)", rec.display_name, e)
            return None


class ImageResolver:
    """Applique la cascade de sources dans l'ordre de priorité demandé."""

    # PYTHON — attribut de classe = TUPLE (parenthèses, immuable). Sert d'ordre
    # par défaut. Un tuple convient pour une séquence figée de constantes.
    DEFAULT_ORDER = (SourceKind.PREVIEW, SourceKind.SMART, SourceKind.ORIGINAL)

    def __init__(
        self,
        previews_dir: Path,
        smart_dir: Path,
        order: tuple[SourceKind, ...] = DEFAULT_ORDER,
        cloud_smart_dir: Path | None = None,
    ):
        self.order = order
        self._standard = StandardPreviewSource(previews_dir)
        self._smart = SmartPreviewSource(smart_dir, cloud_dir=cloud_smart_dir)
        self._original = OriginalFileSource()
        self.log = get_logger()

    def _try(self, kind: SourceKind, rec: PhotoRecord) -> Image.Image | None:
        # PYTHON — `is` compare l'IDENTITÉ (même objet), pas l'égalité. Pour les
        # membres d'Enum (singletons), c'est l'idiome correct. Pas de switch/case
        # natif avant 3.10 ; ici une cascade de if suffit.
        if kind is SourceKind.PREVIEW:
            return self._standard.load(rec)
        if kind is SourceKind.SMART:
            return self._smart.load(rec)
        return self._original.load(rec)

    def resolve(self, rec: PhotoRecord) -> ResolvedImage | None:
        """Renvoie la première image obtenue selon la cascade, ou None."""
        # On essaie chaque source dans l'ordre ; la 1re qui renvoie une image gagne.
        for kind in self.order:
            img = self._try(kind, rec)
            if img is not None:
                # Retour anticipé dès qu'une source aboutit (court-circuite la cascade).
                return ResolvedImage(image=img, kind=kind, path=rec.display_name)
        self.log.warning(
            "%s : AUCUNE source disponible (ni aperçu, ni Smart Preview, ni original) — ignorée",
            f"{Path(rec.folder_abs).name}/{rec.display_name}",
        )
        return None

    def close(self) -> None:
        self._standard.close()
