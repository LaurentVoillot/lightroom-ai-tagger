"""
writers.py — Écriture NON DESTRUCTIVE des tags : sidecars XMP et/ou base LrC.

Deux cibles indépendantes, activables séparément :

  XmpWriter      : écrit/complète un sidecar .xmp à côté de l'image.
  CatalogWriter  : écrit les mots-clés directement dans le catalogue .lrcat.

Règles communes (demande de Laurent) :
  - On N'EFFACE JAMAIS les tags existants.
  - On n'ajoute un tag que s'il n'est pas déjà présent (avec ou sans suffixe) :
    « herbe » et « herbe_AI » sont considérés comme le même tag.
  - Suffixe configurable (défaut « _AI »), pouvant être vide.

Logique XMP reprise de photo-folder-tagger/xmp_manager.py ; logique catalogue
reprise de photo-auto-tagger-AI/lightroom_manager.py (add_tags /
_get_or_create_keyword), adaptées à ce projet.

⚠️ CatalogWriter : écrire dans un .lrcat OUVERT par Lightroom corromprait la
base. On refuse donc d'écrire si un verrou est détecté (cf. catalog_is_locked).
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from lxml import etree

from log_panel import get_logger

# --------------------------------------------------------------------------
# Utilitaire commun : forme « nue » d'un tag (sans suffixe, minuscule)
# --------------------------------------------------------------------------


def bare_tag(tag: str, suffix: str) -> str:
    """Renvoie le tag sans son suffixe, en minuscules (clé de déduplication)."""
    tl = tag.lower()
    if suffix and tl.endswith(suffix.lower()):
        return tl[: -len(suffix)]
    return tl


# --------------------------------------------------------------------------
# Détection de verrou catalogue (sécurité avant écriture dans le .lrcat)
# --------------------------------------------------------------------------


def catalog_is_locked(lrcat_path: str | Path) -> bool:
    """Vrai si Lightroom semble utiliser ce catalogue (écriture dangereuse).

    Double vérification : présence d'un fichier `.lrcat.lock`, et tentative
    d'acquérir un verrou d'écriture SQLite (BEGIN IMMEDIATE) sans rien modifier.
    """
    p = Path(lrcat_path)
    if p.with_suffix(p.suffix + ".lock").exists():
        return True
    try:
        con = sqlite3.connect(str(p), timeout=1.0)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.rollback()
            return False
        finally:
            con.close()
    except sqlite3.OperationalError:
        return True


# --------------------------------------------------------------------------
# XMP sidecar
# --------------------------------------------------------------------------

_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}

_XMP_TEMPLATE = """\
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-tagger">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


class XmpWriter:
    """Écrit/complète les tags dans un sidecar .xmp (non destructif)."""

    def __init__(self, suffix: str = "_AI"):
        self.suffix = suffix
        self.log = get_logger()

    @staticmethod
    def xmp_path_for(image_path: str | Path) -> Path:
        return Path(image_path).with_suffix(".xmp")

    def read_tags(self, xmp_path: Path) -> list[str]:
        if not xmp_path.exists():
            return []
        try:
            root = etree.parse(str(xmp_path)).getroot()
        except Exception as e:
            self.log.warning("XMP illisible %s (%s)", xmp_path.name, e)
            return []
        dc, rdf = _NS["dc"], _NS["rdf"]
        tags: list[str] = []
        for subject in root.iter(f"{{{dc}}}subject"):
            for container in (f"{{{rdf}}}Bag", f"{{{rdf}}}Seq"):
                for cont in subject.iter(container):
                    for li in cont.iter(f"{{{rdf}}}li"):
                        if li.text and li.text.strip():
                            tags.append(li.text.strip())
        return tags

    def write_tags(self, xmp_path: Path, ai_tags: list[str]) -> int:
        """Fusionne ai_tags (suffixés) avec l'existant. Renvoie le nb ajouté."""
        if not ai_tags:
            return 0
        suffixed = [
            f"{t}{self.suffix}" if self.suffix else t for t in ai_tags
        ]
        existing = self.read_tags(xmp_path)
        seen = {bare_tag(t, self.suffix) for t in existing}
        added = 0
        final = list(existing)
        for tag in suffixed:
            b = bare_tag(tag, self.suffix)
            if b not in seen:
                final.append(tag)
                seen.add(b)
                added += 1
        if added == 0:
            return 0

        # Charge ou crée l'arbre, puis remplace le bloc dc:subject par final.
        if xmp_path.exists():
            try:
                parser = etree.XMLParser(recover=True, encoding="UTF-8")
                tree = etree.parse(str(xmp_path), parser)
                root = tree.getroot()
            except Exception:
                root = etree.fromstring(_XMP_TEMPLATE.encode("UTF-8"))
                tree = etree.ElementTree(root)
        else:
            root = etree.fromstring(_XMP_TEMPLATE.encode("UTF-8"))
            tree = etree.ElementTree(root)

        if not self._set_subject(root, final):
            return 0
        try:
            tree.write(str(xmp_path), xml_declaration=False,
                       encoding="UTF-8", pretty_print=True)
            self._wrap_xpacket(xmp_path)
        except Exception as e:
            self.log.error("Écriture XMP échouée %s (%s)", xmp_path.name, e)
            return 0
        return added

    def _set_subject(self, root, tags: list[str]) -> bool:
        dc, rdf = _NS["dc"], _NS["rdf"]
        rdf_rdf = next(root.iter(f"{{{rdf}}}RDF"), None)
        if rdf_rdf is None:
            return False
        desc = rdf_rdf.find(f"{{{rdf}}}Description")
        if desc is None:
            desc = etree.SubElement(rdf_rdf, f"{{{rdf}}}Description")
            desc.set(f"{{{rdf}}}about", "")
        for s in desc.findall(f"{{{dc}}}subject"):
            desc.remove(s)
        subject = etree.SubElement(desc, f"{{{dc}}}subject")
        bag = etree.SubElement(subject, f"{{{rdf}}}Bag")
        for t in tags:
            etree.SubElement(bag, f"{{{rdf}}}li").text = t
        return True

    @staticmethod
    def _wrap_xpacket(xmp_path: Path) -> None:
        content = xmp_path.read_text(encoding="UTF-8")
        if content.startswith("<?xml"):
            content = content[content.index("?>") + 2:].lstrip()
        if "<?xpacket" not in content:
            header = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            footer = '\n<?xpacket end="w"?>'
            xmp_path.write_text(header + content + footer, encoding="UTF-8")


# --------------------------------------------------------------------------
# Catalogue Lightroom (.lrcat)
# --------------------------------------------------------------------------


class CatalogWriter:
    """Écrit les mots-clés dans le catalogue .lrcat (non destructif).

    À n'utiliser que LIGHTROOM FERMÉ : le constructeur refuse d'ouvrir un
    catalogue verrouillé.
    """

    def __init__(self, lrcat_path: str | Path, suffix: str = "_AI"):
        self.lrcat_path = Path(lrcat_path)
        self.suffix = suffix
        self.log = get_logger()
        if catalog_is_locked(self.lrcat_path):
            raise RuntimeError(
                "Catalogue verrouillé : ferme Lightroom avant d'écrire dans la base."
            )
        # Ouverture en lecture/écriture (Lightroom est fermé).
        self.conn = sqlite3.connect(str(self.lrcat_path), timeout=5.0)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "CatalogWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def existing_tags(self, image_id: int) -> list[str]:
        cur = self.conn.execute(
            """
            SELECT DISTINCT k.name
            FROM AgLibraryKeyword k
            JOIN AgLibraryKeywordImage ki ON k.id_local = ki.tag
            WHERE ki.image = ?
            """,
            (image_id,),
        )
        return [r[0] for r in cur.fetchall() if r[0]]

    def _get_or_create_keyword(self, cur: sqlite3.Cursor, name: str) -> int | None:
        cur.execute("SELECT id_local FROM AgLibraryKeyword WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        id_global = str(uuid.uuid4()).upper()
        cur.execute("SELECT MAX(id_local) FROM AgLibraryKeyword")
        new_id = (cur.fetchone()[0] or 0) + 1
        cur.execute(
            """
            INSERT INTO AgLibraryKeyword
                (id_local, id_global, name, lc_name, dateCreated, genealogy)
            VALUES (?, ?, ?, ?, datetime('now'), ?)
            """,
            (new_id, id_global, name, name.lower(), f"/{name}"),
        )
        return new_id

    def add_tags(self, image_id: int, ai_tags: list[str]) -> int:
        """Ajoute les tags (suffixés) à une image, sans doublon. Renvoie nb ajouté."""
        if not ai_tags:
            return 0
        existing = self.existing_tags(image_id)
        seen = {bare_tag(t, self.suffix) for t in existing}
        cur = self.conn.cursor()
        added = 0
        try:
            cur.execute("BEGIN")
            for tag in ai_tags:
                final_name = f"{tag}{self.suffix}" if self.suffix else tag
                if bare_tag(final_name, self.suffix) in seen:
                    continue
                kw_id = self._get_or_create_keyword(cur, final_name)
                if kw_id is None:
                    continue
                cur.execute(
                    "SELECT 1 FROM AgLibraryKeywordImage WHERE image = ? AND tag = ?",
                    (image_id, kw_id),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO AgLibraryKeywordImage (image, tag) VALUES (?, ?)",
                    (image_id, kw_id),
                )
                seen.add(bare_tag(final_name, self.suffix))
                added += 1
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.log.error("Écriture catalogue échouée (image %s) : %s", image_id, e)
            return 0
        return added
