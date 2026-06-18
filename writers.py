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


def _as_path(tag) -> list[str]:
    """Normalise un tag en chemin hiérarchique (liste de niveaux).

    Accepte : une liste/tuple de niveaux, ou une string. Une string contenant
    '>' est découpée en niveaux ('Lieu>France>Isère'). Sinon, tag plat.
    """
    if isinstance(tag, (list, tuple)):
        parts = [str(p).strip() for p in tag if str(p).strip()]
        return parts or ["?"]
    s = str(tag).strip()
    if ">" in s:
        parts = [p.strip() for p in s.split(">") if p.strip()]
        return parts or ["?"]
    return [s]


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
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}

_XMP_TEMPLATE = """\
<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-tagger">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
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

    def write_tags(self, xmp_path: Path, ai_tags: list) -> int:
        """Fusionne ai_tags avec l'existant (non destructif). Renvoie le nb ajouté.

        ai_tags : chaque élément est un tag plat (str), un chemin 'a>b>c' (str)
        ou une liste de niveaux. Le suffixe est appliqué à la FEUILLE.
        On écrit dc:subject (feuilles plates, pour la recherche) ET
        lr:hierarchicalSubject (chemins complets, pour l'arbre Lightroom).
        """
        if not ai_tags:
            return 0

        # Normalise chaque tag en chemin, puis applique le suffixe à la feuille.
        new_paths: list[list[str]] = []
        for tag in ai_tags:
            path = _as_path(tag)
            if self.suffix:
                path = path[:-1] + [f"{path[-1]}{self.suffix}"]
            new_paths.append(path)

        # Existant : on lit les feuilles plates pour dédupliquer.
        existing_flat = self.read_tags(xmp_path)
        existing_hier = self._read_hierarchical(xmp_path)
        seen = {bare_tag(t, self.suffix) for t in existing_flat}

        final_flat = list(existing_flat)
        final_hier = list(existing_hier)
        added = 0
        for path in new_paths:
            leaf = path[-1]
            if bare_tag(leaf, self.suffix) in seen:
                continue
            final_flat.append(leaf)
            final_hier.append("|".join(path))
            seen.add(bare_tag(leaf, self.suffix))
            added += 1
        if added == 0:
            return 0

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

        ok = self._set_bag(root, _NS["dc"], "subject", final_flat)
        # hierarchicalSubject n'a de sens que si au moins un chemin a >1 niveau.
        if any("|" in h for h in final_hier):
            ok = self._set_bag(root, _NS["lr"], "hierarchicalSubject", final_hier) and ok
        if not ok:
            return 0
        try:
            tree.write(str(xmp_path), xml_declaration=False,
                       encoding="UTF-8", pretty_print=True)
            self._wrap_xpacket(xmp_path)
        except Exception as e:
            self.log.error("Écriture XMP échouée %s (%s)", xmp_path.name, e)
            return 0
        return added

    def _read_hierarchical(self, xmp_path: Path) -> list[str]:
        if not xmp_path.exists():
            return []
        try:
            root = etree.parse(str(xmp_path)).getroot()
        except Exception:
            return []
        lr, rdf = _NS["lr"], _NS["rdf"]
        out: list[str] = []
        for hs in root.iter(f"{{{lr}}}hierarchicalSubject"):
            for li in hs.iter(f"{{{rdf}}}li"):
                if li.text and li.text.strip():
                    out.append(li.text.strip())
        return out

    def _set_bag(self, root, ns: str, tag_name: str, values: list[str]) -> bool:
        rdf = _NS["rdf"]
        rdf_rdf = next(root.iter(f"{{{rdf}}}RDF"), None)
        if rdf_rdf is None:
            return False
        desc = rdf_rdf.find(f"{{{rdf}}}Description")
        if desc is None:
            desc = etree.SubElement(rdf_rdf, f"{{{rdf}}}Description")
            desc.set(f"{{{rdf}}}about", "")
        for s in desc.findall(f"{{{ns}}}{tag_name}"):
            desc.remove(s)
        container = etree.SubElement(desc, f"{{{ns}}}{tag_name}")
        bag = etree.SubElement(container, f"{{{rdf}}}Bag")
        for v in values:
            etree.SubElement(bag, f"{{{rdf}}}li").text = v
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

    @staticmethod
    def _genealogy(parent_gen: str | None, new_id: int) -> str:
        """Genealogy Lightroom : chaque ancêtre encodé '/<nb_chiffres><id>'.

        Ex. id 20 -> '/220' ; enfant id 231 sous /220 -> '/220/3231'. Vérifié
        sur le catalogue réel (rebuild == stocké, 100 %).
        """
        seg = f"/{len(str(new_id))}{new_id}"
        return (parent_gen or "") + seg

    def _get_or_create_keyword(
        self, cur: sqlite3.Cursor, name: str, parent_id: int | None,
        parent_gen: str | None,
    ) -> tuple[int, str] | None:
        """Trouve/crée un mot-clé sous un parent donné. Renvoie (id, genealogy)."""
        # Unicité par (name, parent) : un même nom peut exister sous 2 parents.
        if parent_id is None:
            cur.execute(
                "SELECT id_local, genealogy FROM AgLibraryKeyword "
                "WHERE name = ? AND parent IS NULL", (name,))
        else:
            cur.execute(
                "SELECT id_local, genealogy FROM AgLibraryKeyword "
                "WHERE name = ? AND parent = ?", (name, parent_id))
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        id_global = str(uuid.uuid4()).upper()
        cur.execute("SELECT MAX(id_local) FROM AgLibraryKeyword")
        new_id = (cur.fetchone()[0] or 0) + 1
        gen = self._genealogy(parent_gen, new_id)
        cur.execute(
            """
            INSERT INTO AgLibraryKeyword
                (id_local, id_global, name, lc_name, dateCreated, genealogy, parent)
            VALUES (?, ?, ?, ?, datetime('now'), ?, ?)
            """,
            (new_id, id_global, name, name.lower(), gen, parent_id),
        )
        return new_id, gen

    def _resolve_path(self, cur: sqlite3.Cursor, path: list[str]) -> int | None:
        """Crée la chaîne de mots-clés d'un chemin hiérarchique. Renvoie l'id feuille.

        Le suffixe IA n'est appliqué QU'À LA FEUILLE (les niveaux parents restent
        des catégories propres, partagées avec d'éventuels tags manuels).
        """
        parent_id: int | None = None
        parent_gen: str | None = None
        leaf_id: int | None = None
        for i, level in enumerate(path):
            is_leaf = i == len(path) - 1
            name = f"{level}{self.suffix}" if (is_leaf and self.suffix) else level
            res = self._get_or_create_keyword(cur, name, parent_id, parent_gen)
            if res is None:
                return None
            parent_id, parent_gen = res
            leaf_id = res[0]
        return leaf_id

    def add_tags(self, image_id: int, ai_tags, commit: bool = True) -> int:
        """Ajoute des tags à une image, sans doublon. Renvoie le nb ajouté.

        ai_tags : liste où chaque élément est soit une string (tag plat), soit
        une liste de niveaux (chemin hiérarchique, ex. ['Lieu','France','Isère']).
        Une string contenant '>' est aussi traitée comme un chemin.

        commit : si False, n'effectue PAS de commit (écriture par lots — voir
        commit_batch()). Permet de regrouper N images dans une seule transaction
        pour de bien meilleures performances sur de gros volumes.
        """
        if not ai_tags:
            return 0
        existing = self.existing_tags(image_id)
        seen = {bare_tag(t, self.suffix) for t in existing}
        cur = self.conn.cursor()
        added = 0
        try:
            for tag in ai_tags:
                path = _as_path(tag)
                leaf = path[-1]
                # dédup sur la feuille (forme nue), comme pour les tags plats.
                if bare_tag(f"{leaf}{self.suffix}", self.suffix) in seen:
                    continue
                kw_id = self._resolve_path(cur, path)
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
                seen.add(bare_tag(f"{leaf}{self.suffix}", self.suffix))
                added += 1
            if commit:
                self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.log.error("Écriture catalogue échouée (image %s) : %s", image_id, e)
            return 0
        return added

    def commit_batch(self) -> None:
        """Valide une série d'add_tags(commit=False). À appeler en fin de lot."""
        try:
            self.conn.commit()
        except sqlite3.Error as e:
            self.conn.rollback()
            self.log.error("Commit du lot catalogue échoué : %s", e)
