"""
gps_context.py — Contexte géographique d'une photo à partir de ses coordonnées GPS.

Deux usages, tous deux en mode HYBRIDE (offline par défaut, online activable) :

  1) Tags de lieu (reverse geocoding) :
       - offline : reverse_geocoder (ville/admin/pays la plus proche, instantané,
         base embarquée, aucun réseau).
       - online (option) : Nominatim/OpenStreetMap pour un POI précis
         (parc national, lac, monument). Respecte 1 req/s, met en cache.

  2) Liste d'espèces plausibles localement (pour filtrer la passe 2 BioCLIP) :
       - online : GBIF API (occurrences par boîte géographique), agrégées par
         groupe (oiseaux / insectes / autres animaux).
       - cache par GRILLE : les coordonnées sont arrondies à une cellule
         (~10 km par défaut) ; une requête couvre toutes les photos de la zone.
       - offline / réseau coupé : renvoie None → BioCLIP utilise son vocabulaire
         complet (Europe), sans filtrage.

Le cache est un simple JSON sur disque, robuste aux coupures réseau.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import requests

from log_panel import get_logger

# Classes GBIF (taxonKey racine) utiles pour la passe 2.
GBIF_CLASS_KEYS = {
    "birds": 212,      # Aves
    "insects": 216,    # Insecta
    "mammals": 359,    # Mammalia
    "reptiles": 358,   # Reptilia
    "amphibians": 131, # Amphibia
}

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_GBIF_URL = "https://api.gbif.org/v1/occurrence/search"
_USER_AGENT = "photo-tagger/0.1 (local Lightroom tagging)"

# Noms français des classes taxonomiques (pour la hiérarchie de tags d'espèces).
_CLASS_FR = {
    "Aves": "Oiseaux",
    "Insecta": "Insectes",
    "Mammalia": "Mammifères",
    "Reptilia": "Reptiles",
    "Amphibia": "Amphibiens",
    "Arachnida": "Arachnides",
    "Actinopterygii": "Poissons",
}


# PYTHON — dataclass avec VALEURS PAR DÉFAUT (= None) sur chaque champ : on peut
# construire PlaceTags() sans argument (tous None), ou PlaceTags(country="FR").
# Les champs avec défaut doivent venir APRÈS ceux sans défaut (comme les params).
@dataclass
class PlaceTags:
    """Tags de lieu dérivés des coordonnées."""

    country: str | None = None
    admin1: str | None = None   # région / état
    admin2: str | None = None   # département / comté
    city: str | None = None
    poi: str | None = None      # point d'intérêt (online uniquement)

    def as_tags(self) -> list[str]:
        # [a, b, c] = liste littérale. self.country etc. accèdent aux champs.
        vals = [self.country, self.admin1, self.admin2, self.city, self.poi]
        # PYTHON — idiome de DÉDUPLICATION EN PRÉSERVANT L'ORDRE : un set ne garde
        # pas l'ordre, donc on combine un set `seen` (test rapide « déjà vu ? »)
        # avec une liste `out` (résultat ordonné). `if v and v not in seen` :
        # ignore les None/vides ET les doublons.
        seen: set[str] = set()
        out: list[str] = []
        for v in vals:
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def as_hierarchy(self, root: str = "Lieu") -> list[str] | None:
        """Chemin hiérarchique du lieu : ['Lieu', pays, région, dépt, ville, poi].

        Renvoie None si aucune donnée de lieu. Les niveaux vides ou identiques au
        précédent sont sautés (reverse_geocoder répète parfois admin2 == ville),
        en conservant l'ordre géographique (du plus large au plus précis).
        """
        levels = [root]
        for v in (self.country, self.admin1, self.admin2, self.city, self.poi):
            if v and v.strip() and v.strip().lower() != levels[-1].strip().lower():
                levels.append(v.strip())
        return levels if len(levels) > 1 else None


@dataclass
class SpeciesLists:
    """Listes d'espèces plausibles par groupe (noms scientifiques)."""

    by_group: dict[str, list[str]] = field(default_factory=dict)

    def for_group(self, group: str) -> list[str]:
        return self.by_group.get(group, [])


class GpsContext:
    """Fournit tags de lieu et listes d'espèces, avec caches."""

    def __init__(
        self,
        cache_dir: str | Path,
        online_place: bool = False,
        online_species: bool = True,
        grid_deg: float = 0.1,   # ~11 km en latitude
        species_radius_deg: float = 0.25,
        species_limit: int = 60,
    ):
        self.log = get_logger()
        self.online_place = online_place
        self.online_species = online_species
        self.grid_deg = grid_deg
        self.species_radius_deg = species_radius_deg
        self.species_limit = species_limit

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._place_cache_path = self.cache_dir / "place_cache.json"
        self._species_cache_path = self.cache_dir / "species_cache.json"
        self._taxo_cache_path = self.cache_dir / "taxo_cache.json"
        self._place_cache = self._load_json(self._place_cache_path)
        self._species_cache = self._load_json(self._species_cache_path)
        self._taxo_cache = self._load_json(self._taxo_cache_path)

        self._rg = None  # reverse_geocoder, chargé paresseusement
        self._last_nominatim = 0.0

        # B3 : écriture des caches DIFFÉRÉE (au lieu de réécrire tout le JSON à
        # chaque photo). On marque « sale » et on flush périodiquement + en fin.
        self._dirty: set[str] = set()
        self._writes_since_flush = 0
        self._flush_every = 50

    # -- utilitaires cache --------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_json(self, path: Path, data: dict) -> None:
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self.log.warning("Échec écriture cache %s (%s)", path.name, e)

    def _mark_dirty(self, which: str) -> None:
        """Marque un cache modifié ; flush par lots pour limiter les écritures disque."""
        self._dirty.add(which)
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Écrit sur disque les caches modifiés depuis le dernier flush."""
        # PYTHON — DICT LITTÉRAL : { clé: valeur, ... }. Ici les valeurs sont des
        # TUPLES (chemin, données). On mappe un nom -> (où écrire, quoi écrire).
        targets = {
            "place": (self._place_cache_path, self._place_cache),
            "species": (self._species_cache_path, self._species_cache),
            "taxo": (self._taxo_cache_path, self._taxo_cache),
        }
        # `list(self._dirty)` fait une COPIE du set avant d'itérer, car on modifie
        # le set d'origine (clear) ensuite — itérer en modifiant est risqué.
        for which in list(self._dirty):
            # PYTHON — TUPLE UNPACKING : `path, data = (a, b)` affecte a à path et
            # b à data en une ligne (destructuring). Très idiomatique.
            path, data = targets[which]
            self._save_json(path, data)
        self._dirty.clear()       # vide le set en place
        self._writes_since_flush = 0

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "GpsContext":
        return self

    def __exit__(self, *exc) -> None:
        self.flush()

    def _cell_key(self, lat: float, lon: float) -> str:
        g = self.grid_deg
        return f"{round(lat / g) * g:.4f},{round(lon / g) * g:.4f}"

    # -- reverse geocoding --------------------------------------------------

    def _ensure_rg(self):
        if self._rg is None:
            import reverse_geocoder as rg  # chargement lourd, paresseux

            self._rg = rg
        return self._rg

    def place_tags(self, lat: float, lon: float) -> PlaceTags:
        """Tags de lieu, offline par défaut, online si activé (avec cache)."""
        # round(x, 4) arrondit à 4 décimales -> clé de cache ~11 m de précision.
        key = f"{round(lat, 4)},{round(lon, 4)}"
        # `key in dict` teste la présence d'une CLÉ (pas d'une valeur). O(1).
        if key in self._place_cache:
            # Robuste à un changement de schéma de PlaceTags : on ne garde que
            # les champs encore connus (un vieux cache ne fait pas planter).
            cached = self._place_cache[key]
            # fields(PlaceTags) (de dataclasses) liste les champs ; on en fait un
            # set de noms valides via une set-comprehension.
            valid = {f.name for f in fields(PlaceTags)}
            # PYTHON — deux idiomes combinés :
            #  1) DICT COMPREHENSION {k: v for k, v in d.items() if ...} : filtre
            #     le dict caché pour ne garder que les clés connues.
            #  2) `**dict` = UNPACKING DE DICT en arguments nommés : PlaceTags(**{
            #     "country": "FR"}) revient à PlaceTags(country="FR"). C'est la
            #     « splat » : on déballe un dict en kwargs. (`*liste` fait pareil
            #     pour les args positionnels.)
            return PlaceTags(**{k: v for k, v in cached.items() if k in valid})

        tags = self._place_offline(lat, lon)
        if self.online_place:
            poi = self._place_online_poi(lat, lon)
            if poi:
                tags.poi = poi

        self._place_cache[key] = tags.__dict__
        self._mark_dirty("place")
        return tags

    def _place_offline(self, lat: float, lon: float) -> PlaceTags:
        try:
            rg = self._ensure_rg()
            res = rg.search((lat, lon), mode=1)[0]
            return PlaceTags(
                country=res.get("cc") or None,
                admin1=res.get("admin1") or None,
                admin2=res.get("admin2") or None,
                city=res.get("name") or None,
            )
        except Exception as e:
            self.log.warning("Reverse geocoding offline échoué (%s)", e)
            return PlaceTags()

    def _place_online_poi(self, lat: float, lon: float) -> str | None:
        # Respect du quota Nominatim (1 req/s).
        dt = time.time() - self._last_nominatim
        if dt < 1.0:
            time.sleep(1.0 - dt)
        try:
            r = requests.get(
                _NOMINATIM_URL,
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14},
                headers={"User-Agent": _USER_AGENT},
                timeout=10,
            )
            self._last_nominatim = time.time()
            if r.ok:
                addr = r.json().get("address", {})
                for k in ("tourism", "natural", "leisure", "attraction", "park"):
                    if k in addr:
                        return addr[k]
        except Exception as e:
            self.log.info("Nominatim indisponible (%s) — tags lieu offline seuls", e)
        return None

    # -- taxonomie (GBIF) pour tags hiérarchiques d'espèces -----------------

    def species_hierarchy(self, scientific_name: str, root: str = "Faune") -> list[str]:
        """Chemin taxonomique d'une espèce : [root, classe_fr, famille, espèce].

        Ex. 'Alcedo atthis' -> ['Faune', 'Oiseaux', 'Alcedinidae', 'Alcedo atthis'].
        Renvoie au minimum [root, nom] si la taxonomie est indisponible. Caché.
        """
        key = scientific_name.strip()
        if not key:
            return [root]
        if key in self._taxo_cache:
            return self._taxo_cache[key]
        path = [root]
        try:
            r = requests.get(
                "https://api.gbif.org/v1/species/match",
                params={"name": key}, headers={"User-Agent": _USER_AGENT}, timeout=10,
            )
            if r.ok:
                d = r.json()
                cls = _CLASS_FR.get(d.get("class", ""), d.get("class"))
                if cls:
                    path.append(cls)
                if d.get("family"):
                    path.append(d["family"])
                path.append(d.get("species") or key)
            else:
                path.append(key)
        except Exception as e:
            self.log.info("Taxonomie GBIF indisponible (%s) — tag espèce plat", e)
            path.append(key)
        if len(path) == 1:
            path.append(key)
        self._taxo_cache[key] = path
        self._mark_dirty("taxo")
        return path

    # -- espèces (GBIF) -----------------------------------------------------

    def species_for(self, lat: float, lon: float, group: str) -> list[str] | None:
        """Liste d'espèces plausibles pour un groupe, ou None si indisponible.

        None signifie « pas de filtrage » (réseau coupé / offline) → la passe 2
        doit alors utiliser son vocabulaire complet.
        """
        if not self.online_species:
            return None
        cell = self._cell_key(lat, lon)
        self._species_cache.setdefault(cell, {})
        if group in self._species_cache[cell]:
            return self._species_cache[cell][group]

        species = self._gbif_species(lat, lon, group)
        if species is not None:
            self._species_cache[cell][group] = species
            self._mark_dirty("species")
        return species

    def _gbif_species(self, lat: float, lon: float, group: str) -> list[str] | None:
        class_key = GBIF_CLASS_KEYS.get(group)
        if class_key is None:
            return None
        r = self.species_radius_deg
        # Facette directement sur scientificName : une SEULE requête renvoie les
        # noms d'espèces (auparavant : 1 requête + 1 lookup par espèce = N+1).
        params = {
            "decimalLatitude": f"{lat - r},{lat + r}",
            "decimalLongitude": f"{lon - r},{lon + r}",
            "taxonKey": class_key,
            "rank": "SPECIES",
            "limit": 0,
            "facet": "scientificName",
            "facetLimit": self.species_limit,
        }
        try:
            resp = requests.get(
                _GBIF_URL, params=params, headers={"User-Agent": _USER_AGENT}, timeout=15
            )
            if not resp.ok:
                return None
            facets = resp.json().get("facets", [])
            if not facets:
                return []
            counts = facets[0].get("counts", [])
            return [c["name"] for c in counts if c.get("name")]
        except Exception as e:
            self.log.info("GBIF indisponible (%s) — pas de filtrage espèces", e)
            return None
