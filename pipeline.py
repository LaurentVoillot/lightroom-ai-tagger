"""
pipeline.py — Orchestrateur de tagging en DEUX PASSES.

Passe 1 — LLM vision généraliste (Ollama) :
    Envoie l'image à un modèle vision (def: qwen3-vl:30b) et récupère un JSON
    structuré : { "tags": [...], "categories": [...] }.
    Les catégories signalent les groupes nécessitant une identification fine
    (birds, insects, animals, plants, astro).

Passe 2 — moteurs spécialisés (déclenchés selon les catégories) :
    - birds / insects / animals -> BioCLIP (open_clip), vocabulaire restreint
      aux espèces plausibles localement (via gps_context + GBIF) si dispo.
    - astro -> placeholder (branchera astro_client ultérieurement).
    La passe 2 est OPTIONNELLE : si torch/open_clip ne sont pas installés, on
    journalise une info et on garde seulement les tags de la passe 1.

Tags de lieu (GPS) : ajoutés via gps_context si la photo est géolocalisée.

Fusion : tags_lieu + tags_passe1 + tags_passe2, dédupliqués (insensible à la
casse), dans un ordre stable.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field

import requests

from gps_context import GpsContext
from image_source import ResolvedImage
from catalog_reader import PhotoRecord
from log_panel import get_logger

# Catégories de la passe 1 -> groupe GBIF/BioCLIP de la passe 2.
_CATEGORY_TO_GROUP = {
    "birds": "birds",
    "insects": "insects",
    "animals": "mammals",
    "mammals": "mammals",
}

_DEFAULT_PROMPT = (
    "IMPORTANT : réponds EXCLUSIVEMENT en FRANÇAIS. Tous les mots-clés du champ "
    "tags et les noms d'animaux DOIVENT être en français, jamais en anglais.\n"
    "Analyse cette photo. Réponds UNIQUEMENT en JSON avec ce schéma : "
    '{"tags": ["mot-clé", ...], '
    '"categories": ["birds"|"insects"|"animals"|"plants"|"astro"|"none"], '
    '"animals": [{"nom": "nom courant français", "groupe": "birds"|"insects"|"animals", '
    '"box": [x0, y0, x1, y1]}]}. '
    "Les tags en français : 8 à 15 mots-clés concrets (type de scène, lieu, "
    "objets, ambiance, couleurs dominantes). "
    "Le champ categories liste les groupes d'êtres vivants AU PREMIER PLAN et "
    "bien visibles qui méritent une identification d'espèce. "
    "Le champ animals ne contient QUE les animaux nets et bien visibles au "
    "premier plan : pour chacun, son nom courant, son groupe, et sa boîte "
    "englobante box en coordonnées normalisées entre 0 et 1 [gauche, haut, "
    "droite, bas]. Si aucun animal net au premier plan : categories=[\"none\"] "
    "et animals=[].\n"
    "RAPPEL FINAL : les valeurs de tags et de nom sont en FRANÇAIS uniquement."
)


@dataclass
class DetectedAnimal:
    """Un animal détecté au premier plan par la passe 1 (LLM grounding)."""

    name: str                       # nom courant proposé par le LLM
    group: str                      # birds | insects | animals
    box: tuple[float, float, float, float] | None = None  # normalisé 0..1


@dataclass
class TagResult:
    """Résultat de tagging d'une photo, avec provenance des tags."""

    place_tags: list[str] = field(default_factory=list)
    llm_tags: list[str] = field(default_factory=list)
    species_tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    animals: list[DetectedAnimal] = field(default_factory=list)

    def merged(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in self.place_tags + self.llm_tags + self.species_tags:
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(t.strip())
        return out


class OllamaVision:
    """Client minimal pour la passe 1 (Ollama /api/generate, format JSON)."""

    def __init__(
        self,
        model: str = "qwen3-vl:30b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        jpeg_quality: int = 80,
        max_edge: int = 1024,
        timeout: int = 300,
    ):
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.temperature = temperature
        self.jpeg_quality = jpeg_quality
        self.max_edge = max_edge
        self.timeout = timeout
        self.log = get_logger()

    def _encode(self, img) -> str:
        im = img.convert("RGB").copy()
        im.thumbnail((self.max_edge, self.max_edge))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extrait le premier objet JSON d'une réponse en texte libre.

        On n'utilise PAS le mode format=json d'Ollama : certains modèles vision
        (dont qwen3-vl) le gèrent mal et renvoient une sortie vide/dégénérée.
        On demande donc du JSON dans le prompt et on l'extrait ici.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}

    @staticmethod
    def _parse_animals(data: dict) -> list["DetectedAnimal"]:
        out: list[DetectedAnimal] = []
        for a in data.get("animals", []) or []:
            if not isinstance(a, dict):
                continue
            name = str(a.get("nom") or a.get("name") or "").strip()
            group = str(a.get("groupe") or a.get("group") or "animals").lower()
            box = a.get("box")
            tbox = None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    tbox = tuple(float(x) for x in box)  # type: ignore[assignment]
                except (TypeError, ValueError):
                    tbox = None
            if name:
                out.append(DetectedAnimal(name=name, group=group, box=tbox))
        return out

    def analyze(
        self, img, prompt: str = _DEFAULT_PROMPT
    ) -> tuple[list[str], list[str], list["DetectedAnimal"]]:
        """Renvoie (tags, categories, animals). Listes vides si échec."""
        try:
            r = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [self._encode(img)],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            if not r.ok:
                self.log.warning("Ollama HTTP %s", r.status_code)
                return [], [], []
            data = self._extract_json(r.json().get("response", ""))
            # On ne garde que des tags STRING : sous température, le modèle peut
            # renvoyer un nombre ou un objet, qu'on ne veut pas écrire en mot-clé.
            tags = [t.strip() for t in data.get("tags", []) if isinstance(t, str) and t.strip()]
            cats = [str(c).lower() for c in data.get("categories", []) if isinstance(c, str)]
            cats = [c for c in cats if c and c != "none"]
            animals = self._parse_animals(data)
            return tags, cats, animals
        except requests.exceptions.RequestException as e:
            self.log.error("Ollama injoignable (%s)", e)
            return [], [], []
        except Exception as e:
            self.log.warning("Réponse Ollama illisible (%s)", e)
            return [], [], []


class SpeciesClassifier:
    """Passe 2 BioCLIP (open_clip). Chargé paresseusement, dégrade si absent."""

    def __init__(self, model_name: str = "hf-hub:imageomics/bioclip"):
        self.model_name = model_name
        self.log = get_logger()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._available: bool | None = None

    def available(self) -> bool:
        if self._available is None:
            try:
                import torch  # noqa: F401
                import open_clip  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                self.log.info(
                    "Passe 2 désactivée : torch/open_clip absents "
                    "(pip install torch open_clip_torch pour activer BioCLIP)."
                )
        return self._available

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import open_clip

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name
        )
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self._model.eval()

    def classify(
        self,
        img,
        candidate_species: list[str],
        min_cosine: float = 0.20,
        min_margin: float = 0.02,
    ) -> str | None:
        """Identifie l'espèce la plus probable, avec garde-fous de confiance.

        Contrairement à un softmax (qui choisit toujours « le moins pire »),
        on regarde la SIMILARITÉ COSINUS brute :
          - le meilleur candidat doit dépasser `min_cosine` (présence réelle) ;
          - l'écart top1/top2 doit dépasser `min_margin` (choix non ambigu).
        Sinon on renvoie None (pas d'identification fiable).

        candidate_species : noms scientifiques (filtrés GPS). Vide -> None.
        """
        if not candidate_species or not self.available():
            return None
        import torch

        self._ensure_model()
        image = self._preprocess(img.convert("RGB")).unsqueeze(0)
        text = self._tokenizer(candidate_species)
        with torch.no_grad():
            img_feat = self._model.encode_image(image)
            txt_feat = self._model.encode_text(text)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            cos = (img_feat @ txt_feat.T)[0]  # cosinus brut [-1, 1]

        k = min(2, len(candidate_species))
        vals, idx = cos.topk(k)
        vals = vals.tolist()
        best = vals[0]
        margin = best - vals[1] if k > 1 else 1.0
        if best >= min_cosine and margin >= min_margin:
            return candidate_species[idx[0].item()]
        return None


class TaggingPipeline:
    """Assemble passe 1 + GPS + passe 2 pour produire les tags d'une photo."""

    def __init__(
        self,
        ollama: OllamaVision | None = None,
        gps: GpsContext | None = None,
        species: SpeciesClassifier | None = None,
        use_species_pass: bool = False,
    ):
        # use_species_pass désactivé par défaut : sur les aperçus basse résolution,
        # BioCLIP est ambigu (cf. notes). LLM + GPS donnent déjà des tags fiables.
        # Réactivable explicitement (flag --species) quand on travaillera sur des
        # images pleine résolution ou avec un meilleur format de noms d'espèces.
        self.ollama = ollama or OllamaVision()
        self.gps = gps
        self.species = species or SpeciesClassifier()
        self.use_species_pass = use_species_pass
        self.log = get_logger()

    @staticmethod
    def _crop(img, box: tuple[float, float, float, float]):
        """Recadre selon une box normalisée 0..1, avec une petite marge."""
        w, h = img.size
        x0, y0, x1, y1 = box
        # tolère un ordre inversé et borne dans [0,1]
        x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
        y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            return None
        pad = 0.05
        px0 = int(max(0.0, x0 - pad) * w)
        py0 = int(max(0.0, y0 - pad) * h)
        px1 = int(min(1.0, x1 + pad) * w)
        py1 = int(min(1.0, y1 + pad) * h)
        if px1 <= px0 or py1 <= py0:
            return None
        return img.crop((px0, py0, px1, py1))

    def process(self, rec: PhotoRecord, resolved: ResolvedImage) -> TagResult:
        img = resolved.image
        result = TagResult()

        # Tags de lieu (si GPS + contexte GPS fourni).
        if self.gps is not None and rec.has_gps and rec.gps_lat is not None:
            try:
                result.place_tags = self.gps.place_tags(rec.gps_lat, rec.gps_lon).as_tags()
            except Exception as e:
                self.log.info("%s : tags lieu indisponibles (%s)", rec.display_name, e)

        # Passe 1 : LLM généraliste + grounding des animaux au premier plan.
        result.llm_tags, result.categories, result.animals = self.ollama.analyze(img)

        # Passe 2 : identification fine, UNIQUEMENT pour les animaux que le LLM a
        # explicitement détectés au premier plan (garde-fou anti-faux positifs).
        if not (self.use_species_pass and result.animals):
            return result
        if not (self.gps is not None and rec.has_gps and rec.gps_lat is not None):
            return result  # sans GPS, pas de liste d'espèces locales -> on s'abstient

        for animal in result.animals:
            group = _CATEGORY_TO_GROUP.get(animal.group)
            if group is None:
                continue
            candidates = self.gps.species_for(rec.gps_lat, rec.gps_lon, group)
            if not candidates:
                continue
            # Recadrage sur l'animal si une box est fournie, sinon image entière.
            target = img
            if animal.box is not None:
                crop = self._crop(img, animal.box)
                if crop is not None:
                    target = crop
            species = self.species.classify(target, candidates)
            if species:
                result.species_tags.append(species)

        return result
