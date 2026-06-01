# Lightroom AI Tagger

Taggage de photos par **IA locale** (aucun cloud), conçu pour Adobe Lightroom
Classic mais utilisable aussi sur un simple dossier. Fusion et refonte de
[photo-auto-tagger-AI](https://github.com/LaurentVoillot/photo-auto-tagger-AI)
et [photo-folder-tagger](https://github.com/LaurentVoillot/photo-folder-tagger).

## Points clés

- **100 % local** : modèles vision via [Ollama](https://ollama.com) (par défaut
  `qwen3-vl:30b`), aucune image n'est envoyée sur Internet.
- **Source d'image en cascade** : aperçu standard Lightroom → Smart Preview
  (DNG JPEG-XL) → fichier original. On prend le premier disponible.
- **Tags de lieu par GPS** : reverse geocoding hors-ligne (+ Nominatim optionnel).
- **Filtrage d'espèces par localisation** : liste d'espèces plausibles via GBIF,
  pour fiabiliser une éventuelle identification fine (BioCLIP, expérimental).
- **Pipeline en 2 passes** : LLM généraliste, puis identification d'espèces
  ciblée sur les animaux détectés au premier plan.
- **Plugin Lightroom Classic** : tague la **sélection** et écrit les mots-clés
  directement dans le catalogue (suffixe configurable `_AI`, déduplication,
  barre de progression, annulation avec taggage partiel).
- **Mode test** strictement en lecture seule : ne modifie jamais le catalogue.

## Architecture

```
catalog_reader.py   Lecture seule du catalogue .lrcat
image_source.py     Cascade aperçu standard → Smart Preview → original
gps_context.py      Tags de lieu + espèces locales (GBIF), avec caches
pipeline.py         2 passes : LLM Ollama + BioCLIP (espèces)
log_panel.py        Journalisation + pré-vol des volumes
test_report.py      Mode test (lecture seule) en ligne de commande
bench_llm.py        Benchmark des modèles vision Ollama
gui.py              Interface PyQt6
selection_runner.py Traite une sélection exportée (appelé par le plugin)
lrc_plugin/         Plugin Lightroom Classic (Lua)
```

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # torch/open_clip optionnels (passe 2)
```

Installer [Ollama](https://ollama.com) et un modèle vision :
```bash
ollama pull qwen3-vl:30b
```

## Utilisation

### En ligne de commande (mode test, lecture seule)
```bash
.venv/bin/python test_report.py "/chemin/Catalogue.lrcat" \
    --scope "Nom de dossier" --limit 20 --tag --out /tmp/sortie
```

### Plugin Lightroom Classic
Voir [`lrc_plugin/README.md`](lrc_plugin/README.md). En résumé : ajouter le
plugin via le Gestionnaire de modules externes, sélectionner des photos, puis
**Bibliothèque → Modules externes → Taguer la sélection avec l'IA locale**.

## Matériel

Pensé pour Apple Silicon (testé sur M4 Max 64 Go). `qwen3-vl:30b` ≈ quelques
secondes par photo ; `qwen2.5vl:7b` plus rapide. Le tier aperçu standard accélère
fortement le traitement des photos déjà prévisualisées dans Lightroom.

## Licence

GPL v3.0, comme les projets d'origine.
