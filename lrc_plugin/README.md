# Plugin Lightroom Classic — Photo Tagger (IA locale)

Tague les photos **sélectionnées** dans Lightroom Classic (et non le catalogue
entier) avec le pipeline Python local. Évite la création d'un catalogue de test.

## Principe

1. Tu sélectionnes des photos dans LrC.
2. Le plugin les **exporte en JPEG** (long côté 2048 px) dans un dossier temporaire
   et écrit un `manifest.json` (GPS, chemin original, cible XMP).
3. Il ouvre **Terminal.app** et lance `selection_runner.py`, qui te **pose toutes
   les questions d'options** (LLM oui/non, modèle, passe 2 espèces, GPS online,
   écriture XMP ou mode test), puis traite la sélection.
4. Rapport écrit dans le dossier temporaire (`rapport_selection.txt` / `.csv`).

Aucune écriture dans le catalogue : l'export passe par l'API officielle, et le
catalogue n'est jamais ouvert en parallèle.

## Installation

1. Vérifie les chemins en haut de `TagSelection.lua` :
   ```lua
   local PROJECT_DIR = "/Users/laurentvoillot/Claude/photo-tagger"
   local PYTHON_BIN  = PROJECT_DIR .. "/.venv/bin/python"
   ```
2. Dans Lightroom Classic : **Fichier > Gestionnaire de modules externes >
   Ajouter**, puis sélectionne le dossier
   `…/photo-tagger/lrc_plugin/PhotoTagger.lrplugin`.

## Utilisation

1. Sélectionne une ou plusieurs photos dans le module Bibliothèque.
2. Menu **Bibliothèque > Modules externes** (ou **Fichier > Modules d'exportation**)
   → **« Taguer la sélection avec l'IA locale… »**.
3. Une fenêtre confirme l'export ; le Terminal s'ouvre et pose les options.
4. Réponds aux questions. En mode test, rien n'est écrit ; sinon les tags vont
   dans les sidecars `.xmp` (lisibles ensuite par LrC après « Lire les
   métadonnées du fichier »).

## Remarques

- Le premier lancement charge le modèle Ollama (compter quelques dizaines de
  secondes la première photo).
- Lance Ollama au préalable (`ollama serve` ou l'app Ollama).
- macOS demandera peut-être l'autorisation à Lightroom de piloter Terminal
  (Réglages Système > Confidentialité > Automatisation).
