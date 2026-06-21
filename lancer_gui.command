#!/bin/bash
# Lanceur Photo Tagger — double-cliquable depuis le Finder (extension .command).
# Garantit l'utilisation du Python du venv (qui a toutes les dépendances :
# tifffile, imagecodecs, torch, etc.), quel que soit le Python « système ».

# Se place dans le dossier du script, quel que soit l'endroit d'où on le lance.
cd "$(dirname "$0")" || exit 1

VENV_PY="./.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERREUR : environnement virtuel introuvable ($VENV_PY)."
    echo "Crée-le puis installe les dépendances :"
    echo "    python3 -m venv .venv"
    echo "    ./.venv/bin/pip install -r requirements.txt"
    read -r -p "Appuie sur Entrée pour fermer…"
    exit 1
fi

echo "Lancement de Photo Tagger avec $VENV_PY…"
"$VENV_PY" gui.py
