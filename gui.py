"""
gui.py — Interface graphique PyQt6 du Photo Tagger.

Fonctionnalités :
  - Choix du catalogue .lrcat et du périmètre (sous-chaîne de dossier),
    avec liste des dossiers du catalogue triés par nombre de photos.
  - Options : limite, GPS-only, ordre de cascade, activation tagging (LLM),
    modèle Ollama, passe 2 espèces.
  - Fenêtre de log intégrée : warnings d'images indisponibles affichés en
    couleur, avec compteur. Le pré-vol des volumes (1 seul message si démonté)
    apparaît ici.
  - Lancement du MODE TEST (lecture seule) dans un thread pour ne pas figer
    l'UI. Aucune écriture dans le catalogue.

Le traitement réutilise exactement la logique de test_report.run_test mais en
émettant les logs vers la fenêtre via un handler Qt.

Lancement : .venv/bin/python gui.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from catalog_reader import CatalogReader
from image_source import SourceKind
from log_panel import LOGGER_NAME

_LEVEL_COLORS = {
    logging.INFO: QColor("#d8d8d8"),
    logging.WARNING: QColor("#e8a33d"),
    logging.ERROR: QColor("#e05561"),
    logging.CRITICAL: QColor("#e05561"),
}


class QtLogHandler(logging.Handler, QObject):
    """Handler logging qui pousse les enregistrements vers la fenêtre (thread-safe)."""

    record = pyqtSignal(str, int)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, rec: logging.LogRecord) -> None:
        self.record.emit(self.format(rec), rec.levelno)


class Worker(QObject):
    """Exécute le mode test dans un thread séparé."""

    finished = pyqtSignal()

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        # Import tardif : évite de charger torch/ollama tant qu'on ne lance rien.
        from test_report import run_test

        try:
            run_test(**self.params)
        except Exception as e:  # on ne laisse jamais l'UI planter
            logging.getLogger(LOGGER_NAME).error("Échec du traitement : %s", e)
        finally:
            self.finished.emit()


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Photo Tagger — mode test")
        self.resize(900, 700)
        self._thread: QThread | None = None
        self._counts = {"info": 0, "warning": 0, "error": 0}
        self._build_ui()
        self._setup_logging()

    # -- Construction de l'interface ---------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # --- Source ---
        src = QGroupBox("Source")
        g = QGridLayout(src)
        self.catalog_edit = QLineEdit()
        self.catalog_edit.setPlaceholderText("Chemin du catalogue .lrcat")
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self._browse_catalog)
        g.addWidget(QLabel("Catalogue :"), 0, 0)
        g.addWidget(self.catalog_edit, 0, 1)
        g.addWidget(browse, 0, 2)

        self.scope_combo = QComboBox()
        self.scope_combo.setEditable(True)
        self.scope_combo.setMinimumWidth(400)
        load_folders = QPushButton("Lister les dossiers")
        load_folders.clicked.connect(self._load_folders)
        g.addWidget(QLabel("Périmètre :"), 1, 0)
        g.addWidget(self.scope_combo, 1, 1)
        g.addWidget(load_folders, 1, 2)
        root.addWidget(src)

        # --- Options ---
        opt = QGroupBox("Options")
        og = QGridLayout(opt)
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(0, 1_000_000)
        self.limit_spin.setValue(20)
        self.limit_spin.setSpecialValueText("(toutes)")
        og.addWidget(QLabel("Limite :"), 0, 0)
        og.addWidget(self.limit_spin, 0, 1)

        self.gps_only = QCheckBox("Photos géolocalisées uniquement")
        og.addWidget(self.gps_only, 0, 2)

        self.order_combo = QComboBox()
        self.order_combo.addItems(
            [
                "preview, smart, original",
                "smart, preview, original",
                "smart, original",
                "original, smart, preview",
            ]
        )
        og.addWidget(QLabel("Cascade :"), 1, 0)
        og.addWidget(self.order_combo, 1, 1)

        self.tag_check = QCheckBox("Générer les tags (LLM)")
        self.tag_check.setChecked(True)
        og.addWidget(self.tag_check, 1, 2)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            ["qwen3-vl:30b", "qwen3-vl:8b", "qwen2.5vl:7b", "gemma4:latest"]
        )
        og.addWidget(QLabel("Modèle :"), 2, 0)
        og.addWidget(self.model_combo, 2, 1)

        self.species_check = QCheckBox("Passe 2 espèces (BioCLIP, expérimental)")
        og.addWidget(self.species_check, 2, 2)

        self.out_edit = QLineEdit(str(Path.home() / "phototagger_out"))
        og.addWidget(QLabel("Sortie :"), 3, 0)
        og.addWidget(self.out_edit, 3, 1, 1, 2)
        root.addWidget(opt)

        # --- Actions ---
        actions = QHBoxLayout()
        self.run_btn = QPushButton("Lancer le mode test")
        self.run_btn.clicked.connect(self._run)
        clear_btn = QPushButton("Effacer le log")
        clear_btn.clicked.connect(lambda: self.log_view.clear())
        self.level_filter = QComboBox()
        self.level_filter.addItems(["Tout", "Warnings et +", "Erreurs"])
        self.level_filter.currentIndexChanged.connect(self._apply_filter)
        actions.addWidget(self.run_btn)
        actions.addWidget(clear_btn)
        actions.addStretch()
        actions.addWidget(QLabel("Filtre :"))
        actions.addWidget(self.level_filter)
        root.addLayout(actions)

        # --- Log ---
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background:#1e1e1e; font-family:Menlo,monospace; font-size:12px;"
        )
        root.addWidget(self.log_view, stretch=1)

        self.status = QLabel("Prêt.")
        root.addWidget(self.status)

    # -- Logging vers la fenêtre -------------------------------------------

    def _setup_logging(self) -> None:
        self._records: list[tuple[str, int]] = []
        self.qt_handler = QtLogHandler()
        self.qt_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
        )
        self.qt_handler.record.connect(self._on_record)
        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.qt_handler)

    def _on_record(self, text: str, level: int) -> None:
        self._records.append((text, level))
        if level >= logging.ERROR:
            self._counts["error"] += 1
        elif level >= logging.WARNING:
            self._counts["warning"] += 1
        else:
            self._counts["info"] += 1
        self._append_if_visible(text, level)
        self.status.setText(
            "infos: %d · warnings: %d · erreurs: %d"
            % (self._counts["info"], self._counts["warning"], self._counts["error"])
        )

    def _min_level(self) -> int:
        idx = self.level_filter.currentIndex()
        return {0: logging.DEBUG, 1: logging.WARNING, 2: logging.ERROR}[idx]

    def _append_if_visible(self, text: str, level: int) -> None:
        if level < self._min_level():
            return
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(_LEVEL_COLORS.get(level, QColor("#d8d8d8")))
        cursor.insertText(text + "\n", fmt)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _apply_filter(self) -> None:
        self.log_view.clear()
        for text, level in self._records:
            self._append_if_visible(text, level)

    # -- Actions -----------------------------------------------------------

    def _browse_catalog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un catalogue", "", "Catalogues Lightroom (*.lrcat)"
        )
        if path:
            self.catalog_edit.setText(path)

    def _load_folders(self) -> None:
        lrcat = self.catalog_edit.text().strip()
        if not lrcat or not Path(lrcat).is_file():
            self.status.setText("Catalogue introuvable.")
            return
        try:
            with CatalogReader(lrcat) as cat:
                folders = cat.list_folders(limit=300)
        except Exception as e:
            self.status.setText(f"Erreur lecture catalogue : {e}")
            return
        self.scope_combo.clear()
        for path, n in folders:
            # On propose le dernier segment comme périmètre (plus court à saisir).
            name = path.rstrip("/").split("/")[-1]
            self.scope_combo.addItem(f"{name}  ({n})", userData=name)
        self.status.setText(f"{len(folders)} dossiers chargés.")

    def _current_scope(self) -> str | None:
        data = self.scope_combo.currentData()
        if data:
            return data
        text = self.scope_combo.currentText().strip()
        # Retire un éventuel suffixe "  (123)".
        if "  (" in text:
            text = text.split("  (")[0]
        return text or None

    def _parse_order(self) -> tuple[SourceKind, ...]:
        alias = {
            "preview": SourceKind.PREVIEW,
            "smart": SourceKind.SMART,
            "original": SourceKind.ORIGINAL,
        }
        return tuple(
            alias[t.strip()] for t in self.order_combo.currentText().split(",")
        )

    def _run(self) -> None:
        lrcat = self.catalog_edit.text().strip()
        if not lrcat or not Path(lrcat).is_file():
            self.status.setText("Choisis d'abord un catalogue valide.")
            return
        limit = self.limit_spin.value() or None
        params = dict(
            lrcat=lrcat,
            scope=self._current_scope(),
            limit=limit,
            gps_only=self.gps_only.isChecked(),
            order=self._parse_order(),
            out_dir=self.out_edit.text().strip() or None,
            tag=self.tag_check.isChecked(),
            model=self.model_combo.currentText().strip(),
            species_pass=self.species_check.isChecked(),
        )
        self.run_btn.setEnabled(False)
        self.status.setText("Traitement en cours…")

        self._thread = QThread()
        self._worker = Worker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_done(self) -> None:
        self.run_btn.setEnabled(True)
        self.status.setText(
            "Terminé. infos: %d · warnings: %d · erreurs: %d"
            % (self._counts["info"], self._counts["warning"], self._counts["error"])
        )


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    # Pré-remplit avec le catalogue de test s'il existe.
    default_cat = "/Volumes/X10/LR-v15/LR-v15.lrcat"
    if Path(default_cat).is_file():
        win.catalog_edit.setText(default_cat)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
