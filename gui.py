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
    QProgressBar,
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
    progress = pyqtSignal(int, int, str)  # done, total, nom de la photo courante

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        # Import tardif : évite de charger torch/ollama tant qu'on ne lance rien.
        from test_report import run_test

        try:
            run_test(progress_cb=lambda d, t, n: self.progress.emit(d, t, n),
                     **self.params)
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
        from PyQt6.QtCore import QSettings

        self._settings = QSettings("phototagger", "lightroom-ai-tagger")
        self._build_ui()
        # Vérifie le catalogue chaque fois que le champ est édité.
        self.catalog_edit.editingFinished.connect(self._on_catalog_changed)
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

        # Type de périmètre : sélection courante / tout le catalogue / dossier.
        self.perimeter_combo = QComboBox()
        self.perimeter_combo.addItems(
            ["Dossier / sous-chaîne", "Sélection courante (LrC)", "Tout le catalogue"]
        )
        self.perimeter_combo.currentIndexChanged.connect(self._on_perimeter_changed)
        g.addWidget(QLabel("Traiter :"), 1, 0)
        g.addWidget(self.perimeter_combo, 1, 1, 1, 2)

        self.scope_combo = QComboBox()
        self.scope_combo.setEditable(True)
        self.scope_combo.setMinimumWidth(400)
        load_folders = QPushButton("Lister les dossiers")
        load_folders.clicked.connect(self._load_folders)
        g.addWidget(QLabel("Dossier :"), 2, 0)
        g.addWidget(self.scope_combo, 2, 1)
        g.addWidget(load_folders, 2, 2)
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

        self.skip_tagged_check = QCheckBox("Ignorer les photos déjà taguées par l'IA")
        self.skip_tagged_check.setChecked(True)
        og.addWidget(self.skip_tagged_check, 0, 3)

        self.resume_check = QCheckBox("Reprendre (ignorer les photos déjà traitées)")
        og.addWidget(self.resume_check, 4, 0, 1, 2)

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

        # --- Écriture (mode test vs écriture réelle) ---
        wr = QGroupBox("Écriture")
        wg = QGridLayout(wr)
        self.test_check = QCheckBox("Mode test (lecture seule, aucune écriture)")
        self.test_check.setChecked(True)  # test par défaut, sécurité
        self.test_check.toggled.connect(self._on_test_toggled)
        wg.addWidget(self.test_check, 0, 0, 1, 3)

        self.xmp_check = QCheckBox("Créer / compléter les sidecars .xmp")
        self.catalog_check = QCheckBox("Écrire les mots-clés dans la base LrC (Lightroom fermé)")
        # désactivés tant que le mode test est coché
        self.xmp_check.setEnabled(False)
        self.catalog_check.setEnabled(False)
        wg.addWidget(self.xmp_check, 1, 0, 1, 3)
        wg.addWidget(self.catalog_check, 2, 0, 1, 3)

        wg.addWidget(QLabel("Suffixe des tags :"), 3, 0)
        self.suffix_edit = QLineEdit("_AI")
        self.suffix_edit.setMaximumWidth(120)
        wg.addWidget(self.suffix_edit, 3, 1)
        wg.addWidget(QLabel("(vide = aucun suffixe ; non destructif)"), 3, 2)

        self.hier_check = QCheckBox(
            "Mots-clés hiérarchiques (Lieu>Pays>Ville, Faune>Classe>Espèce)"
        )
        self.hier_check.setEnabled(False)
        wg.addWidget(self.hier_check, 4, 0, 1, 3)
        root.addWidget(wr)

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

        # --- Progression ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

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

    @staticmethod
    def _diagnose_catalog(lrcat: str) -> tuple[bool, str]:
        """Évalue un catalogue. Renvoie (utilisable, message).

        Un catalogue est jugé inadapté s'il n'a pas de dossier Smart Previews à
        côté ET très peu de photos (cas typique du catalogue cloud/mobile par
        défaut, ~/Pictures/Lightroom/Lightroom Catalog.lrcat).
        """
        p = Path(lrcat)
        if not p.is_file():
            return False, "Catalogue introuvable."
        smart = p.with_name(p.stem + " Smart Previews.lrdata")
        try:
            with CatalogReader(lrcat) as cat:
                n = cat.count_in_scope()
        except Exception as e:
            return False, f"Lecture impossible : {e}"
        has_smart = smart.is_dir()
        if not has_smart and n < 100:
            return False, (
                f"⚠ Catalogue inadapté : {n} photo(s), aucun dossier Smart "
                f"Previews. C'est probablement le catalogue cloud/mobile par "
                f"défaut. Choisis ton catalogue principal (ex. sur /Volumes)."
            )
        warn = "" if has_smart else " (pas de Smart Previews — sera lent)"
        return True, f"Catalogue OK : {n} photo(s){warn}."

    def _on_catalog_changed(self) -> None:
        """Vérifie le catalogue dès qu'il change, et mémorise s'il est valide."""
        lrcat = self.catalog_edit.text().strip()
        if not lrcat:
            return
        ok, msg = self._diagnose_catalog(lrcat)
        self.status.setText(msg)
        if ok:
            self._settings.setValue("last_catalog", lrcat)

    def _browse_catalog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choisir un catalogue", "", "Catalogues Lightroom (*.lrcat)"
        )
        if path:
            self.catalog_edit.setText(path)
            self._on_catalog_changed()

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

    def _on_perimeter_changed(self, idx: int) -> None:
        """Active le champ dossier seulement pour le périmètre « Dossier »."""
        self.scope_combo.setEnabled(idx == 0)

    def _on_test_toggled(self, checked: bool) -> None:
        """Active/désactive les options d'écriture selon le mode test."""
        self.xmp_check.setEnabled(not checked)
        self.catalog_check.setEnabled(not checked)
        self.suffix_edit.setEnabled(not checked)
        self.hier_check.setEnabled(not checked)
        self.run_btn.setText("Lancer le mode test" if checked else "Lancer le taggage")

    def _run(self) -> None:
        lrcat = self.catalog_edit.text().strip()
        if not lrcat or not Path(lrcat).is_file():
            self.status.setText("Choisis d'abord un catalogue valide.")
            return

        # Avertissement NON bloquant : si le catalogue semble inadapté
        # (cloud/mobile), on prévient mais on laisse l'utilisateur confirmer —
        # certains catalogues cloud sont volontairement traités.
        ok, msg = self._diagnose_catalog(lrcat)
        if not ok:
            from PyQt6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Catalogue inhabituel",
                msg + "\n\nLancer quand même ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                self.status.setText(msg)
                return

        test_mode = self.test_check.isChecked()
        write_xmp = (not test_mode) and self.xmp_check.isChecked()
        write_catalog = (not test_mode) and self.catalog_check.isChecked()

        # Garde-fou : écriture base seulement si Lightroom est fermé.
        if write_catalog:
            from writers import catalog_is_locked

            if catalog_is_locked(lrcat):
                self.status.setText(
                    "⚠ Catalogue verrouillé : ferme Lightroom pour écrire dans la base."
                )
                return

        # Confirmation avant toute écriture réelle.
        if write_xmp or write_catalog:
            cibles = []
            if write_xmp:
                cibles.append("sidecars .xmp")
            if write_catalog:
                cibles.append("base Lightroom")
            from PyQt6.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                self,
                "Confirmer l'écriture",
                "Écriture RÉELLE (non destructive) dans : "
                + " + ".join(cibles)
                + f"\nSuffixe : « {self.suffix_edit.text()} ».\n\nContinuer ?",
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        limit = self.limit_spin.value() or None
        # Périmètre : 0 = dossier, 1 = sélection courante, 2 = tout le catalogue.
        perim = self.perimeter_combo.currentIndex()
        selected_only = perim == 1
        scope = self._current_scope() if perim == 0 else None
        params = dict(
            lrcat=lrcat,
            scope=scope,
            limit=limit,
            gps_only=self.gps_only.isChecked(),
            order=self._parse_order(),
            out_dir=self.out_edit.text().strip() or None,
            tag=self.tag_check.isChecked(),
            model=self.model_combo.currentText().strip(),
            species_pass=self.species_check.isChecked(),
            test_mode=test_mode,
            write_xmp=write_xmp,
            write_catalog=write_catalog,
            suffix=self.suffix_edit.text(),
            selected_only=selected_only,
            skip_tagged=self.skip_tagged_check.isChecked(),
            hierarchical=(not test_mode) and self.hier_check.isChecked(),
            resume=self.resume_check.isChecked(),
        )
        self.run_btn.setEnabled(False)
        self.status.setText("Traitement en cours…")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self._thread = QThread()
        self._worker = Worker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress_bar.setMaximum(max(total, 1))
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"%v / %m — {name}")

    def _on_done(self) -> None:
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status.setText(
            "Terminé. infos: %d · warnings: %d · erreurs: %d"
            % (self._counts["info"], self._counts["warning"], self._counts["error"])
        )


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    # Pré-remplit : dernier catalogue valide mémorisé, sinon le défaut X10.
    last = win._settings.value("last_catalog", "")
    default_cat = "/Volumes/X10/LR-v15/LR-v15.lrcat"
    chosen = last if last and Path(last).is_file() else default_cat
    if Path(chosen).is_file():
        win.catalog_edit.setText(chosen)
        win._on_catalog_changed()  # affiche le diagnostic au démarrage
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
