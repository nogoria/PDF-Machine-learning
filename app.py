# -*- coding: utf-8 -*-
"""
Aplicación de etiquetado y extracción de datos desde PDFs usando Tkinter.

Características principales:
- Selección de carpeta y PDFs de ejemplo (máx. 20) para etiquetar manualmente.
- Visualización del PDF en un canvas, dibujo de cajas y asignación de etiquetas.
- Entrenamiento local de un modelo sencillo (RandomForest) con las cajas etiquetadas.
- Previsualización de resultados sobre ejemplos y procesamiento masivo de todos los PDFs.
- Exportación de resultados, renombrado opcional de PDFs y protección con contraseña.

Requisitos de librerías: PyMuPDF (fitz), Pillow, numpy, pandas, scikit-learn, (opcional) pikepdf.
"""
import base64
import ctypes
import datetime
import hashlib
import json
import os
import pickle
import subprocess
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from cryptography.fernet import Fernet
from sklearn.ensemble import RandomForestClassifier

ZOOM = 2.0  # zoom para renderizar páginas en pantalla
MAX_EXAMPLES = 20
TRAINING_FILE = "training_samples.json"
CONFIG_FOLDER = Path.home() / ".pdf_labeler_configs"
CONFIG_FILE = CONFIG_FOLDER / "configs.enc"
CONFIG_PASSPHRASE = "pdf_labeler_secret_phrase"


@dataclass
class Sample:
    """Representa una caja etiquetada dentro de un PDF."""

    file: str
    page: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    text: str
    label: str

    def to_feature_dict(self) -> dict:
        """Construye el diccionario de features para el modelo."""
        text = self.text or ""
        num_digits = sum(ch.isdigit() for ch in text)
        num_alpha = sum(ch.isalpha() for ch in text)
        has_space = 1 if any(ch.isspace() for ch in text) else 0
        return {
            "x_min": float(self.x_min),
            "y_min": float(self.y_min),
            "x_max": float(self.x_max),
            "y_max": float(self.y_max),
            "text_len": len(text),
            "num_digits": num_digits,
            "num_alpha": num_alpha,
            "only_digits": 1 if text.isdigit() and text != "" else 0,
            "alnum": 1 if text.isalnum() and text != "" else 0,
            "has_space": has_space,
        }

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "page": self.page,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "text": self.text,
            "label": self.label,
        }


class ModelManager:
    """Encapsula el entrenamiento y la predicción con scikit-learn."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        # Estadísticas por etiqueta y página: label -> page -> media de coords
        self.label_box_stats: dict[str, dict[int, np.ndarray]] = {}

    def train(self, samples: list[Sample]) -> None:
        if not samples:
            raise ValueError("Se requieren muestras para entrenar.")

        X = pd.DataFrame([s.to_feature_dict() for s in samples])
        y = np.array([s.label for s in samples])

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        self.model = clf

        # Calcular coordenadas promedio por etiqueta y página
        self.label_box_stats = {}
        for s in samples:
            self.label_box_stats.setdefault(s.label, {}).setdefault(s.page, []).append(
                (s.x_min, s.y_min, s.x_max, s.y_max)
            )

        for label, pages in self.label_box_stats.items():
            for page, coords in pages.items():
                arr = np.array(coords, dtype=float)
                pages[page] = arr.mean(axis=0)

    def predict(self, sample_features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        return self.model.predict(sample_features)

    def predict_single(self, feature: dict) -> str:
        df = pd.DataFrame([feature])
        return self.predict(df)[0]


class PdfLabelApp(tk.Tk):
    """Aplicación principal basada en Tkinter."""

    def __init__(self):
        super().__init__()
        self.title("Etiquetador y extractor de PDFs")
        self.geometry("1200x720")

        self.folder_path: Path | None = None
        self.pdf_files: list[str] = []
        self.example_files: list[str] = []
        self.samples: list[Sample] = []
        self.model_manager = ModelManager()
        self.results_df: pd.DataFrame | None = None
        self.tables_df: pd.DataFrame | None = None
        self.table_configs: dict[str, dict] = {}
        self.column_order: list[str] = []
        self.known_labels: set[str] = set()

        self.include_examples_var = tk.BooleanVar(value=True)

        self._build_ui()

    # ==========================
    # Utilidades de cifrado/persistencia
    # ==========================
    def _ensure_config_dir(self) -> None:
        CONFIG_FOLDER.mkdir(parents=True, exist_ok=True)
        if os.name == "nt" and CONFIG_FOLDER.exists():
            try:
                ctypes.windll.kernel32.SetFileAttributesW(str(CONFIG_FOLDER), 2)
            except Exception:
                pass

    def _get_cipher(self) -> Fernet:
        key = hashlib.sha256(CONFIG_PASSPHRASE.encode("utf-8")).digest()
        key_b64 = base64.urlsafe_b64encode(key)
        return Fernet(key_b64)

    def _load_config_store(self) -> dict:
        if not CONFIG_FILE.exists():
            return {"groups": {}}
        try:
            data = CONFIG_FILE.read_bytes()
            decrypted = self._get_cipher().decrypt(data)
            return pickle.loads(decrypted)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo leer el archivo de configuración.\n{exc}")
            return {"groups": {}}

    def _save_config_store(self, store: dict) -> None:
        try:
            self._ensure_config_dir()
            payload = pickle.dumps(store)
            encrypted = self._get_cipher().encrypt(payload)
            CONFIG_FILE.write_bytes(encrypted)
            if os.name == "nt":
                try:
                    ctypes.windll.kernel32.SetFileAttributesW(str(CONFIG_FILE), 2)
                except Exception:
                    pass
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar la configuración.\n{exc}")

    def _update_group_buttons(self) -> None:
        if self.model_manager.model is not None and self.model_manager.label_box_stats:
            self.btn_guardar_grupo.config(state=tk.NORMAL)
        else:
            self.btn_guardar_grupo.config(state=tk.DISABLED)

    # ==========================
    # Construcción de la interfaz
    # ==========================
    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(
            top_frame,
            text="1) Seleccionar carpeta de PDFs",
            command=self.select_folder,
        ).pack(side=tk.LEFT, padx=5)

        self.btn_label_examples = ttk.Button(
            top_frame,
            text="2) Etiquetar ejemplos (máx. 20)",
            command=self.start_label_examples,
            state=tk.DISABLED,
        )
        self.btn_label_examples.pack(side=tk.LEFT, padx=5)

        self.btn_train_preview1 = ttk.Button(
            top_frame,
            text="3) Entrenar modelo y ver ejemplos",
            command=self.train_and_preview_examples,
            state=tk.DISABLED,
        )
        self.btn_train_preview1.pack(side=tk.LEFT, padx=5)

        self.btn_process_all = ttk.Button(
            top_frame,
            text="4) Procesar todos los PDFs",
            command=self.process_all_pdfs,
            state=tk.DISABLED,
        )
        self.btn_process_all.pack(side=tk.LEFT, padx=5)

        self.btn_guardar_grupo = ttk.Button(
            top_frame,
            text="Guardar grupo",
            command=self.save_group,
            state=tk.DISABLED,
        )
        self.btn_guardar_grupo.pack(side=tk.LEFT, padx=5)

        self.btn_cargar_grupo = ttk.Button(
            top_frame,
            text="Cargar grupo",
            command=self.load_group,
        )
        self.btn_cargar_grupo.pack(side=tk.LEFT, padx=5)

        self.btn_export = ttk.Button(
            top_frame,
            text="5) Exportar resultados Excel",
            command=self.export_results,
            state=tk.DISABLED,
        )
        self.btn_export.pack(side=tk.LEFT, padx=5)

        self.btn_export_tables = ttk.Button(
            top_frame,
            text="Exportar tablas",
            command=self.export_tables,
            state=tk.DISABLED,
        )
        self.btn_export_tables.pack(side=tk.LEFT, padx=5)

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.LabelFrame(bottom_frame, text="PDFs en la carpeta")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        self.listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            width=50,
            height=30,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        right_frame = ttk.LabelFrame(bottom_frame, text="Acciones sobre resultados")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.include_examples_check = ttk.Checkbutton(
            right_frame,
            text="Incluir PDFs de ejemplo en el procesamiento masivo",
            variable=self.include_examples_var,
        )
        self.include_examples_check.pack(anchor="w", pady=5)

        self.btn_rename = ttk.Button(
            right_frame,
            text="Renombrar PDFs según columnas",
            command=self.rename_pdfs,
            state=tk.DISABLED,
        )
        self.btn_rename.pack(pady=5, anchor="w")

        self.btn_protect = ttk.Button(
            right_frame,
            text="Proteger PDFs con contraseña (columna)",
            command=self.protect_pdfs,
            state=tk.DISABLED,
        )
        self.btn_protect.pack(pady=5, anchor="w")

        self.status_var = tk.StringVar(value="Seleccione una carpeta de PDFs para comenzar.")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==========================
    # Paso 1: seleccionar carpeta
    # ==========================
    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Seleccionar carpeta con PDFs")
        if not folder:
            return
        self.folder_path = Path(folder)
        self.pdf_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".pdf")])

        self.listbox.delete(0, tk.END)
        for name in self.pdf_files:
            self.listbox.insert(tk.END, name)

        self.status_var.set(
            f"Carpeta seleccionada: {folder} — {len(self.pdf_files)} PDFs encontrados."
        )

        if self.pdf_files:
            self.btn_label_examples.config(state=tk.NORMAL)
        else:
            self.btn_label_examples.config(state=tk.DISABLED)

    # ==========================
    # Paso 2: etiquetar ejemplos
    # ==========================
    def start_label_examples(self) -> None:
        if not self.folder_path or not self.pdf_files:
            messagebox.showerror("Error", "Primero seleccione una carpeta con PDFs.")
            return

        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Debe seleccionar al menos un PDF como ejemplo.")
            return
        if len(selected_indices) > MAX_EXAMPLES:
            messagebox.showerror("Error", f"Máximo {MAX_EXAMPLES} PDFs de ejemplo para etiquetar.")
            return

        self.example_files = [self.pdf_files[i] for i in selected_indices]
        self.status_var.set(
            f"{len(self.example_files)} PDFs seleccionados como ejemplos para etiquetar."
        )
        LabelWindow(self, self.folder_path, self.example_files)

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)
        if len(self.samples) > 0:
            self.btn_train_preview1.config(state=tk.NORMAL)

    # ==========================
    # Paso 3: entrenar y previsualizar
    # ==========================
    def train_and_preview_examples(self) -> None:
        if not self.samples:
            messagebox.showerror("Error", "No hay ejemplos etiquetados todavía.")
            return

        try:
            self.model_manager.train(self.samples)
        except Exception as exc:  # el entrenamiento puede fallar con pocos datos
            messagebox.showerror("Error en entrenamiento", str(exc))
            return

        self.known_labels = set(s.label for s in self.samples)

        # Guardar ejemplos etiquetados junto al folder
        if self.folder_path:
            try:
                training_path = self.folder_path / TRAINING_FILE
                with open(training_path, "w", encoding="utf-8") as fh:
                    json.dump([s.to_dict() for s in self.samples], fh, ensure_ascii=False, indent=2)
            except Exception:
                # No bloquea la app si falla el guardado
                pass

        X = pd.DataFrame([s.to_feature_dict() for s in self.samples])
        preds = self.model_manager.predict(X)

        preview_data = []
        for s, pred in zip(self.samples, preds):
            preview_data.append(
                {
                    "NombreArchivo": s.file,
                    "Pagina": s.page,
                    "EtiquetaEsperada": s.label,
                    "EtiquetaPredicha": pred,
                    "TextoRecortado": (s.text or "").replace("\n", " ")[:100],
                }
            )

        self.show_preview_table(
            preview_data,
            title="Previsualización 1: ejemplos etiquetados vs predicción",
        )
        self.status_var.set(
            "Modelo entrenado. Revise la previsualización 1. Ahora puede procesar todos los PDFs."
        )
        self.btn_process_all.config(state=tk.NORMAL)
        self._update_group_buttons()

    # ==========================
    # Persistencia de grupos
    # ==========================
    def save_group(self) -> None:
        if self.model_manager.model is None or not self.model_manager.label_box_stats:
            messagebox.showerror("Error", "No hay modelo ni etiquetas para guardar.")
            return

        group_name = simpledialog.askstring(
            "Guardar grupo", "Nombre del grupo de etiquetas:", parent=self
        )
        if not group_name:
            return

        store = self._load_config_store()
        groups = store.setdefault("groups", {})
        if group_name in groups:
            if not messagebox.askyesno(
                "Sobrescribir", f"El grupo '{group_name}' ya existe. ¿Desea sobrescribirlo?"
            ):
                return

        groups[group_name] = {
            "label_box_stats": self.model_manager.label_box_stats,
            "table_configs": self.table_configs,
            "known_labels": list(self.known_labels),
            "model": self.model_manager.model,
        }
        self._save_config_store(store)
        messagebox.showinfo("Guardado", f"Grupo '{group_name}' guardado correctamente.")

    def load_group(self) -> None:
        if not CONFIG_FILE.exists():
            messagebox.showerror("Error", "No hay archivo de configuraciones guardado.")
            return

        store = self._load_config_store()
        groups = store.get("groups", {})
        if not groups:
            messagebox.showinfo("Sin grupos", "No hay grupos guardados para cargar.")
            return

        chooser = tk.Toplevel(self)
        chooser.title("Seleccionar grupo")
        chooser.geometry("300x300")

        tk.Label(chooser, text="Seleccione un grupo de etiquetas:").pack(padx=5, pady=5)
        listbox = tk.Listbox(chooser)
        listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for name in groups.keys():
            listbox.insert(tk.END, name)

        def do_load() -> None:
            selection = listbox.curselection()
            if not selection:
                messagebox.showerror("Error", "Seleccione un grupo")
                return
            group_name_local = listbox.get(selection[0])
            data = groups.get(group_name_local)
            if not data:
                messagebox.showerror("Error", "Grupo no encontrado")
                return

            try:
                self.model_manager.label_box_stats = data.get("label_box_stats", {})
                self.table_configs = data.get("table_configs", {})
                self.known_labels = set(data.get("known_labels", []))
                self.model_manager.model = data.get("model")
                self.status_var.set(
                    f"Grupo '{group_name_local}' cargado. Puede procesar todos los PDFs."
                )
                self.btn_process_all.config(state=tk.NORMAL)
                self.btn_export.config(state=tk.DISABLED)
                self.btn_export_tables.config(state=tk.DISABLED)
                self.btn_rename.config(state=tk.DISABLED)
                self.btn_protect.config(state=tk.DISABLED)
                self._update_group_buttons()
            except Exception as exc:
                messagebox.showerror("Error", f"No se pudo cargar el grupo.\n{exc}")
            chooser.destroy()

        ttk.Button(chooser, text="Cargar", command=do_load).pack(pady=5)

    # ==========================
    # Paso 4: procesamiento masivo
    # ==========================
    def process_all_pdfs(self) -> None:
        if self.model_manager.model is None:
            messagebox.showerror("Error", "Primero debe entrenar el modelo con ejemplos (paso 3).")
            return
        if not self.folder_path or not self.pdf_files:
            messagebox.showerror("Error", "No hay carpeta ni PDFs cargados.")
            return

        files_to_process = list(self.pdf_files)
        if not self.include_examples_var.get():
            files_to_process = [f for f in self.pdf_files if f not in self.example_files]

        rows: list[dict] = []
        total = len(files_to_process)
        for idx, filename in enumerate(files_to_process, start=1):
            full_path = self.folder_path / filename
            try:
                doc = fitz.open(full_path)
            except Exception as exc:
                messagebox.showwarning("PDF omitido", f"No se pudo abrir {filename}.\nMotivo: {exc}")
                continue

            row = {"archivo": filename}
            for label, pages in self.model_manager.label_box_stats.items():
                for page_num, coords in pages.items():
                    if page_num - 1 >= doc.page_count:
                        continue
                    page = doc[page_num - 1]
                    page_rect = page.rect
                    pix_width = page_rect.width * ZOOM
                    pix_height = page_rect.height * ZOOM

                    x_min_n, y_min_n, x_max_n, y_max_n = coords
                    x0_px = x_min_n * pix_width
                    y0_px = y_min_n * pix_height
                    x1_px = x_max_n * pix_width
                    y1_px = y_max_n * pix_height

                    clip_rect = fitz.Rect(
                        x0_px / ZOOM,
                        y0_px / ZOOM,
                        x1_px / ZOOM,
                        y1_px / ZOOM,
                    )
                    try:
                        text = page.get_text("text", clip=clip_rect)
                    except Exception:
                        text = ""

                    feat = Sample(
                        file=filename,
                        page=page_num,
                        x_min=x_min_n,
                        y_min=y_min_n,
                        x_max=x_max_n,
                        y_max=y_max_n,
                        text=text,
                        label=label,
                    ).to_feature_dict()

                    pred_label = self.model_manager.predict_single(feat)
                    if pred_label == label:
                        row[label] = (text or "").replace("\n", " ").strip()
                        break
                if label not in row:
                    row[label] = ""

            doc.close()
            rows.append(row)
            self.status_var.set(f"Procesando PDFs... ({idx}/{total})")
            self.update_idletasks()

        self.results_df = pd.DataFrame(rows)
        self.column_order = list(self.results_df.columns)
        self.show_preview_table(
            self.results_df.to_dict(orient="records"),
            title="Previsualización 2: resultados de todos los PDFs",
        )

        self.status_var.set(
            f"Procesamiento masivo finalizado. {len(self.results_df)} filas generadas."
        )
        self.btn_export.config(state=tk.NORMAL)
        if self.tables_df is not None and not self.tables_df.empty:
            self.btn_export_tables.config(state=tk.NORMAL)
        else:
            self.btn_export_tables.config(state=tk.DISABLED)
        self.btn_rename.config(state=tk.NORMAL)
        self.btn_protect.config(state=tk.NORMAL)

    # ==========================
    # Paso 5: exportar resultados
    # ==========================
    def export_results(self) -> None:
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados para exportar.")
            return
        if not self.folder_path:
            messagebox.showerror("Error", "No hay carpeta base.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_xlsx = self.folder_path / f"Extraccion_PDFs_{ts}.xlsx"

        try:
            if self.tables_df is None or self.tables_df.empty:
                df_final = self.results_df.copy()
            else:
                merged_rows: list[dict] = []
                for _, rrow in self.results_df.iterrows():
                    archivo = rrow.get("archivo")
                    matching = self.tables_df
                    if archivo is not None and not self.tables_df.empty:
                        matching = self.tables_df[self.tables_df.get("archivo") == archivo]
                    if matching is None or matching.empty:
                        merged_rows.append(rrow.to_dict())
                    else:
                        for _, trow in matching.iterrows():
                            merged = rrow.to_dict()
                            for col, val in trow.items():
                                if col == "archivo":
                                    continue
                                merged[col] = val
                            merged_rows.append(merged)
                df_final = pd.DataFrame(merged_rows)

            if not self.column_order:
                self.column_order = list(df_final.columns)
            ordered = [c for c in self.column_order if c in df_final.columns]
            remaining = [c for c in df_final.columns if c not in ordered]
            df_final = df_final[ordered + remaining]
            df_final.to_excel(default_xlsx, index=False)
        except Exception as exc:
            messagebox.showerror("Error al exportar", str(exc))
            return

        if os.name == "nt":
            try:
                os.startfile(default_xlsx)  # type: ignore[attr-defined]
            except Exception:
                pass

        messagebox.showinfo("Exportación completa", f"Archivo guardado en:\n{default_xlsx}")

    # ==========================
    # Vista de tablas (preview 1 y 2)
    # ==========================
    def show_preview_table(self, records: list[dict], title: str = "Previsualización") -> None:
        if not records:
            messagebox.showinfo(title, "No hay datos para mostrar.")
            return

        cols = list(records[0].keys())

        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1000x560")

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=cols, show="headings")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=150, anchor="w")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.configure(xscrollcommand=hsb.set)

        for rec in records:
            values = [rec.get(c, "") for c in cols]
            tree.insert("", tk.END, values=values)

    def export_tables(self) -> None:
        if self.tables_df is None or self.tables_df.empty:
            messagebox.showerror("Error", "No hay tablas para exportar.")
            return
        if not self.folder_path:
            messagebox.showerror("Error", "No hay carpeta base.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_xlsx = self.folder_path / f"Tablas_PDFs_{ts}.xlsx"
        try:
            self.tables_df.to_excel(default_xlsx, index=False)
        except Exception as exc:
            messagebox.showerror("Error al exportar tablas", str(exc))
            return

        if os.name == "nt":
            try:
                os.startfile(default_xlsx)  # type: ignore[attr-defined]
            except Exception:
                pass

        messagebox.showinfo("Exportación completa", f"Archivo guardado en:\n{default_xlsx}")

    # ==========================
    # Renombrado de PDFs
    # ==========================
    def rename_pdfs(self) -> None:
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados cargados.")
            return

        cols = [c for c in self.results_df.columns if c != "archivo"]
        if not cols:
            messagebox.showerror("Error", "No hay columnas de datos (solo 'archivo').")
            return

        win = tk.Toplevel(self)
        win.title("Seleccionar columnas para renombrar PDFs")

        tk.Label(win, text="Seleccione columnas a concatenar en el nombre:").pack(
            anchor="w", padx=5, pady=5
        )

        listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=40, height=10)
        listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        for c in cols:
            listbox.insert(tk.END, c)

        tk.Label(win, text="Separador (por ejemplo '_')").pack(anchor="w", padx=5)
        sep_var = tk.StringVar(value="_")
        sep_entry = ttk.Entry(win, textvariable=sep_var, width=10)
        sep_entry.pack(anchor="w", padx=5, pady=(0, 5))

        def do_rename() -> None:
            sel_idx = listbox.curselection()
            if not sel_idx:
                messagebox.showerror("Error", "Debe seleccionar al menos una columna.")
                return
            selected_cols = [cols[i] for i in sel_idx]
            sep = sep_var.get()

            renamed_count = 0
            for _, row in self.results_df.iterrows():
                original_name = row.get("archivo")
                if not original_name:
                    continue
                old_path = self.folder_path / original_name
                if not old_path.exists():
                    continue

                parts = []
                for col in selected_cols:
                    val = str(row.get(col, "")).strip()
                    if val:
                        parts.append(val)

                if not parts:
                    continue

                base = sep.join(parts)
                safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
                new_name = safe + ".pdf"
                new_path = self.folder_path / new_name

                counter = 1
                while new_path.exists():
                    new_name = f"{safe}_{counter}.pdf"
                    new_path = self.folder_path / new_name
                    counter += 1

                try:
                    os.rename(old_path, new_path)
                except Exception:
                    continue

                self.results_df.loc[self.results_df["archivo"] == original_name, "archivo"] = new_name
                renamed_count += 1

            messagebox.showinfo("Renombrado completado", f"Se renombraron {renamed_count} archivos.")
            win.destroy()

        ttk.Button(win, text="Renombrar", command=do_rename).pack(pady=5)

    # ==========================
    # Protección de PDFs
    # ==========================
    def protect_pdfs(self) -> None:
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados cargados.")
            return
        try:
            import pikepdf  # type: ignore
        except ImportError:
            messagebox.showerror(
                "Falta dependencia",
                "Para proteger PDFs necesitas instalar 'pikepdf':\n\npip install pikepdf",
            )
            return

        cols = [c for c in self.results_df.columns if c != "archivo"]
        if not cols:
            messagebox.showerror("Error", "No hay columnas disponibles para usar como contraseña.")
            return

        win = tk.Toplevel(self)
        win.title("Proteger PDFs con contraseña")

        tk.Label(win, text="Seleccione la columna que será la contraseña:").pack(
            anchor="w", padx=5, pady=5
        )

        col_var = tk.StringVar(value=cols[0])
        combo = ttk.Combobox(win, textvariable=col_var, values=cols, state="readonly")
        combo.pack(padx=5, pady=5)

        def do_protect() -> None:
            col = col_var.get()
            if not col:
                messagebox.showerror("Error", "Debe seleccionar una columna.")
                return

            target_folder = self.folder_path / "protegidos"
            target_folder.mkdir(exist_ok=True)

            protected_count = 0
            for _, row in self.results_df.iterrows():
                filename = row.get("archivo")
                if not filename:
                    continue
                password = str(row.get(col, "")).strip()
                if not password:
                    continue

                src_path = self.folder_path / filename
                if not src_path.exists():
                    continue

                dst_path = target_folder / filename

                try:
                    with pikepdf.open(src_path) as pdf:  # type: ignore
                        pdf.save(
                            dst_path,
                            encryption=pikepdf.Encryption(user=password, owner=password, R=4),
                        )
                    protected_count += 1
                except Exception:
                    continue

            messagebox.showinfo(
                "Protección completada",
                f"Se generaron {protected_count} PDFs protegidos en la carpeta 'protegidos'.",
            )
            win.destroy()

        ttk.Button(win, text="Proteger", command=do_protect).pack(pady=5)


class LabelWindow(tk.Toplevel):
    """Ventana para visualizar y etiquetar PDFs de ejemplo."""

    def __init__(self, app: PdfLabelApp, folder_path: Path, example_files: list[str]):
        super().__init__(app)
        self.app = app
        self.folder_path = folder_path
        self.example_files = example_files
        self.current_example_index = 0

        self.doc: fitz.Document | None = None
        self.current_page_index = 0
        self.pix_width: int | None = None
        self.pix_height: int | None = None
        self.zoom = ZOOM

        self.start_x: float | None = None
        self.start_y: float | None = None
        self.current_rect_id: int | None = None

        self.title("Etiquetado de ejemplos")
        self.geometry("900x700")

        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = ttk.Label(top_frame, text="")
        self.lbl_info.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(top_frame, text="Página anterior", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Página siguiente", command=self.next_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Siguiente PDF de ejemplo", command=self.next_example).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(top_frame, text="Terminar etiquetado", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(self, bg="grey")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.load_current_example()

    # Navegación de ejemplos/páginas
    def load_current_example(self) -> None:
        if self.doc is not None:
            try:
                self.doc.close()
            except Exception:
                pass

        if self.current_example_index >= len(self.example_files):
            messagebox.showinfo("Fin", "Ya no hay más PDFs de ejemplo.")
            self.destroy()
            return

        filename = self.example_files[self.current_example_index]
        full_path = self.folder_path / filename
        try:
            self.doc = fitz.open(full_path)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo abrir {filename}.\n{exc}")
            self.current_example_index += 1
            self.load_current_example()
            return

        self.current_page_index = 0
        self.update_page_view()

    def update_page_view(self) -> None:
        if self.doc is None or self.current_page_index < 0 or self.current_page_index >= self.doc.page_count:
            return

        page = self.doc[self.current_page_index]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        self.pix_width, self.pix_height = pix.width, pix.height

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(
            width=self.pix_width,
            height=self.pix_height,
            scrollregion=(0, 0, self.pix_width, self.pix_height),
        )
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        filename = self.example_files[self.current_example_index]
        info = f"Archivo: {filename} — Página {self.current_page_index + 1}/{self.doc.page_count}"
        self.lbl_info.config(text=info)

    def prev_page(self) -> None:
        if self.doc is None:
            return
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.update_page_view()

    def next_page(self) -> None:
        if self.doc is None:
            return
        if self.current_page_index < self.doc.page_count - 1:
            self.current_page_index += 1
            self.update_page_view()

    def next_example(self) -> None:
        self.current_example_index += 1
        if self.current_example_index >= len(self.example_files):
            messagebox.showinfo("Fin", "Has llegado al último PDF de ejemplo.")
            self.destroy()
            return
        self.load_current_example()

    # Dibujo de rectángulos y etiquetado
    def on_mouse_down(self, event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2
        )

    def on_mouse_drag(self, event) -> None:
        if self.current_rect_id is not None:
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event) -> None:
        if self.current_rect_id is None:
            return

        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.x, event.y

        if self.pix_width is None or self.pix_height is None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        x_min_px = min(x0, x1)
        y_min_px = min(y0, y1)
        x_max_px = max(x0, x1)
        y_max_px = max(y0, y1)

        if abs(x_max_px - x_min_px) < 5 or abs(y_max_px - y_min_px) < 5:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        label = simpledialog.askstring(
            "Etiqueta", "Nombre de la etiqueta (ej: NOMBRE_ASEGURADO):", parent=self
        )
        if not label:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        x_min_n = x_min_px / self.pix_width
        y_min_n = y_min_px / self.pix_height
        x_max_n = x_max_px / self.pix_width
        y_max_n = y_max_px / self.pix_height

        page = self.doc[self.current_page_index]
        clip_rect = fitz.Rect(
            x_min_px / self.zoom,
            y_min_px / self.zoom,
            x_max_px / self.zoom,
            y_max_px / self.zoom,
        )
        try:
            text = page.get_text("text", clip=clip_rect)
        except Exception:
            text = ""

        filename = self.example_files[self.current_example_index]
        sample = Sample(
            file=filename,
            page=self.current_page_index + 1,
            x_min=x_min_n,
            y_min=y_min_n,
            x_max=x_max_n,
            y_max=y_max_n,
            text=text,
            label=label,
        )
        self.app.add_sample(sample)

        # se deja el rectángulo dibujado como referencia
        self.current_rect_id = None


if __name__ == "__main__":
    app = PdfLabelApp()
    app.mainloop()
