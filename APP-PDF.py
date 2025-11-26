import os
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


ZOOM = 2.0  # zoom inicial para renderizado de páginas


class PdfLabelApp(tk.Tk):
    """
    Aplicación principal:
    - Seleccionar carpeta
    - Elegir ejemplos y etiquetar (campos y tablas)
    - Entrenar modelo + previsualización 1
    - Procesar masivo + previsualización 2
    - Exportar resultados / tablas / renombrar / proteger PDFs
    """

    def __init__(self):
        super().__init__()
        self.title("Etiquetador y extractor de PDFs")
        self.geometry("1200x650")

        self.folder_path = None
        self.pdf_files = []          # lista de nombres de archivo (str)
        self.example_files = []      # subset para entrenamiento
        self.samples = []            # ejemplos etiquetados (dicts)
        self.model = None            # RandomForest entrenado
        self.label_box_stats = {}    # etiqueta -> coords medias (x_min,...)
        self.results_df = None       # DataFrame final con todos los PDFs

        # etiquetas conocidas (para combo en el etiquetador)
        self.known_labels = set()

        # configuración de tablas por etiqueta: {label: {"num_cols": int, "header_first_row": bool}}
        self.table_configs = {}
        # resultados de tablas
        self.table_results = []
        self.tables_df = None

        # orden de columnas preferido para resultados principales
        self.column_order = None

        self._build_ui()

    # ==========================
    # UI
    # ==========================

    def _build_ui(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        btn_folder = ttk.Button(
            top_frame,
            text="1) Seleccionar carpeta de PDFs",
            command=self.select_folder
        )
        btn_folder.pack(side=tk.LEFT, padx=5)

        self.btn_label_examples = ttk.Button(
            top_frame,
            text="2) Etiquetar ejemplos (máx. 20)",
            command=self.start_label_examples,
            state=tk.DISABLED
        )
        self.btn_label_examples.pack(side=tk.LEFT, padx=5)

        self.btn_train_preview1 = ttk.Button(
            top_frame,
            text="3) Entrenar modelo y ver ejemplos",
            command=self.train_and_preview_examples,
            state=tk.DISABLED
        )
        self.btn_train_preview1.pack(side=tk.LEFT, padx=5)

        self.btn_process_all = ttk.Button(
            top_frame,
            text="4) Procesar todos los PDFs",
            command=self.process_all_pdfs,
            state=tk.DISABLED
        )
        self.btn_process_all.pack(side=tk.LEFT, padx=5)

        self.btn_export = ttk.Button(
            top_frame,
            text="5) Exportar resultados CSV",
            command=self.export_results,
            state=tk.DISABLED
        )
        self.btn_export.pack(side=tk.LEFT, padx=5)

        # Exportar tablas
        self.btn_export_tables = ttk.Button(
            top_frame,
            text="Exportar tablas CSV",
            command=self.export_tables,
            state=tk.DISABLED
        )
        self.btn_export_tables.pack(side=tk.LEFT, padx=5)

        # Ordenar columnas
        self.btn_reorder_cols = ttk.Button(
            top_frame,
            text="Ordenar columnas",
            command=self.reorder_columns,
            state=tk.DISABLED
        )
        self.btn_reorder_cols.pack(side=tk.LEFT, padx=5)

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.LabelFrame(bottom_frame, text="PDFs en la carpeta")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        self.listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            width=50,
            height=30
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        right_frame = ttk.LabelFrame(bottom_frame, text="Acciones sobre resultados")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Botones de renombrado y protección
        self.btn_rename = ttk.Button(
            right_frame,
            text="Renombrar PDFs según columnas",
            command=self.rename_pdfs,
            state=tk.DISABLED
        )
        self.btn_rename.pack(pady=5, anchor="w")

        self.btn_protect = ttk.Button(
            right_frame,
            text="Proteger PDFs con contraseña (columna)",
            command=self.protect_pdfs,
            state=tk.DISABLED
        )
        self.btn_protect.pack(pady=5, anchor="w")

        self.status_var = tk.StringVar(value="Seleccione una carpeta de PDFs para comenzar.")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==========================
    # Utilidades de features
    # ==========================

    @staticmethod
    def build_feature_dict(x_min, y_min, x_max, y_max, text):
        if text is None:
            text = ""
        text = str(text)
        num_digits = sum(ch.isdigit() for ch in text)
        num_alpha = sum(ch.isalpha() for ch in text)
        feature = {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
            "text_len": len(text),
            "num_digits": num_digits,
            "num_alpha": num_alpha,
            "only_digits": 1 if text.isdigit() and text != "" else 0,
            "alnum": 1 if text.isalnum() and text != "" else 0,
        }
        return feature

    @staticmethod
    def extract_table_from_clip(page, clip_rect, num_cols, header_first_row):
        """
        Extrae una tabla sencilla a partir de las 'words' dentro de clip_rect.
        Devuelve (headers, rows) donde rows es lista de listas.
        """
        words = page.get_text("words", clip=clip_rect)
        if not words:
            return [], []

        heights = [w[3] - w[1] for w in words]
        median_h = float(np.median(heights)) if heights else 5.0
        row_thr = median_h * 0.7 if median_h > 0 else 5.0

        # agrupar por filas según la coordenada Y
        words_sorted = sorted(words, key=lambda w: ((w[1] + w[3]) / 2.0))
        rows_words = []
        current = []
        last_y = None
        for w in words_sorted:
            y_center = (w[1] + w[3]) / 2.0
            if last_y is None or abs(y_center - last_y) <= row_thr:
                current.append(w)
            else:
                if current:
                    rows_words.append(current)
                current = [w]
            last_y = y_center
        if current:
            rows_words.append(current)

        # límites X globales
        x_centers = [ (w[0] + w[2]) / 2.0 for w in words ]
        min_x = min(x_centers)
        max_x = max(x_centers)
        if max_x <= min_x:
            max_x = min_x + 1.0

        table_rows = []
        for r in rows_words:
            cols = [""] * num_cols
            for w in sorted(r, key=lambda w: ((w[0] + w[2]) / 2.0)):
                x_c = (w[0] + w[2]) / 2.0
                rel = (x_c - min_x) / (max_x - min_x)
                idx = int(rel * num_cols)
                if idx < 0:
                    idx = 0
                elif idx >= num_cols:
                    idx = num_cols - 1
                word = w[4]
                if cols[idx]:
                    cols[idx] += " " + word
                else:
                    cols[idx] = word
            table_rows.append([c.strip() for c in cols])

        if not table_rows:
            return [], []

        if header_first_row:
            headers_raw = table_rows[0]
            headers = []
            for i, h in enumerate(headers_raw):
                if h:
                    h_clean = " ".join(h.split())
                else:
                    h_clean = f"col_{i+1}"
                headers.append(h_clean)
            data_rows = table_rows[1:]
        else:
            headers = [f"col_{i+1}" for i in range(num_cols)]
            data_rows = table_rows

        return headers, data_rows

    # ==========================
    # Paso 1: seleccionar carpeta
    # ==========================

    def select_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta con PDFs")
        if not folder:
            return
        self.folder_path = folder
        self.pdf_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        ]
        self.pdf_files.sort()
        self.listbox.delete(0, tk.END)
        for name in self.pdf_files:
            self.listbox.insert(tk.END, name)
        self.status_var.set(f"Carpeta seleccionada: {folder} — {len(self.pdf_files)} PDFs encontrados.")

        if self.pdf_files:
            self.btn_label_examples.config(state=tk.NORMAL)
        else:
            self.btn_label_examples.config(state=tk.DISABLED)

    # ==========================
    # Paso 2: etiquetar ejemplos
    # ==========================

    def start_label_examples(self):
        if not self.folder_path or not self.pdf_files:
            messagebox.showerror("Error", "Primero seleccione una carpeta con PDFs.")
            return

        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Debe seleccionar al menos un PDF como ejemplo.")
            return

        if len(selected_indices) > 20:
            messagebox.showerror("Error", "Máximo 20 PDFs de ejemplo para etiquetar.")
            return

        self.example_files = [self.pdf_files[i] for i in selected_indices]
        self.status_var.set(f"{len(self.example_files)} PDFs seleccionados como ejemplos para etiquetar.")

        LabelWindow(self, self.folder_path, self.example_files)

    def add_sample(self, sample_dict):
        """
        sample_dict:
            file, page, x_min, y_min, x_max, y_max, text, label,
            is_table (bool), num_cols, header_first_row (bool)
        """
        label_clean = sample_dict.get("label", "").strip()
        sample_dict["label"] = label_clean
        if label_clean:
            self.known_labels.add(label_clean)

        is_table = bool(sample_dict.get("is_table", False))
        if is_table and label_clean:
            cfg = self.table_configs.get(label_clean)
            if not cfg:
                num_cols = int(sample_dict.get("num_cols") or 1)
                header_first_row = bool(sample_dict.get("header_first_row", False))
                self.table_configs[label_clean] = {
                    "num_cols": num_cols,
                    "header_first_row": header_first_row
                }

        self.samples.append(sample_dict)
        if len(self.samples) > 0:
            self.btn_train_preview1.config(state=tk.NORMAL)

    # ==========================
    # Paso 3: entrenar modelo + preview 1
    # ==========================

    def train_and_preview_examples(self):
        if not self.samples:
            messagebox.showerror("Error", "No hay ejemplos etiquetados todavía.")
            return

        # construir features (no diferenciamos campos/tablas para el modelo;
        # solo se usan para previsualización y para promedios de posición)
        feature_rows = []
        labels = []
        for s in self.samples:
            feat = self.build_feature_dict(
                s["x_min"], s["y_min"],
                s["x_max"], s["y_max"],
                s["text"]
            )
            feature_rows.append(feat)
            labels.append(s["label"])

        X = pd.DataFrame(feature_rows)
        y = np.array(labels)

        if len(set(y)) < 2:
            messagebox.showwarning(
                "Aviso",
                "Solo hay una etiqueta distinta en los ejemplos. El modelo entrenará, "
                "pero su capacidad de generalización será limitada."
            )

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        clf.fit(X, y)
        self.model = clf

        # coords medias por etiqueta (para generación de cajas en masivo)
        self.label_box_stats = {}
        for label in set(y):
            coords = [
                (s["x_min"], s["y_min"], s["x_max"], s["y_max"])
                for s in self.samples
                if s["label"] == label
            ]
            arr = np.array(coords, dtype=float)
            self.label_box_stats[label] = arr.mean(axis=0)

        # previsualización 1: ejemplos
        preds = clf.predict(X)

        preview_data = []
        for s, pred in zip(self.samples, preds):
            tipo = "Tabla" if s.get("is_table") else "Campo"
            preview_data.append({
                "archivo": s["file"],
                "pagina": s["page"],
                "tipo": tipo,
                "etiqueta_esperada": s["label"],
                "etiqueta_predicha": pred,
                "texto": (s["text"] or "").replace("\n", " ")[:100]
            })

        self.show_preview_table(
            preview_data,
            title="Previsualización 1: ejemplos etiquetados vs predicción"
        )

        self.status_var.set(
            "Modelo entrenado. Revise la previsualización 1. "
            "Ahora puede procesar todos los PDFs."
        )
        self.btn_process_all.config(state=tk.NORMAL)

    # ==========================
    # Paso 4: procesamiento masivo
    # ==========================

    def process_all_pdfs(self):
        if not self.model or not self.label_box_stats:
            messagebox.showerror(
                "Error",
                "Primero debe entrenar el modelo con ejemplos (paso 3)."
            )
            return
        if not self.folder_path or not self.pdf_files:
            messagebox.showerror(
                "Error",
                "No hay carpeta ni PDFs cargados."
            )
            return

        self.table_results = []

        rows = []
        total = len(self.pdf_files)
        for idx, filename in enumerate(self.pdf_files, start=1):
            full_path = os.path.join(self.folder_path, filename)
            try:
                doc = fitz.open(full_path)
            except Exception as e:
                messagebox.showwarning(
                    "PDF omitido",
                    f"No se pudo abrir {filename}.\nMotivo: {e}"
                )
                continue

            row = {"archivo": filename}

            # por simplicidad, usar solo la primera página
            if doc.page_count == 0:
                doc.close()
                rows.append(row)
                continue

            page = doc[0]
            page_rect = page.rect
            pix_width = page_rect.width * ZOOM
            pix_height = page_rect.height * ZOOM

            for label, coords in self.label_box_stats.items():
                x_min_n, y_min_n, x_max_n, y_max_n = coords
                # de normalizado a pixeles
                x0_px = x_min_n * pix_width
                y0_px = y_min_n * pix_height
                x1_px = x_max_n * pix_width
                y1_px = y_max_n * pix_height

                # pixeles -> puntos (coordenadas reales del PDF)
                clip_rect = fitz.Rect(
                    x0_px / ZOOM,
                    y0_px / ZOOM,
                    x1_px / ZOOM,
                    y1_px / ZOOM
                )

                table_cfg = self.table_configs.get(label)

                if table_cfg:
                    # ---- etiqueta de TABLA ----
                    headers, data_rows = self.extract_table_from_clip(
                        page,
                        clip_rect,
                        table_cfg["num_cols"],
                        table_cfg["header_first_row"]
                    )

                    if headers and data_rows:
                        # resumen en el resultado principal
                        row[label] = f"{len(data_rows)} filas"
                        # guardar filas detalladas
                        for r_idx, row_vals in enumerate(data_rows, start=1):
                            rec = {
                                "archivo": filename,
                                "tabla": label,
                                "fila": r_idx
                            }
                            for h, v in zip(headers, row_vals):
                                rec[h] = v
                            self.table_results.append(rec)
                    else:
                        row[label] = ""
                else:
                    # ---- etiqueta de CAMPO simple ----
                    try:
                        text = page.get_text("text", clip=clip_rect)
                    except Exception:
                        text = ""

                    feat = self.build_feature_dict(
                        x_min_n, y_min_n, x_max_n, y_max_n, text
                    )
                    X_candidate = pd.DataFrame([feat])
                    _ = self.model.predict(X_candidate)[0]  # no se usa ahora

                    row[label] = (text or "").replace("\n", " ").strip()

            doc.close()
            rows.append(row)
            self.status_var.set(f"Procesando PDFs... ({idx}/{total})")
            self.update_idletasks()

        self.results_df = pd.DataFrame(rows)

        # definir orden inicial de columnas (archivo primero, luego resto alfabético)
        cols = list(self.results_df.columns)
        if "archivo" in cols:
            other = [c for c in cols if c != "archivo"]
            self.column_order = ["archivo"] + sorted(other)
        else:
            self.column_order = cols

        self.show_preview_table(
            self.results_df.to_dict(orient="records"),
            title="Previsualización 2: resultados de todos los PDFs",
            columns_order=self.column_order
        )

        # tablas
        if self.table_results:
            self.tables_df = pd.DataFrame(self.table_results)
            self.show_preview_table(
                self.tables_df.to_dict(orient="records"),
                title="Tablas extraídas (todas las etiquetas de tabla)"
            )
            self.btn_export_tables.config(state=tk.NORMAL)
        else:
            self.tables_df = None
            self.btn_export_tables.config(state=tk.DISABLED)

        self.status_var.set(
            f"Procesamiento masivo finalizado. {len(self.results_df)} filas de certificados."
        )
        self.btn_export.config(state=tk.NORMAL)
        self.btn_rename.config(state=tk.NORMAL)
        self.btn_protect.config(state=tk.NORMAL)
        self.btn_reorder_cols.config(state=tk.NORMAL)

    # ==========================
    # Paso 5: exportar CSV (campos simples)
    # ==========================

    def export_results(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados para exportar.")
            return
        if not self.folder_path:
            messagebox.showerror("Error", "No hay carpeta base.")
            return

        df = self.results_df
        if self.column_order:
            cols = [c for c in self.column_order if c in df.columns]
            df = df[cols + [c for c in df.columns if c not in cols]]

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Extraccion_PDFs_{ts}.csv"
        out_path = os.path.join(self.folder_path, default_name)

        try:
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))
            return

        messagebox.showinfo("Exportación completa", f"Archivo guardado en:\n{out_path}")

    # ==========================
    # Exportar tablas
    # ==========================

    def export_tables(self):
        if self.tables_df is None or self.tables_df.empty:
            messagebox.showerror("Error", "No hay tablas para exportar.")
            return
        if not self.folder_path:
            messagebox.showerror("Error", "No hay carpeta base.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Tablas_PDFs_{ts}.csv"
        out_path = os.path.join(self.folder_path, default_name)

        try:
            self.tables_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            messagebox.showerror("Error al exportar tablas", str(e))
            return

        messagebox.showinfo("Exportación de tablas", f"Archivo guardado en:\n{out_path}")

    # ==========================
    # Vista de tablas (preview 1 y 2)
    # ==========================

    def show_preview_table(self, records, title="Previsualización", columns_order=None):
        if not records:
            messagebox.showinfo(title, "No hay datos para mostrar.")
            return

        if columns_order is None:
            cols = list(records[0].keys())
        else:
            cols = [c for c in columns_order if c in records[0]] + [
                c for c in records[0].keys() if c not in columns_order
            ]

        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("900x500")

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=cols, show="headings")
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="w")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.configure(xscrollcommand=hsb.set)

        for rec in records:
            values = [rec.get(c, "") for c in cols]
            tree.insert("", tk.END, values=values)

    # ==========================
    # Ordenar columnas
    # ==========================

    def reorder_columns(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados para ordenar.")
            return

        cols = list(self.results_df.columns)
        if self.column_order:
            cols = [c for c in self.column_order if c in cols] + [
                c for c in cols if c not in self.column_order
            ]

        win = tk.Toplevel(self)
        win.title("Ordenar columnas")

        tk.Label(win, text="Orden de columnas (use Subir/Bajar y luego Aplicar):").pack(anchor="w", padx=5, pady=5)

        listbox = tk.Listbox(win, selectmode=tk.SINGLE, width=50, height=15)
        listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        for c in cols:
            listbox.insert(tk.END, c)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=5)

        def move_up():
            idx = listbox.curselection()
            if not idx:
                return
            i = idx[0]
            if i == 0:
                return
            text = listbox.get(i)
            listbox.delete(i)
            listbox.insert(i - 1, text)
            listbox.select_set(i - 1)

        def move_down():
            idx = listbox.curselection()
            if not idx:
                return
            i = idx[0]
            if i == listbox.size() - 1:
                return
            text = listbox.get(i)
            listbox.delete(i)
            listbox.insert(i + 1, text)
            listbox.select_set(i + 1)

        def apply_order():
            new_order = list(listbox.get(0, tk.END))
            self.column_order = new_order
            self.show_preview_table(
                self.results_df.to_dict(orient="records"),
                title="Previsualización 2 (orden actualizado)",
                columns_order=self.column_order
            )
            win.destroy()

        ttk.Button(btn_frame, text="Subir", command=move_up).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Bajar", command=move_down).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Aplicar", command=apply_order).grid(row=0, column=2, padx=5)

    # ==========================
    # Renombrar PDFs
    # ==========================

    def rename_pdfs(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados cargados.")
            return

        cols = [c for c in self.results_df.columns if c != "archivo"]
        if not cols:
            messagebox.showerror(
                "Error",
                "No hay columnas de datos (solo 'archivo')."
            )
            return

        win = tk.Toplevel(self)
        win.title("Seleccionar columnas para renombrar PDFs")

        tk.Label(win, text="Seleccione columnas a concatenar en el nombre:").pack(anchor="w", padx=5, pady=5)

        listbox = tk.Listbox(win, selectmode=tk.MULTIPLE, width=40, height=10)
        listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        for c in cols:
            listbox.insert(tk.END, c)

        tk.Label(win, text="Separador (por ejemplo '_')").pack(anchor="w", padx=5)
        sep_var = tk.StringVar(value="_")
        sep_entry = ttk.Entry(win, textvariable=sep_var, width=10)
        sep_entry.pack(anchor="w", padx=5, pady=(0, 5))

        def do_rename():
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
                old_path = os.path.join(self.folder_path, original_name)
                if not os.path.exists(old_path):
                    continue

                parts = []
                for col in selected_cols:
                    val = str(row.get(col, "")).strip()
                    if val:
                        parts.append(val)

                if not parts:
                    continue

                base = sep.join(parts)
                safe = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_"
                    for ch in base
                )
                new_name = safe + ".pdf"
                new_path = os.path.join(self.folder_path, new_name)

                counter = 1
                while os.path.exists(new_path):
                    new_name = f"{safe}_{counter}.pdf"
                    new_path = os.path.join(self.folder_path, new_name)
                    counter += 1

                try:
                    os.rename(old_path, new_path)
                except Exception:
                    continue

                self.results_df.loc[self.results_df["archivo"] == original_name, "archivo"] = new_name
                renamed_count += 1

            messagebox.showinfo(
                "Renombrado completado",
                f"Se renombraron {renamed_count} archivos."
            )
            win.destroy()

        ttk.Button(win, text="Renombrar", command=do_rename).pack(pady=5)

    # ==========================
    # Proteger PDFs
    # ==========================

    def protect_pdfs(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados cargados.")
            return
        try:
            import pikepdf
        except ImportError:
            messagebox.showerror(
                "Falta dependencia",
                "Para proteger PDFs necesitas instalar 'pikepdf':\n\npip install pikepdf"
            )
            return

        cols = [c for c in self.results_df.columns if c != "archivo"]
        if not cols:
            messagebox.showerror("Error", "No hay columnas disponibles para usar como contraseña.")
            return

        win = tk.Toplevel(self)
        win.title("Proteger PDFs con contraseña")

        tk.Label(win, text="Seleccione la columna que será la contraseña:").pack(anchor="w", padx=5, pady=5)

        col_var = tk.StringVar(value=cols[0])
        combo = ttk.Combobox(win, textvariable=col_var, values=cols, state="readonly")
        combo.pack(padx=5, pady=5)

        def do_protect():
            col = col_var.get()
            if not col:
                messagebox.showerror("Error", "Debe seleccionar una columna.")
                return

            target_folder = os.path.join(self.folder_path, "protegidos")
            os.makedirs(target_folder, exist_ok=True)

            protected_count = 0

            for _, row in self.results_df.iterrows():
                filename = row.get("archivo")
                if not filename:
                    continue
                password = str(row.get(col, "")).strip()
                if not password:
                    continue

                src_path = os.path.join(self.folder_path, filename)
                if not os.path.exists(src_path):
                    continue

                dst_path = os.path.join(target_folder, filename)

                try:
                    with pikepdf.open(src_path) as pdf:
                        pdf.save(
                            dst_path,
                            encryption=pikepdf.Encryption(
                                user=password,
                                owner=password,
                                R=4
                            )
                        )
                    protected_count += 1
                except Exception:
                    continue

            messagebox.showinfo(
                "Protección completada",
                f"Se generaron {protected_count} PDFs protegidos en la carpeta 'protegidos'."
            )
            win.destroy()

        ttk.Button(win, text="Proteger", command=do_protect).pack(pady=5)


class LabelWindow(tk.Toplevel):
    """
    Ventana para etiquetar PDFs de ejemplo:
    - Muestra el PDF como imagen en un canvas
    - Permite dibujar rectángulos y asignar etiquetas
    - Scroll y zoom
    - Campo / Tabla (con nº columnas y opción de encabezado)
    """

    def __init__(self, app: PdfLabelApp, folder_path, example_files):
        super().__init__(app)
        self.app = app
        self.folder_path = folder_path
        self.example_files = example_files
        self.current_example_index = 0

        self.doc = None
        self.current_page_index = 0
        self.pix_width = None
        self.pix_height = None

        self.zoom = ZOOM

        self.start_x = None
        self.start_y = None
        self.current_rect_id = None

        self.title("Etiquetado de ejemplos")
        self.geometry("1000x700")

        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = ttk.Label(top_frame, text="")
        self.lbl_info.pack(side=tk.LEFT, padx=5, pady=5)

        btn_prev_page = ttk.Button(top_frame, text="Página anterior", command=self.prev_page)
        btn_prev_page.pack(side=tk.LEFT, padx=5)

        btn_next_page = ttk.Button(top_frame, text="Página siguiente", command=self.next_page)
        btn_next_page.pack(side=tk.LEFT, padx=5)

        btn_zoom_out = ttk.Button(top_frame, text="Zoom -", command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=5)

        btn_zoom_in = ttk.Button(top_frame, text="Zoom +", command=self.zoom_in)
        btn_zoom_in.pack(side=tk.LEFT, padx=5)

        btn_next_example = ttk.Button(top_frame, text="Siguiente PDF de ejemplo", command=self.next_example)
        btn_next_example.pack(side=tk.LEFT, padx=5)

        btn_close = ttk.Button(top_frame, text="Terminar etiquetado", command=self.destroy)
        btn_close.pack(side=tk.RIGHT, padx=5)

        # Frame con canvas + scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="grey")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Eventos de ratón
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Scroll con rueda del ratón
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)      # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)        # Linux up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)        # Linux down

        self.load_current_example()

    # Carga y navegación de PDFs/páginas

    def load_current_example(self):
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
        full_path = os.path.join(self.folder_path, filename)
        try:
            self.doc = fitz.open(full_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir {filename}.\n{e}")
            self.current_example_index += 1
            self.load_current_example()
            return

        self.current_page_index = 0
        self.update_page_view()

    def update_page_view(self):
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
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, self.pix_width, self.pix_height))

        filename = self.example_files[self.current_example_index]
        info = f"Archivo: {filename} — Página {self.current_page_index + 1}/{self.doc.page_count} — Zoom: {self.zoom:.1f}x"
        self.lbl_info.config(text=info)

    def prev_page(self):
        if self.doc is None:
            return
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.update_page_view()

    def next_page(self):
        if self.doc is None:
            return
        if self.current_page_index < self.doc.page_count - 1:
            self.current_page_index += 1
            self.update_page_view()

    def next_example(self):
        self.current_example_index += 1
        if self.current_example_index >= len(self.example_files):
            messagebox.showinfo("Fin", "Has llegado al último PDF de ejemplo.")
            self.destroy()
            return
        self.load_current_example()

    # Zoom

    def zoom_in(self):
        self.zoom = min(self.zoom * 1.25, 6.0)
        self.update_page_view()

    def zoom_out(self):
        self.zoom = max(self.zoom / 1.25, 0.5)
        self.update_page_view()

    # Scroll con rueda del ratón

    def on_mouse_wheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            delta = int(-1 * (event.delta / 120))
            self.canvas.yview_scroll(delta * 3, "units")

    # Dibujo de rectángulos y etiquetado

    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_mouse_drag(self, event):
        if self.current_rect_id is not None:
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def ask_label_and_table_config(self):
        """
        Diálogo:
        - seleccionar etiqueta existente o escribir nueva
        - marcar si es tabla
        - si es tabla: nº columnas y si la 1ª fila es encabezado
        """
        dlg = tk.Toplevel(self)
        dlg.title("Etiqueta / Tabla")
        dlg.grab_set()

        tk.Label(dlg, text="Seleccione una etiqueta o escriba una nueva:").pack(
            anchor="w", padx=5, pady=5
        )

        labels = sorted(self.app.known_labels)
        label_var = tk.StringVar()
        combo = ttk.Combobox(dlg, textvariable=label_var, values=labels, state="normal", width=40)
        combo.pack(padx=5, pady=5)
        combo.focus()

        is_table_var = tk.BooleanVar(value=False)
        chk = ttk.Checkbutton(dlg, text="Esta zona es una TABLA", variable=is_table_var)
        chk.pack(anchor="w", padx=5, pady=5)

        num_cols_var = tk.StringVar(value="4")
        tk.Label(dlg, text="Número de columnas (solo si es tabla):").pack(anchor="w", padx=5)
        entry_cols = ttk.Entry(dlg, textvariable=num_cols_var, width=10)
        entry_cols.pack(anchor="w", padx=5, pady=(0, 5))

        header_var = tk.BooleanVar(value=True)
        chk_header = ttk.Checkbutton(dlg, text="Usar la primera fila como encabezado", variable=header_var)
        chk_header.pack(anchor="w", padx=5, pady=5)

        result = {"label": None, "is_table": False, "num_cols": None, "header_first_row": False}

        def on_ok():
            val = label_var.get().strip()
            if val == "":
                messagebox.showerror("Error", "La etiqueta no puede estar vacía.", parent=dlg)
                return
            result["label"] = val
            result["is_table"] = bool(is_table_var.get())
            if result["is_table"]:
                try:
                    ncols = int(num_cols_var.get())
                    if ncols <= 0:
                        raise ValueError
                except Exception:
                    messagebox.showerror("Error", "Número de columnas inválido.", parent=dlg)
                    return
                result["num_cols"] = ncols
                result["header_first_row"] = bool(header_var.get())
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Aceptar", command=on_ok).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=on_cancel).grid(row=0, column=1, padx=5)

        dlg.wait_window()
        if result["label"] is None:
            return None
        return result

    def on_mouse_up(self, event):
        if self.current_rect_id is None:
            return

        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        x0, y0 = self.start_x, self.start_y
        x1, y1 = end_x, end_y

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

        info = self.ask_label_and_table_config()
        if not info:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        label = info["label"]
        is_table = info["is_table"]
        num_cols = info["num_cols"]
        header_first_row = info["header_first_row"]

        # normalizados [0,1]
        x_min_n = x_min_px / self.pix_width
        y_min_n = y_min_px / self.pix_height
        x_max_n = x_max_px / self.pix_width
        y_max_n = y_max_px / self.pix_height

        # extraer texto dentro del rectángulo (para vista previa; en tablas será solo texto plano)
        page = self.doc[self.current_page_index]
        clip_rect = fitz.Rect(
            x_min_px / self.zoom,
            y_min_px / self.zoom,
            x_max_px / self.zoom,
            y_max_px / self.zoom
        )
        try:
            text = page.get_text("text", clip=clip_rect)
        except Exception:
            text = ""

        filename = self.example_files[self.current_example_index]
        sample = {
            "file": filename,
            "page": self.current_page_index + 1,
            "x_min": x_min_n,
            "y_min": y_min_n,
            "x_max": x_max_n,
            "y_max": y_max_n,
            "text": text,
            "label": label,
            "is_table": is_table,
            "num_cols": num_cols,
            "header_first_row": header_first_row
        }
        self.app.add_sample(sample)

        self.current_rect_id = None


if __name__ == "__main__":
    app = PdfLabelApp()
    app.mainloop()
