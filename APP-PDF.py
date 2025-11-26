import os
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
import fitz  # PyMuPDF
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


ZOOM = 2.0  # zoom para renderizado de páginas


class PdfLabelApp(tk.Tk):
    """
    Aplicación principal:
    - Seleccionar carpeta
    - Elegir ejemplos y etiquetar
    - Entrenar modelo + previsualización 1
    - Procesar masivo + previsualización 2
    - Exportar / renombrar / proteger PDFs
    """

    def __init__(self):
        super().__init__()
        self.title("Etiquetador y extractor de PDFs")
        self.geometry("1100x650")

        self.folder_path = None
        self.pdf_files = []          # lista de nombres de archivo (str)
        self.example_files = []      # subset para entrenamiento
        self.samples = []            # ejemplos etiquetados (dicts)
        self.model = None            # RandomForest entrenado
        self.label_box_stats = {}    # etiqueta -> coords medias (x_min,...)
        self.results_df = None       # DataFrame final con todos los PDFs

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
            file, page, x_min, y_min, x_max, y_max, text, label
        """
        self.samples.append(sample_dict)
        # en cuanto haya al menos un ejemplo, habilitar entrenamiento
        if len(self.samples) > 0:
            self.btn_train_preview1.config(state=tk.NORMAL)

    # ==========================
    # Paso 3: entrenar modelo + preview 1
    # ==========================

    def train_and_preview_examples(self):
        if not self.samples:
            messagebox.showerror("Error", "No hay ejemplos etiquetados todavía.")
            return

        # construir features
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
            preview_data.append({
                "archivo": s["file"],
                "pagina": s["page"],
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

                try:
                    text = page.get_text("text", clip=clip_rect)
                except Exception:
                    text = ""

                feat = self.build_feature_dict(
                    x_min_n, y_min_n, x_max_n, y_max_n, text
                )
                X_candidate = pd.DataFrame([feat])
                pred_label = self.model.predict(X_candidate)[0]

                # si el modelo no cree que sea esa etiqueta, igual guardamos el texto;
                # en una versión más estricta se podría descartar.
                if pred_label != label:
                    # podríamos marcarlo de alguna forma, pero para mantener la tabla
                    # limpia simplemente lo dejamos.
                    pass

                row[label] = (text or "").replace("\n", " ").strip()

            doc.close()
            rows.append(row)
            self.status_var.set(f"Procesando PDFs... ({idx}/{total})")
            self.update_idletasks()

        self.results_df = pd.DataFrame(rows)
        self.show_preview_table(
            self.results_df.to_dict(orient="records"),
            title="Previsualización 2: resultados de todos los PDFs"
        )

        self.status_var.set(
            f"Procesamiento masivo finalizado. {len(self.results_df)} filas generadas."
        )
        self.btn_export.config(state=tk.NORMAL)
        self.btn_rename.config(state=tk.NORMAL)
        self.btn_protect.config(state=tk.NORMAL)

    # ==========================
    # Paso 5: exportar CSV
    # ==========================

    def export_results(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showerror("Error", "No hay resultados para exportar.")
            return
        if not self.folder_path:
            messagebox.showerror("Error", "No hay carpeta base.")
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Extraccion_PDFs_{ts}.csv"
        out_path = os.path.join(self.folder_path, default_name)

        try:
            self.results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))
            return

        messagebox.showinfo("Exportación completa", f"Archivo guardado en:\n{out_path}")

    # ==========================
    # Vista de tablas (preview 1 y 2)
    # ==========================

    def show_preview_table(self, records, title="Previsualización"):
        if not records:
            messagebox.showinfo(title, "No hay datos para mostrar.")
            return

        cols = list(records[0].keys())

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
                # limpiar caracteres raros
                safe = "".join(
                    ch if ch.isalnum() or ch in ("-", "_") else "_"
                    for ch in base
                )
                new_name = safe + ".pdf"
                new_path = os.path.join(self.folder_path, new_name)

                # manejar colisiones
                counter = 1
                while os.path.exists(new_path):
                    new_name = f"{safe}_{counter}.pdf"
                    new_path = os.path.join(self.folder_path, new_name)
                    counter += 1

                try:
                    os.rename(old_path, new_path)
                except Exception:
                    continue

                # actualizar en el DataFrame
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
        self.geometry("800x600")

        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = ttk.Label(top_frame, text="")
        self.lbl_info.pack(side=tk.LEFT, padx=5, pady=5)

        btn_prev_page = ttk.Button(top_frame, text="Página anterior", command=self.prev_page)
        btn_prev_page.pack(side=tk.LEFT, padx=5)

        btn_next_page = ttk.Button(top_frame, text="Página siguiente", command=self.next_page)
        btn_next_page.pack(side=tk.LEFT, padx=5)

        btn_next_example = ttk.Button(top_frame, text="Siguiente PDF de ejemplo", command=self.next_example)
        btn_next_example.pack(side=tk.LEFT, padx=5)

        btn_close = ttk.Button(top_frame, text="Terminar etiquetado", command=self.destroy)
        btn_close.pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(self, bg="grey")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

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
        self.canvas.config(width=self.pix_width, height=self.pix_height, scrollregion=(0, 0, self.pix_width, self.pix_height))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        filename = self.example_files[self.current_example_index]
        info = f"Archivo: {filename} — Página {self.current_page_index + 1}/{self.doc.page_count}"
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

    # Dibujo de rectángulos y etiquetado

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_mouse_drag(self, event):
        if self.current_rect_id is not None:
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.current_rect_id is None:
            return

        x0, y0 = self.start_x, self.start_y
        x1, y1 = event.x, event.y

        # normalizar coordenadas
        if self.pix_width is None or self.pix_height is None:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        x_min_px = min(x0, x1)
        y_min_px = min(y0, y1)
        x_max_px = max(x0, x1)
        y_max_px = max(y0, y1)

        if abs(x_max_px - x_min_px) < 5 or abs(y_max_px - y_min_px) < 5:
            # demasiado pequeño
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        label = simpledialog.askstring("Etiqueta", "Nombre de la etiqueta (ej: NOMBRE_ASEGURADO):", parent=self)
        if not label:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
            return

        # normalizados [0,1]
        x_min_n = x_min_px / self.pix_width
        y_min_n = y_min_px / self.pix_height
        x_max_n = x_max_px / self.pix_width
        y_max_n = y_max_px / self.pix_height

        # extraer texto dentro del rectángulo
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
            "label": label
        }
        self.app.add_sample(sample)

        # dejar el rectángulo dibujado como referencia
        self.current_rect_id = None


if __name__ == "__main__":
    app = PdfLabelApp()
    app.mainloop()
