"""Image/PDF folder viewer with session-date sorting (DDMMYYYY filenames).

Supports .png and .pdf files. Multi-page PDFs expand into one entry per page.
Navigation: <- -> arrow keys, Home/End, thumbnail click. Press f for fullscreen.
"""
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except ImportError:
    _HAVE_FITZ = False

SUPPORTED_EXTS = {".png", ".pdf"}


def _session_sort_key(path: Path):
    """Sort by DDMMYYYY date extracted from stem; non-matching go last."""
    m = re.search(r"(?<!\d)(\d{7,8})(?!\d)", path.stem)
    if m:
        s = m.group(1).zfill(8)
        try:
            dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
            return (0, yyyy, mm, dd, path.stem)
        except ValueError:
            pass
    return (1, 0, 0, 0, path.stem)


def _load_entries(folder: Path) -> list:
    """Return (path, page_index) pairs sorted by session date then page."""
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS],
        key=_session_sort_key,
    )
    entries = []
    for p in files:
        if p.suffix.lower() == ".pdf" and _HAVE_FITZ:
            try:
                doc = fitz.open(str(p))
                for i in range(doc.page_count):
                    entries.append((p, i))
                doc.close()
            except Exception:
                entries.append((p, 0))
        else:
            entries.append((p, 0))
    return entries


def _render_entry(path: Path, page: int, max_w: int, max_h: int) -> Image.Image:
    """Return a PIL Image for a PNG or a PDF page, scaled to fit max_w x max_h."""
    if path.suffix.lower() == ".pdf":
        if not _HAVE_FITZ:
            raise RuntimeError("PyMuPDF not installed — run: pip install pymupdf")
        doc = fitz.open(str(path))
        pdf_page = doc[page]
        mat = fitz.Matrix(2.0, 2.0)  # 2x for sharpness
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        img = Image.open(path).convert("RGB")

    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


def _entry_label(path: Path, page: int, total_pages: int) -> str:
    if path.suffix.lower() == ".pdf" and total_pages > 1:
        return f"{path.name}  [p{page + 1}/{total_pages}]"
    return path.name


class Viewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Figure Viewer")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1e1e1e")

        self.entries: list = []       # list of (Path, page_index)
        self._page_counts: dict = {}  # Path -> number of pages in entries
        self.idx = 0
        self._photo = None

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        bar = tk.Frame(self.root, bg="#2d2d2d", pady=4)
        bar.pack(fill=tk.X)

        btn = {"bg": "#444", "fg": "white", "relief": tk.FLAT,
               "padx": 10, "pady": 4, "cursor": "hand2"}

        tk.Button(bar, text="Open Folder", command=self.open_folder, **btn).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(bar, text="<  Prev",     command=self.prev,        **btn).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Next  >",     command=self.next,        **btn).pack(side=tk.LEFT, padx=2)

        self.counter_var = tk.StringVar(value="No folder loaded")
        tk.Label(bar, textvariable=self.counter_var, bg="#2d2d2d", fg="#aaa",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=12)

        self.filename_var = tk.StringVar()
        tk.Label(bar, textvariable=self.filename_var, bg="#2d2d2d", fg="#ddd",
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=4)

        strip_frame = tk.Frame(self.root, bg="#252525", height=70)
        strip_frame.pack(fill=tk.X)
        strip_frame.pack_propagate(False)

        self.canvas_strip = tk.Canvas(strip_frame, bg="#252525", height=70,
                                      highlightthickness=0)
        self.canvas_strip.pack(fill=tk.X, expand=True)
        self._thumb_photos: list = []

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def _bind_keys(self):
        self.root.bind("<Right>",  lambda e: self.next())
        self.root.bind("<Left>",   lambda e: self.prev())
        self.root.bind("<Home>",   lambda e: self._show(0))
        self.root.bind("<End>",    lambda e: self._show(len(self.entries) - 1))
        self.root.bind("<f>",      lambda e: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

    def _toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select folder with PNG/PDF files")
        if not folder:
            return
        self._load(Path(folder))

    def _load(self, folder: Path):
        self.entries = _load_entries(folder)
        self._page_counts = {}
        for path, _ in self.entries:
            if path not in self._page_counts:
                self._page_counts[path] = sum(1 for p, _ in self.entries if p == path)

        if not self.entries:
            self.counter_var.set("No PNG/PDF files found")
            self.filename_var.set("")
            self.canvas.delete("all")
            return

        self._show(0)
        self._build_thumbnails()

    def _show(self, idx: int):
        if not self.entries:
            return
        self.idx = max(0, min(idx, len(self.entries) - 1))
        self._redraw()
        self._update_strip_highlight()

    def prev(self):
        self._show(self.idx - 1)

    def next(self):
        self._show(self.idx + 1)

    def _redraw(self):
        if not self.entries:
            return
        path, page = self.entries[self.idx]
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 700

        try:
            img = _render_entry(path, page, cw, ch)
        except Exception as exc:
            self.canvas.delete("all")
            self.canvas.create_text(cw // 2, ch // 2, text=f"Error: {exc}",
                                    fill="red", font=("Consolas", 12))
            return

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)

        n_pages = self._page_counts.get(path, 1)
        self.counter_var.set(f"{self.idx + 1} / {len(self.entries)}")
        self.filename_var.set(_entry_label(path, page, n_pages))

    def _build_thumbnails(self):
        self.canvas_strip.delete("all")
        self._thumb_photos = []
        x = 4
        THUMB_H = 58

        for i, (path, page) in enumerate(self.entries):
            try:
                img = _render_entry(path, page, 90, THUMB_H)
                ph = ImageTk.PhotoImage(img)
                self._thumb_photos.append(ph)
                tag = f"thumb_{i}"
                self.canvas_strip.create_image(x, 4, anchor=tk.NW, image=ph, tags=tag)
                self.canvas_strip.tag_bind(tag, "<Button-1>",
                                           lambda e, idx=i: self._show(idx))
                x += img.width + 6
            except Exception:
                self._thumb_photos.append(None)
                x += 96

        self.canvas_strip.configure(scrollregion=(0, 0, x, 70))
        self._update_strip_highlight()

    def _update_strip_highlight(self):
        self.canvas_strip.delete("highlight")
        if not self._thumb_photos:
            return
        total_w = sum((ph.width() + 6 if ph else 96) for ph in self._thumb_photos)
        x = sum(
            (self._thumb_photos[i].width() + 6 if self._thumb_photos[i] else 96)
            for i in range(self.idx)
        )
        tw = self._thumb_photos[self.idx].width() if self._thumb_photos[self.idx] else 90
        if total_w > 0:
            self.canvas_strip.xview_moveto(max(0.0, x / total_w - 0.05))
        self.canvas_strip.create_rectangle(
            x, 2, x + tw, 68, outline="#4af", width=2, tags="highlight"
        )


def main():
    if not _HAVE_FITZ:
        print("Note: PyMuPDF not found — PDF support disabled. Install with: pip install pymupdf")

    root = tk.Tk()
    app = Viewer(root)

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
        if folder.is_dir():
            root.after(100, lambda: app._load(folder))

    root.mainloop()


if __name__ == "__main__":
    main()
