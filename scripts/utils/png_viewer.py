"""Simple PNG folder viewer with session-date sorting (DDMMYYYY filenames)."""
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image, ImageTk


def _session_sort_key(path: Path):
    """Sort PNGs by DDMMYYYY date extracted from stem; non-matching go last."""
    m = re.search(r"(?<!\d)(\d{7,8})(?!\d)", path.stem)
    if m:
        s = m.group(1).zfill(8)
        try:
            dd, mm, yyyy = int(s[:2]), int(s[2:4]), int(s[4:])
            return (0, yyyy, mm, dd, path.stem)
        except ValueError:
            pass
    return (1, 0, 0, 0, path.stem)


class PngViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PNG Viewer")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1e1e1e")

        self.images: list[Path] = []
        self.idx = 0
        self._photo = None

        self._build_ui()
        self._bind_keys()

    def _build_ui(self):
        # Top toolbar
        bar = tk.Frame(self.root, bg="#2d2d2d", pady=4)
        bar.pack(fill=tk.X)

        btn_style = {"bg": "#444", "fg": "white", "relief": tk.FLAT,
                     "padx": 10, "pady": 4, "cursor": "hand2"}

        tk.Button(bar, text="Open Folder", command=self.open_folder, **btn_style).pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(bar, text="◀  Prev", command=self.prev, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bar, text="Next  ▶", command=self.next, **btn_style).pack(side=tk.LEFT, padx=2)

        self.counter_var = tk.StringVar(value="No folder loaded")
        tk.Label(bar, textvariable=self.counter_var, bg="#2d2d2d", fg="#aaa",
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=12)

        self.filename_var = tk.StringVar()
        tk.Label(bar, textvariable=self.filename_var, bg="#2d2d2d", fg="#ddd",
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=4)

        # Thumbnail strip
        strip_frame = tk.Frame(self.root, bg="#252525", height=70)
        strip_frame.pack(fill=tk.X)
        strip_frame.pack_propagate(False)

        self.canvas_strip = tk.Canvas(strip_frame, bg="#252525", height=70,
                                       highlightthickness=0)
        self.canvas_strip.pack(fill=tk.X, expand=True)
        self._thumb_photos: list = []

        # Main image canvas
        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Home>", lambda e: self._show(0))
        self.root.bind("<End>", lambda e: self._show(len(self.images) - 1))
        self.root.bind("<f>", lambda e: self._toggle_fullscreen())
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

    def _toggle_fullscreen(self):
        state = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not state)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select folder with PNG files")
        if not folder:
            return
        self.images = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() == ".png"],
            key=_session_sort_key,
        )
        if not self.images:
            self.counter_var.set("No PNG files found")
            self.filename_var.set("")
            self.canvas.delete("all")
            return
        self._show(0)
        self._build_thumbnails()

    def _show(self, idx: int):
        if not self.images:
            return
        self.idx = max(0, min(idx, len(self.images) - 1))
        self._redraw()
        self._update_strip_highlight()

    def _redraw(self):
        if not self.images:
            return
        path = self.images[self.idx]
        cw = self.canvas.winfo_width() or 1200
        ch = self.canvas.winfo_height() or 700

        img = Image.open(path)
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)

        self.counter_var.set(f"{self.idx + 1} / {len(self.images)}")
        self.filename_var.set(path.name)

    def _build_thumbnails(self):
        self.canvas_strip.delete("all")
        self._thumb_photos = []
        x = 4
        THUMB_H = 58
        for i, p in enumerate(self.images):
            try:
                img = Image.open(p)
                img.thumbnail((90, THUMB_H), Image.LANCZOS)
                ph = ImageTk.PhotoImage(img)
                self._thumb_photos.append(ph)
                tag = f"thumb_{i}"
                self.canvas_strip.create_image(x, 4, anchor=tk.NW, image=ph, tags=tag)
                self.canvas_strip.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._show(idx))
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
        # Scroll the strip to keep current thumb visible
        total_w = sum(
            (ph.width() + 6 if ph else 96) for ph in self._thumb_photos
        )
        x = sum(
            (self._thumb_photos[i].width() + 6 if self._thumb_photos[i] else 96)
            for i in range(self.idx)
        )
        tw = self._thumb_photos[self.idx].width() if self._thumb_photos[self.idx] else 90
        if total_w > 0:
            frac = x / total_w
            self.canvas_strip.xview_moveto(max(0.0, frac - 0.05))
        self.canvas_strip.create_rectangle(
            x, 2, x + tw, 68, outline="#4af", width=2, tags="highlight"
        )

    def prev(self):
        self._show(self.idx - 1)

    def next(self):
        self._show(self.idx + 1)


def main():
    try:
        from PIL import Image  # noqa — just check it's available
    except ImportError:
        print("Pillow is required: pip install pillow")
        sys.exit(1)

    root = tk.Tk()
    app = PngViewer(root)

    # Open folder from CLI arg if provided
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        app.images = sorted(
            [p for p in Path(folder).iterdir() if p.suffix.lower() == ".png"],
            key=_session_sort_key,
        )
        if app.images:
            root.after(100, lambda: (app._show(0), app._build_thumbnails()))

    root.mainloop()


if __name__ == "__main__":
    main()
