#!/usr/bin/env python3
"""
Chop & Drop  (standalone)
=========================
"Chop the wells, drop the frames."

A small GUI to prepare MAT-rig recordings for analysis. Pick a CROP option and
optionally tick "Drop frame rate" — they combine freely in ONE GPU decode pass.

  CROP (chop):
    - No crop          - keep the full frame.
    - Crop wells       - split a multi-well plate (96 / 24 / 6-well) into one
                         video per well (auto-detected grid).
    - Trim black border- each camera has a different field of view; auto-detect
                         the bright imaged area and trim the black surround,
                         leaving a small margin. One output per video.

  DROP (frames):
    - Drop frame rate  - re-time to a lower fps (e.g. 25 -> 7.5) while KEEPING
                         the same duration (frames dropped, playback NOT slowed).

  Preview crop         - overlay the detected wells / content box on a frame so
                         you can confirm detection (and tune the margin) first.

GPU: decode with NVDEC (-hwaccel cuda, codec-agnostic -> handles HEVC & H.264)
     and encode with NVENC (h264_nvenc / hevc_nvenc). The GPU is pinned with
     CUDA_VISIBLE_DEVICES so it never touches the live recorder on GPU 0
     (default analysis GPU = 1). A CPU fallback (libx264) is also available.

This script is fully self-contained (no project imports). Run it with a Python
that has cv2 + numpy + tkinter, e.g. the conda base env:

    /home/serhat/miniconda3/bin/python3 /home/serhat/Desktop/chop_and_drop.py

It calls /usr/bin/ffmpeg directly (that binary has NVENC; the tierpsy conda
ffmpeg does not).
"""

import os
import re
import shutil
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

import numpy as np
import cv2

# ---- fixed config (matches the MAT-rig conventions) -------------------------
FFMPEG = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else (shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = "/usr/bin/ffprobe" if os.path.exists("/usr/bin/ffprobe") else (shutil.which("ffprobe") or "ffprobe")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".hevc", ".h265")

# Encoder candidates, in preference order. kind drives how we accelerate:
#   nvenc        -> NVIDIA GPU (CUDA decode + NVENC encode, pinnable per GPU)
#   videotoolbox -> Apple GPU (encode-only HW accel; macOS)
#   cpu          -> software (works everywhere)
ENC_CANDIDATES = [
    {"codec": "h264_nvenc",        "kind": "nvenc",        "label": "h264_nvenc (NVIDIA GPU)"},
    {"codec": "hevc_nvenc",        "kind": "nvenc",        "label": "hevc_nvenc (NVIDIA GPU)"},
    {"codec": "h264_videotoolbox", "kind": "videotoolbox", "label": "h264_videotoolbox (Apple GPU)"},
    {"codec": "hevc_videotoolbox", "kind": "videotoolbox", "label": "hevc_videotoolbox (Apple GPU)"},
    {"codec": "libx264",           "kind": "cpu",          "label": "libx264 (CPU)"},
    {"codec": "libx265",           "kind": "cpu",          "label": "libx265 (CPU)"},
]


def available_encoders():
    """Probe `ffmpeg -encoders` and return the ENC_CANDIDATES that are present.

    Always guarantees at least libx264 (CPU) so the GUI is usable; if ffmpeg
    can't be run at all, returns the libx264 entry as a best-effort fallback.
    """
    present = set()
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15)
        text = out.stdout + out.stderr
        for c in ENC_CANDIDATES:
            # encoder lines look like " V....D h264_nvenc   NVIDIA NVENC ..."
            if re.search(r"\b" + re.escape(c["codec"]) + r"\b", text):
                present.add(c["codec"])
    except Exception:
        pass
    encs = [c for c in ENC_CANDIDATES if c["codec"] in present]
    if not any(c["kind"] == "cpu" for c in encs):
        encs.append(ENC_CANDIDATES[4])  # ensure libx264 fallback is always offered
    return encs


# =============================================================================
# Well-grid detection  (self-contained copy of the project's crop_wells logic)
# =============================================================================
def box_with_margin(cell, margin, W, H):
    """Grow (margin>0) or shrink (margin<0) a cell box by `margin` px, clamped.

    Width/height are forced EVEN (yuv420p requires it; libx264 errors otherwise).
    """
    x, y, w, h = cell
    x0 = max(0, x - margin); y0 = max(0, y - margin)
    x1 = min(W, x + w + margin); y1 = min(H, y + h + margin)
    bw = (x1 - x0) & ~1  # round width/height down to even
    bh = (y1 - y0) & ~1
    return x0, y0, bw, bh


def median_frame(path, n=15):
    """Median of n evenly-spaced frames: removes moving worms, leaves the grid."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 7500
    idx = np.linspace(total * 0.1, total * 0.9, n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {os.path.basename(path)}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def bright_runs(profile, frac_thresh=0.5, min_len=80):
    """Contiguous BRIGHT runs (wells) in a 1-D fraction-bright profile."""
    bright = profile > frac_thresh
    runs, s = [], None
    for i, b in enumerate(bright):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s >= min_len:
                runs.append((s, i))
            s = None
    if s is not None and len(profile) - s >= min_len:
        runs.append((s, len(profile)))
    return runs


def detect_cells(gray):
    """Return (cells, bw). cells = [(x,y,w,h)] row-major for full wells only."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bwf = bw > 0
    cols = bright_runs(bwf.mean(axis=0))
    rows = bright_runs(bwf.mean(axis=1))
    if not cols or not rows:
        return [], bw

    def keep_full(runs):
        spans = np.array([e - s for s, e in runs])
        med = np.median(spans)
        return [r for r, sp in zip(runs, spans) if sp >= 0.6 * med]

    cols, rows = keep_full(cols), keep_full(rows)
    cells = []
    for (y0, y1) in rows:
        for (x0, x1) in cols:
            cells.append((x0, y0, x1 - x0, y1 - y0))
    return cells, bw


def detect_boxes(path, margin):
    """median frame -> detect cells -> apply margin. Returns (boxes, (W,H))."""
    gray = median_frame(path)
    H, W = gray.shape
    cells, _ = detect_cells(gray)
    boxes = [box_with_margin(c, margin, W, H) for c in cells]
    return boxes, (W, H)


def grid_overlay(path, margin):
    """Return a BGR image with the detected grid drawn (for preview)."""
    gray = median_frame(path)
    H, W = gray.shape
    cells, _ = detect_cells(gray)
    boxes = [box_with_margin(c, margin, W, H) for c in cells]
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.putText(vis, str(i), (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return vis, len(boxes)


def detect_content_box(gray, margin, frac=0.02):
    """Bounding box of the bright imaged area (trims the black border).

    Cameras differ in field of view, so each frame has a black surround. We
    Otsu-binarise the median frame, then take the outer extent of the bright
    region (ignoring rows/cols that are <`frac` bright, i.e. stray pixels) and
    grow it OUTWARD by `margin` px so a thin black border is kept.
    Returns (x, y, w, h).
    """
    H, W = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bwf = bw > 0
    cols = np.where(bwf.mean(axis=0) > frac)[0]
    rows = np.where(bwf.mean(axis=1) > frac)[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, W, H)  # nothing bright -> don't crop
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return box_with_margin((x0, y0, x1 - x0, y1 - y0), margin, W, H)


def content_overlay(path, margin):
    """BGR median frame with the detected content (trim) box drawn (preview)."""
    gray = median_frame(path)
    H, W = gray.shape
    x, y, w, h = detect_content_box(gray, margin)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return vis, 1


def probe_duration(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


# =============================================================================
# GUI
# =============================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🪓 Chop & Drop — chop the wells, drop the frames")
        self.geometry("780x880")
        self.minsize(720, 800)

        self.files = []
        self.worker = None
        self.cancel_flag = threading.Event()
        self.msg_q = queue.Queue()
        self._preview_img = None  # keep ref so Tk doesn't GC it

        self._apply_style()
        self._build_ui()
        self.after(100, self._drain_queue)
        self._startup_checks()

    # minimalistic light palette (soft neutrals + one calm accent)
    PALETTE = {
        "bg": "#F4F6F9", "surface": "#FFFFFF", "text": "#23272F",
        "muted": "#98A1AE", "border": "#E2E6EC", "trough": "#E8EBF0",
        "accent": "#3B82C4", "accent_dark": "#2F6CA6", "accent_text": "#FFFFFF",
    }

    def _apply_style(self):
        """Modern minimalistic look: flat theme, native UI font, soft light palette."""
        c = self.PALETTE
        style = ttk.Style(self)
        if "clam" in style.theme_names():     # flat/clean; supports colour config
            style.theme_use("clam")

        # pick the first modern UI font that's actually installed
        families = set(tkfont.families())
        prefs = ["Segoe UI", "SF Pro Text", "SF Pro Display", "Helvetica Neue",
                 "Inter", "Ubuntu", "Cantarell", "Noto Sans", "Roboto", "DejaVu Sans"]
        self.ui_family = next((f for f in prefs if f in families), None)
        self.ui_size = 10
        if self.ui_family:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    tkfont.nametofont(name).configure(family=self.ui_family, size=self.ui_size)
                except tk.TclError:
                    pass
        base = (self.ui_family or "TkDefaultFont", self.ui_size)

        self.configure(bg=c["bg"])
        style.configure(".", background=c["bg"], foreground=c["text"],
                        fieldbackground=c["surface"], bordercolor=c["border"], font=base)
        style.configure("TFrame", background=c["bg"])
        style.configure("TLabel", background=c["bg"], foreground=c["text"])
        style.configure("Footer.TLabel", background=c["bg"], foreground=c["muted"],
                        font=(self.ui_family or "TkDefaultFont", 9))
        style.configure("TLabelframe", background=c["bg"], bordercolor=c["border"],
                        relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=c["bg"], foreground=c["text"],
                        font=(self.ui_family or "TkDefaultFont", self.ui_size, "bold"))
        for w in ("TCheckbutton", "TRadiobutton"):
            style.configure(w, background=c["bg"], foreground=c["text"])
            style.map(w, background=[("active", c["bg"])])

        # neutral buttons + one accent (primary) style
        style.configure("TButton", background=c["surface"], foreground=c["text"],
                        bordercolor=c["border"], focuscolor=c["bg"], padding=(12, 6), relief="flat")
        style.map("TButton", background=[("active", "#ECEFF3"), ("pressed", "#E3E7ED"),
                                         ("disabled", c["bg"])],
                  bordercolor=[("active", c["accent"])], foreground=[("disabled", c["muted"])])
        style.configure("Accent.TButton", background=c["accent"], foreground=c["accent_text"],
                        bordercolor=c["accent"], padding=(16, 6))
        style.map("Accent.TButton",
                  background=[("active", c["accent_dark"]), ("pressed", c["accent_dark"]),
                             ("disabled", "#BBC4D0")],
                  foreground=[("disabled", "#EEF2F6")])

        # inputs
        for w in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(w, fieldbackground=c["surface"], background=c["surface"],
                            foreground=c["text"], bordercolor=c["border"], arrowcolor=c["text"])
        style.map("TCombobox", fieldbackground=[("readonly", c["surface"])],
                  foreground=[("readonly", c["text"])])

        # progress bars + scrollbar
        style.configure("TProgressbar", background=c["accent"], troughcolor=c["trough"],
                        bordercolor=c["border"], lightcolor=c["accent"], darkcolor=c["accent"])
        style.configure("TScrollbar", background=c["surface"], troughcolor=c["bg"],
                        bordercolor=c["border"], arrowcolor=c["text"])

    def _startup_checks(self):
        if shutil.which(FFMPEG) is None and not os.path.exists(FFMPEG):
            messagebox.showwarning(
                "FFmpeg not found",
                "Could not find 'ffmpeg' on your PATH.\n\n"
                "Install it first:\n"
                "  • Ubuntu/Debian:  sudo apt install ffmpeg\n"
                "  • macOS (Homebrew):  brew install ffmpeg\n"
                "  • Windows:  download from ffmpeg.org and add it to PATH")
            return
        kinds = {c["kind"] for c in self.encoders}
        if "nvenc" in kinds:
            self._log(f"Encoders: NVIDIA NVENC available. Default = {self.enc_var.get()}.\n")
        elif "videotoolbox" in kinds:
            self._log(f"Encoders: no NVENC; Apple VideoToolbox available. Default = {self.enc_var.get()}.\n")
        else:
            self._log("Encoders: no GPU encoder found — using CPU (libx264). This is normal "
                      "on Macs / non-NVIDIA PCs; conversion just runs on the CPU.\n")

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        c = self.PALETTE

        # ----- input files -----
        frm_files = ttk.LabelFrame(self, text="Input videos")
        frm_files.pack(fill="both", expand=True, **pad)
        self.listbox = tk.Listbox(frm_files, selectmode=tk.EXTENDED, height=7,
                                  font=(self.ui_family or "TkDefaultFont", self.ui_size),
                                  borderwidth=0, highlightthickness=1, activestyle="none",
                                  bg=c["surface"], fg=c["text"],
                                  selectbackground=c["accent"], selectforeground=c["accent_text"],
                                  highlightbackground=c["border"], highlightcolor=c["accent"])
        self.listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb = ttk.Scrollbar(frm_files, command=self.listbox.yview)
        sb.pack(side="left", fill="y", pady=8)
        self.listbox.config(yscrollcommand=sb.set)
        b = ttk.Frame(frm_files); b.pack(side="left", fill="y", padx=8, pady=8)
        ttk.Button(b, text="Add files…", command=self.add_files).pack(fill="x", pady=2)
        ttk.Button(b, text="Add folder…", command=self.add_folder).pack(fill="x", pady=2)
        self.recurse_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(b, text="Include\nsubfolders", variable=self.recurse_var).pack(anchor="w", pady=2)
        ttk.Button(b, text="Remove", command=self.remove_selected).pack(fill="x", pady=2)
        ttk.Button(b, text="Clear", command=self.clear_files).pack(fill="x", pady=2)

        # ----- crop choice -----
        frm_op = ttk.LabelFrame(self, text="Crop  (chop)")
        frm_op.pack(fill="x", **pad)
        self.crop_mode = tk.StringVar(value="wells")
        crops = [
            ("No crop", "none"),
            ("Crop wells (multi-well plate)", "wells"),
            ("Trim black border (single FOV)", "trim"),
        ]
        for i, (label, val) in enumerate(crops):
            ttk.Radiobutton(frm_op, text=label, variable=self.crop_mode, value=val,
                            command=self._sync_enabled).grid(row=0, column=i, sticky="w", padx=10, pady=6)

        # ----- settings grid -----
        frm_set = ttk.LabelFrame(self, text="Settings")
        frm_set.pack(fill="x", **pad)

        self.fps_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_set, text="Drop frame rate to:", variable=self.fps_on,
                        command=self._sync_enabled).grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.fps_var = tk.StringVar(value="7.5")
        self.fps_entry = ttk.Entry(frm_set, textvariable=self.fps_var, width=8)
        self.fps_entry.grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(frm_set, text="Crop margin (px):").grid(row=0, column=2, sticky="e", padx=8, pady=4)
        self.margin_var = tk.StringVar(value="12")
        self.margin_entry = ttk.Entry(frm_set, textvariable=self.margin_var, width=8)
        self.margin_entry.grid(row=0, column=3, sticky="w", pady=4)

        # encoders detected from this machine's ffmpeg (best HW option first)
        self.encoders = available_encoders()
        self.enc_specs = {c["label"]: c for c in self.encoders}
        ttk.Label(frm_set, text="Encoder:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.enc_var = tk.StringVar(value=self.encoders[0]["label"])
        ttk.Combobox(frm_set, textvariable=self.enc_var, width=28, state="readonly",
                     values=[c["label"] for c in self.encoders]
                     ).grid(row=1, column=1, columnspan=2, sticky="w", pady=4)

        ttk.Label(frm_set, text="Quality (CQ/CRF):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.q_var = tk.StringVar(value="19")
        ttk.Spinbox(frm_set, from_=0, to=51, textvariable=self.q_var, width=8
                    ).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(frm_set, text="GPU index:").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.gpu_var = tk.StringVar(value="1")
        ttk.Spinbox(frm_set, from_=0, to=7, textvariable=self.gpu_var, width=8
                    ).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(frm_set, text="(NVIDIA only; on the MAT rig GPU 0 = live recorder)",
                  foreground=c["muted"]).grid(row=3, column=2, columnspan=2, sticky="w", padx=8)

        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_set, text="Overwrite existing outputs",
                        variable=self.overwrite_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # ----- output -----
        frm_out = ttk.LabelFrame(self, text="Output")
        frm_out.pack(fill="x", **pad)
        ttk.Label(frm_out, text="Output folder:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.outdir_var = tk.StringVar(value="")
        ttk.Entry(frm_out, textvariable=self.outdir_var).grid(row=0, column=1, sticky="we", pady=4)
        ttk.Button(frm_out, text="Browse…", command=self.pick_outdir).grid(row=0, column=2, padx=8)
        ttk.Label(frm_out, text="(blank = beside each source video)",
                  foreground=c["muted"]).grid(row=1, column=1, sticky="w")
        frm_out.columnconfigure(1, weight=1)

        # ----- progress -----
        frm_p = ttk.Frame(self); frm_p.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm_p, textvariable=self.status_var).pack(anchor="w", padx=8)
        ttk.Label(frm_p, text="Current file:").pack(anchor="w", padx=8)
        self.file_pb = ttk.Progressbar(frm_p, mode="determinate", maximum=100)
        self.file_pb.pack(fill="x", padx=8, pady=2)
        ttk.Label(frm_p, text="Overall:").pack(anchor="w", padx=8)
        self.total_pb = ttk.Progressbar(frm_p, mode="determinate", maximum=100)
        self.total_pb.pack(fill="x", padx=8, pady=2)

        # ----- log -----
        frm_log = ttk.LabelFrame(self, text="Log / report")
        frm_log.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(frm_log, height=8, wrap="word", state="disabled",
                           font=(self.ui_family or "TkDefaultFont", self.ui_size),
                           borderwidth=0, highlightthickness=1, padx=6, pady=4,
                           bg=c["surface"], fg=c["text"], insertbackground=c["text"],
                           highlightbackground=c["border"], highlightcolor=c["accent"])
        self.log.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        lsb = ttk.Scrollbar(frm_log, command=self.log.yview)
        lsb.pack(side="left", fill="y", pady=8)
        self.log.config(yscrollcommand=lsb.set)

        # ----- actions -----
        frm_a = ttk.Frame(self); frm_a.pack(fill="x", **pad)
        self.preview_btn = ttk.Button(frm_a, text="Preview crop", command=self.preview_grid)
        self.preview_btn.pack(side="left", padx=8)
        self.convert_btn = ttk.Button(frm_a, text="Convert", command=self.start, style="Accent.TButton")
        self.convert_btn.pack(side="left", padx=8)
        self.cancel_btn = ttk.Button(frm_a, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left")

        # ----- footer credit (minimalistic) -----
        ttk.Label(self, text="Chop & Drop  ·  MAT System  ·  Serhat Turkmen  ·  2026",
                  style="Footer.TLabel", anchor="e").pack(fill="x", padx=12, pady=(0, 6))

        self._sync_enabled()

    def _sync_enabled(self):
        uses_crop = self.crop_mode.get() in ("wells", "trim")
        self.fps_entry.config(state="normal" if self.fps_on.get() else "disabled")
        self.margin_entry.config(state="normal" if uses_crop else "disabled")
        self.preview_btn.config(state="normal" if uses_crop else "disabled")

    # ---------------- file handling ----------------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select recordings",
            filetypes=[("Video files", " ".join("*" + e for e in VIDEO_EXTS)), ("All files", "*.*")])
        for p in paths:
            self._add(p)

    def add_folder(self):
        d = filedialog.askdirectory(title="Select folder")
        if not d:
            return
        before = len(self.files)
        if self.recurse_var.get():
            for root, _dirs, names in os.walk(d):
                for name in sorted(names):
                    if name.lower().endswith(VIDEO_EXTS):
                        self._add(os.path.join(root, name))
        else:
            for name in sorted(os.listdir(d)):
                if name.lower().endswith(VIDEO_EXTS):
                    self._add(os.path.join(d, name))
        added = len(self.files) - before
        scope = "subfolders included" if self.recurse_var.get() else "top level only"
        self._log(f"Added {added} video(s) from {d} ({scope}).\n")

    def _add(self, p):
        if p not in self.files:
            self.files.append(p)
            self.listbox.insert(tk.END, p)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            del self.files[i]

    def clear_files(self):
        self.listbox.delete(0, tk.END)
        self.files.clear()

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir_var.set(d)

    def _selected_or_first(self):
        sel = self.listbox.curselection()
        if sel:
            return self.files[sel[0]]
        return self.files[0] if self.files else None

    # ---------------- preview ----------------
    def preview_grid(self):
        src = self._selected_or_first()
        if not src:
            messagebox.showwarning("No file", "Add (and optionally select) a video first.")
            return
        try:
            margin = int(self.margin_var.get())
        except ValueError:
            margin = 12
        cm = self.crop_mode.get()
        self._log(f"Detecting {'wells' if cm=='wells' else 'content border'} on {os.path.basename(src)} …\n")
        threading.Thread(target=self._preview_worker, args=(src, margin, cm), daemon=True).start()

    def _preview_worker(self, src, margin, cm):
        try:
            vis, n = (grid_overlay(src, margin) if cm == "wells"
                      else content_overlay(src, margin))
        except Exception as e:
            self.msg_q.put(("log", f"Preview failed: {e}\n"))
            return
        # scale so the longest side <= ~900 px, save PNG (Tk PhotoImage reads PNG)
        H, W = vis.shape[:2]
        scale = max(1, int(round(max(W, H) / 900.0)))
        small = vis[::scale, ::scale]
        tmp = os.path.join(os.path.dirname(os.path.abspath(src)), ".chopdrop_preview.png")
        cv2.imwrite(tmp, small)
        kind = "wells" if cm == "wells" else "content box"
        self.msg_q.put(("preview", (tmp, n, os.path.basename(src), kind)))

    def _show_preview(self, tmp, n, name, kind):
        win = tk.Toplevel(self)
        win.title(f"Crop preview — {name}: {n} {kind}")
        img = tk.PhotoImage(file=tmp)
        self._preview_img = img  # keep ref
        ttk.Label(win, text=f"{name} — detected {n} {kind} (red box incl. margin)").pack(padx=8, pady=6)
        tk.Label(win, image=img).pack(padx=8, pady=8)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)

    # ---------------- conversion ----------------
    def _enc_opts(self):
        """Return (ffmpeg output options, kind) for the selected encoder."""
        spec = self.enc_specs.get(self.enc_var.get(), self.encoders[0])
        codec, kind = spec["codec"], spec["kind"]
        try:
            q = int(self.q_var.get().strip())
        except ValueError:
            q = 19
        if kind == "nvenc":
            opts = ["-c:v", codec, "-preset", "p5", "-rc", "vbr",
                    "-cq", str(q), "-b:v", "0", "-pix_fmt", "yuv420p"]
        elif kind == "videotoolbox":
            # VideoToolbox uses -q:v 1..100 (higher = better); map from CQ 0..51.
            vt = max(1, min(100, round((51 - q) / 51 * 100)))
            opts = ["-c:v", codec, "-q:v", str(vt), "-pix_fmt", "yuv420p"]
        else:  # cpu (libx264 / libx265)
            opts = ["-c:v", codec, "-preset", "veryfast", "-crf", str(q), "-pix_fmt", "yuv420p"]
        return opts, kind

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.files:
            messagebox.showwarning("No files", "Add at least one video.")
            return
        crop = self.crop_mode.get()
        fps_on = self.fps_on.get()
        if crop == "none" and not fps_on:
            messagebox.showwarning("Nothing to do", "Pick a crop option and/or enable 'Drop frame rate'.")
            return
        if fps_on:
            try:
                assert float(self.fps_var.get()) > 0
            except Exception:
                messagebox.showerror("Invalid FPS", "Target FPS must be a positive number.")
                return
        out_root = self.outdir_var.get().strip()
        if out_root and not os.path.isdir(out_root):
            messagebox.showerror("Output folder", "Output folder does not exist.")
            return

        self.cancel_flag.clear()
        self.convert_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.preview_btn.config(state="disabled")
        enc_opts, enc_kind = self._enc_opts()
        params = dict(
            files=list(self.files), crop=crop, fps_on=fps_on,
            fps=self.fps_var.get().strip(),
            margin=int(self.margin_var.get() or 12),
            enc_opts=enc_opts, enc_kind=enc_kind,
            gpu=self.gpu_var.get().strip() or "1",
            overwrite=self.overwrite_var.get(),
            out_root=out_root,
        )
        self.worker = threading.Thread(target=self._run, args=(params,), daemon=True)
        self.worker.start()

    def cancel(self):
        self.cancel_flag.set()
        self._log("Cancel requested …\n")

    def _out_dir_for(self, src, p):
        return p["out_root"] if p["out_root"] else os.path.dirname(os.path.abspath(src))

    def _run(self, p):
        n = len(p["files"])
        accel = f"NVIDIA GPU {p['gpu']}" if p["enc_kind"] == "nvenc" else (
            "Apple GPU (VideoToolbox)" if p["enc_kind"] == "videotoolbox" else "CPU")
        report = ["Chop & Drop — run report",
                  f"crop={p['crop']} fps={'off' if not p['fps_on'] else p['fps']} "
                  f"margin={p['margin']} encoder={p['enc_opts'][1]} accel={accel}", ""]
        for idx, src in enumerate(p["files"]):
            if self.cancel_flag.is_set():
                break
            base = os.path.basename(src)
            self.msg_q.put(("status", f"[{idx+1}/{n}] {base}"))
            self.msg_q.put(("total", int(idx / n * 100)))
            self.msg_q.put(("file", 0))
            try:
                line = self._process_one(src, p)
            except Exception as e:
                line = f"ERROR {base}: {e}"
                self.msg_q.put(("log", line + "\n"))
            report.append(line)
            if self.cancel_flag.is_set():
                break

        self.msg_q.put(("total", 100))
        # write report next to output (or first source)
        rep_dir = p["out_root"] if p["out_root"] else (
            os.path.dirname(os.path.abspath(p["files"][0])) if p["files"] else ".")
        try:
            rep_path = os.path.join(rep_dir, "conversion_report.txt")
            with open(rep_path, "w") as fh:
                fh.write("\n".join(report) + "\n")
            self.msg_q.put(("log", f"\nReport written: {rep_path}\n"))
        except Exception:
            pass
        self.msg_q.put(("status", "Cancelled." if self.cancel_flag.is_set() else "All done."))
        self.msg_q.put(("done", None))

    def _process_one(self, src, p):
        base = os.path.basename(src)
        stem, ext = os.path.splitext(base)
        outdir = self._out_dir_for(src, p)
        duration = probe_duration(src)
        enc_opts = p["enc_opts"]
        # CUDA decode + per-GPU pinning only applies to NVENC. VideoToolbox is
        # encode-only HW accel (no input flags); CPU needs neither.
        in_opts = ["-hwaccel", "cuda", "-hwaccel_device", "0"] if p["enc_kind"] == "nvenc" else []
        env = dict(os.environ)
        if p["enc_kind"] == "nvenc":
            env["CUDA_VISIBLE_DEVICES"] = p["gpu"]  # physical GPU -> appears as device 0
        ow = ["-y"] if p["overwrite"] else ["-n"]
        fps_suffix = f"_{p['fps']}fps" if p["fps_on"] else ""
        oext = ext if ext else ".mp4"

        def vf_chain(crop_str=None):
            parts = []
            if crop_str:
                parts.append(crop_str)
            if p["fps_on"]:
                parts.append(f"fps={p['fps']}")
            return ",".join(parts)

        # ---- single-output modes (no crop, or trim-border) ----
        if p["crop"] in ("none", "trim"):
            if p["crop"] == "trim":
                self.msg_q.put(("log", f"Detecting content border in {base} …\n"))
                gray = median_frame(src)
                x, y, w, h = detect_content_box(gray, p["margin"])
                crop_str = f"crop={w}:{h}:{x}:{y}"
                tag = "trim"
                out = os.path.join(outdir, f"{stem}_trim{fps_suffix}{oext}")
                self.msg_q.put(("log", f"  content box {w}x{h} @ ({x},{y})\n"))
            else:
                crop_str = None
                tag = "fps"
                out = os.path.join(outdir, f"{stem}{fps_suffix or '_reencode'}{oext}")

            if os.path.abspath(out) == os.path.abspath(src):
                self.msg_q.put(("log", f"SKIP (out==in): {base}\n")); return f"SKIP out==in: {base}"
            if os.path.exists(out) and not p["overwrite"]:
                self.msg_q.put(("log", f"SKIP (exists): {out}\n")); return f"SKIP exists: {base}"

            vf = vf_chain(crop_str)
            cmd = [FFMPEG, "-nostdin"] + ow + in_opts + ["-i", src]
            if vf:
                cmd += ["-vf", vf]
            cmd += enc_opts + ["-an", "-progress", "pipe:1", "-nostats", out]
            self.msg_q.put(("log", f"{tag}  {base} -> {os.path.basename(out)}\n"))
            rc = self._run_ffmpeg(cmd, duration, env)
            return f"{'OK ' if rc==0 else 'FAIL'} {tag} {base}"

        # ---- crop wells -> one decode pass, many outputs ----
        self.msg_q.put(("log", f"Detecting wells in {base} …\n"))
        boxes, (W, H) = detect_boxes(src, p["margin"])
        if not boxes:
            self.msg_q.put(("log", f"  no full wells detected in {base}\n"))
            return f"SKIP no-wells: {base}"
        well_dir = os.path.join(outdir, stem)
        os.makedirs(well_dir, exist_ok=True)
        with open(os.path.join(well_dir, f"{stem}_wells.csv"), "w") as fh:
            fh.write("well,x,y,w,h\n")
            for i, (x, y, w, h) in enumerate(boxes, 1):
                fh.write(f"{i:02d},{x},{y},{w},{h}\n")
        first = os.path.join(well_dir, f"{stem}_well01{fps_suffix}.mp4")
        if os.path.exists(first) and not p["overwrite"]:
            self.msg_q.put(("log", f"SKIP (exists): {well_dir}\n"))
            return f"SKIP exists: {base} ({len(boxes)} wells)"

        cmd = [FFMPEG, "-nostdin"] + ow + in_opts + ["-i", src]
        for i, (x, y, w, h) in enumerate(boxes, 1):
            out = os.path.join(well_dir, f"{stem}_well{i:02d}{fps_suffix}.mp4")
            cmd += ["-filter:v", vf_chain(f"crop={w}:{h}:{x}:{y}")] + enc_opts + ["-an", out]
        cmd += ["-progress", "pipe:1", "-nostats"]
        tag = "wells+fps" if p["fps_on"] else "wells"
        self.msg_q.put(("log", f"{tag}  {base}: {len(boxes)} wells -> {well_dir} (one decode pass)\n"))
        rc = self._run_ffmpeg(cmd, duration, env)
        return f"{'OK ' if rc==0 else 'FAIL'} {tag} {base}: {len(boxes)} wells"

    def _run_ffmpeg(self, cmd, duration, env):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env)
        except Exception as e:
            self.msg_q.put(("log", f"Failed to launch ffmpeg: {e}\n"))
            return -1
        time_re = re.compile(r"out_time_ms=(\d+)")
        err_tail = []
        for line in proc.stdout:
            if self.cancel_flag.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                return -2
            line = line.strip()
            m = time_re.search(line)
            if m and duration:
                secs = int(m.group(1)) / 1_000_000.0
                self.msg_q.put(("file", max(0, min(100, int(secs / duration * 100)))))
            elif "error" in line.lower():
                err_tail.append(line)
        proc.wait()
        if proc.returncode != 0 and err_tail:
            self.msg_q.put(("log", "  " + "\n  ".join(err_tail[-6:]) + "\n"))
        if proc.returncode == 0:
            self.msg_q.put(("file", 100))
        return proc.returncode

    # ---------------- UI queue pump ----------------
    def _drain_queue(self):
        try:
            while True:
                kind, val = self.msg_q.get_nowait()
                if kind == "status":
                    self.status_var.set(val)
                elif kind == "file":
                    self.file_pb["value"] = val
                elif kind == "total":
                    self.total_pb["value"] = val
                elif kind == "log":
                    self._log(val)
                elif kind == "preview":
                    self._show_preview(*val)
                elif kind == "done":
                    self.convert_btn.config(state="normal")
                    self.cancel_btn.config(state="disabled")
                    self._sync_enabled()
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.config(state="disabled")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
