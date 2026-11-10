"""EDITH - a glassy Jarvis/EDITH-style HUD for a voice assistant.

Run:  python assistant_interface.py
Needs assistant_core.py next to it and a .env file (see .env.example).
tkinter ships with the normal Windows Python installer.

Press Ctrl+E anywhere to start or stop the assistant.
"""
import ctypes
import math
import queue
import threading
import time
import tkinter as tk
from ctypes import wintypes

import assistant_core as core

NAME = core.ASSISTANT_NAME.upper()

# --- Glassy colour palette ---------------------------------------------------
BG          = "#060a14"   # deep navy-black
GLASS       = "#0a1424"   # dark glass panel
GLASS_LIGHT = "#0e1a2e"   # hover / active glass
GLASS_DEEP  = "#040810"   # log background
GLOW_OUT    = "#002838"   # outer glow ring
GLOW_MID    = "#004060"   # middle glow ring
CYAN        = "#00e5ff"
CYAN_DIM    = "#0b5560"
AMBER       = "#ffb300"
RED         = "#ff3d57"
VIOLET      = "#9b7bff"
MINT        = "#2dffb2"
TEXT        = "#c8e8ff"
TEXT_DIM    = "#486078"
FONT        = "Consolas"

# --- Global hotkey (Ctrl+E) --------------------------------------------------
_MOD_CONTROL = 0x0002
_VK_E = 0x45
_WM_HOTKEY = 0x0312
_HOTKEY_ID = 9001


def _start_hotkey_listener(root, callback):
    """Register Ctrl+E as a global hotkey; call *callback* on the main thread."""
    user32 = ctypes.windll.user32
    if not user32.RegisterHotKey(None, _HOTKEY_ID, _MOD_CONTROL, _VK_E):
        return False

    def _listen():
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                root.after(0, callback)

    threading.Thread(target=_listen, daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# Glassy button helper
# ---------------------------------------------------------------------------

class GlassButton(tk.Label):
    """A label styled as a glassy button with hover and press states."""

    def __init__(self, parent, text, colour, command, **kw):
        super().__init__(parent, text=text, bg=GLASS, fg=colour,
                         font=(FONT, 9, "bold"), padx=14, pady=6, cursor="hand2",
                         highlightbackground=colour, highlightthickness=1, **kw)
        self._colour = colour
        self._command = command
        self.bind("<Button-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", self._hover)
        self.bind("<Leave>", self._leave)

    def _hover(self, _):
        self.config(bg=GLASS_LIGHT, highlightbackground=self._colour,
                    highlightthickness=2)

    def _leave(self, _):
        self.config(bg=GLASS, highlightthickness=1)

    def _press(self, _):
        self.config(bg=GLASS_DEEP)

    def _release(self, _):
        self.config(bg=GLASS_LIGHT)
        self._command()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GlassApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.engine = core.Assistant(emit=lambda k, t: self.q.put((k, t)))
        self.error_until = 0.0
        self.angle = 0.0
        self.tick = 0
        self.hotkey_ok = False

        # --- window ---------------------------------------------------------
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.94)
        root.geometry("370x660+100+80")
        root.configure(bg=GLOW_OUT)

        # Layered glow border (outer → middle → content)
        glow_mid = tk.Frame(root, bg=GLOW_MID)
        glow_mid.pack(fill="both", expand=True, padx=1, pady=1)
        main = tk.Frame(glow_mid, bg=BG, highlightbackground=CYAN,
                        highlightthickness=1)
        main.pack(fill="both", expand=True, padx=1, pady=1)

        # --- title bar ------------------------------------------------------
        title_bar = tk.Frame(main, bg=GLASS, highlightbackground=CYAN_DIM,
                             highlightthickness=1)
        title_bar.pack(fill="x")
        title_bar.grid_columnconfigure(0, weight=1)
        tk.Label(title_bar, text=f"◈  {NAME}  //  A.I. INTERFACE",
                 bg=GLASS, fg=CYAN, font=(FONT, 9, "bold"),
                 pady=7, padx=10).grid(row=0, column=0, sticky="w")
        close = tk.Label(title_bar, text="✕", bg=GLASS, fg=RED,
                         font=(FONT, 11, "bold"), padx=12, cursor="hand2")
        close.grid(row=0, column=1)
        close.bind("<Button-1>", lambda e: self.quit())
        for w in (title_bar,):
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)

        # glass reflection strip
        tk.Frame(main, bg=GLOW_MID, height=1).pack(fill="x")

        # --- arc reactor canvas ---------------------------------------------
        self.canvas = tk.Canvas(main, width=370, height=250, bg=BG,
                                 highlightthickness=0)
        self.canvas.pack()

        # --- status ---------------------------------------------------------
        self.status = tk.Label(main, text="OFFLINE", bg=BG, fg=TEXT_DIM,
                               font=(FONT, 12, "bold"))
        self.status.pack()

        # --- buttons --------------------------------------------------------
        row = tk.Frame(main, bg=BG)
        row.pack(pady=12)
        self.btn_start = GlassButton(row, "▶ START", CYAN, self.start)
        self.btn_mute  = GlassButton(row, "🎤 MUTE", AMBER, self.toggle_mute)
        self.btn_stop  = GlassButton(row, "■ END", RED, self.stop)
        for b in (self.btn_start, self.btn_mute, self.btn_stop):
            b.pack(side="left", padx=5)

        # --- log panel ------------------------------------------------------
        log_outer = tk.Frame(main, bg=GLASS, highlightbackground=CYAN_DIM,
                             highlightthickness=1)
        log_outer.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        self.log = tk.Text(log_outer, bg=GLASS_DEEP, fg=TEXT,
                           font=(FONT, 9), wrap="word", bd=0, padx=10, pady=8,
                           state="disabled", height=8)
        self.log.pack(fill="both", expand=True, padx=1, pady=1)
        self.log.tag_config("agent", foreground=CYAN)
        self.log.tag_config("user", foreground="#ffffff")
        self.log.tag_config("sys", foreground=AMBER)
        self.log.tag_config("err", foreground=RED)

        # --- footer ---------------------------------------------------------
        footer = tk.Frame(main, bg=BG)
        footer.pack(fill="x", pady=(0, 6))
        tk.Label(footer, text=f"Created by {core.CREATOR_NAME}",
                 bg=BG, fg=TEXT_DIM, font=(FONT, 8)).pack(side="left", padx=12)

        # --- global hotkey --------------------------------------------------
        self.hotkey_ok = _start_hotkey_listener(root, self.toggle)
        hotkey_text = "Ctrl+E to toggle" if self.hotkey_ok else "Click START to begin"
        self.hotkey_label = tk.Label(footer, text=hotkey_text, bg=BG,
                                      fg=VIOLET, font=(FONT, 8, "bold"))
        self.hotkey_label.pack(side="right", padx=12)

        # fallback binding when the window is focused
        root.bind("<Control-e>", lambda e: self.toggle())

        # --- boot -----------------------------------------------------------
        self._say("sys", f"{NAME} offline. Initiate Neural Module.")
        if not self.hotkey_ok:
            self._say("sys", "Ctrl+E is taken by another app. Click START instead.")
        else:
            self._say("sys", "Press Ctrl+E or click START to begin.")
        self._animate()
        self._poll()

    # --- dragging ------------------------------------------------------------

    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_rootx(), \
                             e.y_root - self.root.winfo_rooty()

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    # --- log -----------------------------------------------------------------

    def _say(self, tag, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")

    # --- engine actions ------------------------------------------------------

    def start(self):
        self.btn_mute.config(text="🎤 MUTE")
        if not self.engine.running:
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            self.log.config(state="disabled")
        self.engine.start()

    def stop(self):
        self.engine.stop()

    def toggle_mute(self):
        if not self.engine.running:
            return
        self.engine.muted = not self.engine.muted
        self.btn_mute.config(text="🎤 UNMUTE" if self.engine.muted else "🎤 MUTE")

    def toggle(self):
        """Called by Ctrl+E — start if idle, stop if running."""
        if self.engine.running:
            self.stop()
        else:
            self.start()

    def quit(self):
        self.engine.stop()
        try:
            ctypes.windll.user32.UnregisterHotKey(None, _HOTKEY_ID)
        except Exception:  # noqa: BLE001
            pass
        self.root.after(200, self.root.destroy)

    # --- message pump --------------------------------------------------------

    def _poll(self):
        try:
            while True:
                kind, text = self.q.get_nowait()
                if kind == "agent":
                    self._say("agent", f"{NAME}: {text}")
                elif kind == "user":
                    self._say("user", f"YOU: {text}")
                elif kind == "sys":
                    self._say("sys", f"SYSTEM: {text}")
                elif kind == "err":
                    self.error_until = time.time() + 4
                    self._say("err", f"ERROR: {text}")
                elif kind == "ended":
                    self.btn_mute.config(text="🎤 MUTE")
                    self._say("sys", "Session closed.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # --- status / colours ---------------------------------------------------

    def _look(self):
        e = self.engine
        s = e.state
        if time.time() < self.error_until and s == "offline":
            return "ERROR", RED
        if s == "offline":
            return "SYSTEM OFFLINE", TEXT_DIM
        if s == "limit":
            left = core.fmt_duration(e.blocked_until - time.time())
            return f"LIMIT REACHED — BACK IN {left}", AMBER
        if e.muted:
            return "MIC MUTED", AMBER
        return {
            "listening": ("LISTENING…", CYAN),
            "hearing":   ("HEARING YOU", CYAN),
            "thinking":  ("THINKING…", VIOLET),
            "speaking":  (f"{NAME} SPEAKING", MINT),
        }.get(s, (s.upper(), CYAN))

    # --- animated arc reactor ------------------------------------------------

    def _animate(self):
        c = self.canvas
        c.delete("all")
        self.tick += 1
        cx, cy = 185, 125
        e = self.engine
        text, col = self._look()
        self.status.config(text=text, fg=col)

        fast = e.state in ("hearing", "thinking", "speaking") and not e.muted
        self.angle += 3.4 if fast else 0.8

        # outer ring
        c.create_oval(cx - 100, cy - 100, cx + 100, cy + 100,
                      outline=col, width=2)

        # tick marks
        for i in range(36):
            a = math.radians(i * 10)
            r1, r2 = 92, 100 if i % 3 else 108
            c.create_line(cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                          cx + r2 * math.cos(a), cy + r2 * math.sin(a),
                          fill=col)

        # rotating arcs (3, offset 120°)
        for k in range(3):
            c.create_arc(cx - 80, cy - 80, cx + 80, cy + 80,
                         start=self.angle + k * 120, extent=70,
                         style="arc", outline=col, width=4)

        # inner counter-rotating arc
        c.create_arc(cx - 60, cy - 60, cx + 60, cy + 60,
                     start=-self.angle * 1.6, extent=250,
                     style="arc", outline=col, width=2)

        # core glow (3 layers, outside-in)
        if e.state == "speaking":
            pulse = 5 + 4 * math.sin(self.tick / 3)
        else:
            pulse = e.level * 18
            e.level *= 0.8
        r = 24 + pulse
        c.create_oval(cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10,
                      outline=GLOW_OUT, width=1)
        c.create_oval(cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6,
                      outline=GLOW_MID, width=1)
        c.create_oval(cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3,
                      outline=col, width=1)

        # solid core
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col, outline="")
        c.create_oval(cx - r / 2, cy - r / 2, cx + r / 2, cy + r / 2,
                      fill=BG, outline="")

        self.root.after(40, self._animate)


if __name__ == "__main__":
    root = tk.Tk()
    GlassApp(root)
    root.mainloop()
