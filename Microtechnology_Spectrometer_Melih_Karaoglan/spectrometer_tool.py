"""
Live spectrometer prototype (lab-tool oriented) with:
 - integration time slider (ms)
 - smoothing window slider
 - optional auto-saving every N seconds (toggle with 'a') [RAW only]
 - choose output folder (press 'o')
 - pause/resume acquisition (press 'p')
 - dark reference ('d') and white reference ('w')
 - corrected spectrum (T = (raw-dark)/(white-dark))
 - absorbance spectrum (A = -log10(T))
 - peak detection (within ROI) + wavelength-to-RGB color box
 - UI polish: buttons, help overlay, autoscale lock toggle, ROI sliders
 - multi-view toggles: Raw / Smoothed / Corrected / Absorbance on same plot

Keys:
  d = capture dark reference (saves references/dark_reference.csv)
  w = capture white reference (saves references/white_reference.csv)
  s = manual save (saves selected curves; corrected/absorbance only if refs exist)
  a = toggle auto-save ON/OFF (RAW only)
  o = choose output folder
  p = pause/resume acquisition
  x = toggle autoscale ON/OFF (lock axes)
  h = toggle help overlay
  q = quit

Author: Melih Karaoglan
"""

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

# ---- seabreeze import ----
try:
    from seabreeze.spectrometers import Spectrometer
    SEABREEZE_AVAILABLE = True
except ImportError:
    Spectrometer = None
    SEABREEZE_AVAILABLE = False


# ---- settings ----
SAVE_INTERVAL_SECONDS = 2.0

DEFAULT_INT_TIME_MS = 50
MIN_INT_TIME_MS = 1
MAX_INT_TIME_MS = 500

DEFAULT_SMOOTH_WIN = 31
MIN_SMOOTH_WIN = 3
MAX_SMOOTH_WIN = 101

# ROI defaults (nm)
DEFAULT_ROI_MIN = 400
DEFAULT_ROI_MAX = 800
MIN_ROI = 350
MAX_ROI = 1050

# ---- globals/state ----
integration_time_us = DEFAULT_INT_TIME_MS * 1000
smoothing_window = DEFAULT_SMOOTH_WIN

roi_min_nm = DEFAULT_ROI_MIN
roi_max_nm = DEFAULT_ROI_MAX

last_save_time = 0.0
last_wavelengths = None
last_raw = None

dark_ref = None
white_ref = None

running = True
auto_save_enabled = False
acquisition_enabled = True
autoscale_enabled = True

DEFAULT_SAVE_DIR = os.path.join(os.getcwd(), "data_autosave")
save_dir = DEFAULT_SAVE_DIR

# Spectrometer cache
spec_cached = None
spec_name = None


# ---- folder picker (Windows-friendly) ----
def choose_output_folder(initial_dir: str):
    """Native folder selection dialog using tkinter. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(initialdir=initial_dir, title="Select output folder for CSV files")
        root.destroy()
        return folder if folder else None
    except Exception as e:
        print(f"[WARN] Folder picker unavailable ({e}).")
        return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ts_now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv_with_metadata(path: str, wavelengths: np.ndarray, values: np.ndarray, meta: dict):
    """
    Save CSV with metadata lines starting with '#', then header, then data.
    This is super useful for reproducibility (lab/demo).
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("wavelength_nm,value\n")
        for wl, val in zip(wavelengths, values):
            f.write(f"{float(wl)},{float(val)}\n")
    print(f"[INFO] Saved: {path}")


def make_paths(base_dir: str):
    """Return organized subfolders."""
    return {
        "raw": os.path.join(base_dir, "raw"),
        "corrected": os.path.join(base_dir, "corrected"),
        "absorbance": os.path.join(base_dir, "absorbance"),
        "references": os.path.join(base_dir, "references"),
    }


# ---- color helpers ----
def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8):
    """Approximate visible color for a wavelength (380–780 nm) as RGB(0–255)."""
    wl = wavelength_nm
    if wl < 380 or wl > 780:
        return (0, 0, 0)

    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl <= 700:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

    def adj(c):
        return int(255 * ((c * factor) ** gamma)) if c > 0 else 0

    return (adj(r), adj(g), adj(b))


# ---- acquisition ----
def connect_spectrometer():
    """Connect once and cache the spectrometer object."""
    global spec_cached, spec_name
    if not SEABREEZE_AVAILABLE:
        raise RuntimeError("Seabreeze not available.")
    spec_cached = Spectrometer.from_first_available()
    try:
        spec_name = str(spec_cached)
    except Exception:
        spec_name = "Spectrometer"
    # apply current integration time
    spec_cached.integration_time_micros(integration_time_us)
    return spec_cached


def acquire_real_spectrum():
    """Acquire one spectrum using cached spectrometer (reconnect on failure)."""
    global spec_cached
    if spec_cached is None:
        connect_spectrometer()

    try:
        spec_cached.integration_time_micros(integration_time_us)
        return spec_cached.wavelengths(), spec_cached.intensities()
    except Exception as e:
        # try reconnect once
        print(f"[WARN] Spectrometer read failed ({type(e).__name__}). Reconnecting...")
        spec_cached = None
        connect_spectrometer()
        return spec_cached.wavelengths(), spec_cached.intensities()


def acquire_dummy_spectrum():
    """Dummy spectrum for testing without hardware (scaled by integration time)."""
    wl = np.linspace(400, 800, 2000)
    peak1 = np.exp(-0.5 * ((wl - 500) / 5) ** 2) * 400
    peak2 = np.exp(-0.5 * ((wl - 650) / 8) ** 2) * 250
    noise = np.random.normal(0, 5, size=wl.shape)

    scale = integration_time_us / 50_000  # reference 50 ms
    intensities = (peak1 + peak2 + 100) * scale + noise * np.sqrt(scale)
    return wl, intensities


# ---- processing ----
def smooth(data: np.ndarray, win: int) -> np.ndarray:
    """Moving average smoothing."""
    if win < 3:
        return data
    win = min(win, len(data))
    kernel = np.ones(win) / win
    return np.convolve(data, kernel, mode="same")


def compute_corrected(raw: np.ndarray, dark: np.ndarray, white: np.ndarray) -> np.ndarray:
    """T = (raw-dark)/(white-dark), safe against division by ~0."""
    denom = (white - dark)
    eps = 1e-12
    denom_safe = np.where(np.abs(denom) < eps, np.nan, denom)
    return (raw - dark) / denom_safe


def compute_absorbance(T: np.ndarray) -> np.ndarray:
    """A = -log10(T). Clip T to avoid log(<=0)."""
    T_safe = np.clip(T, 1e-9, None)
    return -np.log10(T_safe)


def within_roi(wl: np.ndarray, roi_min: float, roi_max: float):
    lo = min(roi_min, roi_max)
    hi = max(roi_min, roi_max)
    mask = (wl >= lo) & (wl <= hi)
    return mask, lo, hi


def analyze_peak_in_roi(wavelengths: np.ndarray, intensities: np.ndarray, roi_min: float, roi_max: float):
    """Return peak wavelength and value within ROI. Fallback to global if ROI mask empty."""
    mask, lo, hi = within_roi(wavelengths, roi_min, roi_max)
    if np.any(mask):
        idx_local = int(np.nanargmax(intensities[mask]))
        wl_roi = wavelengths[mask]
        it_roi = intensities[mask]
        return float(wl_roi[idx_local]), float(it_roi[idx_local]), (lo, hi)
    idx = int(np.nanargmax(intensities))
    return float(wavelengths[idx]), float(intensities[idx]), (lo, hi)


# ---- main ----
def main():
    global integration_time_us, smoothing_window, roi_min_nm, roi_max_nm
    global last_save_time, last_wavelengths, last_raw
    global running, auto_save_enabled, dark_ref, white_ref
    global acquisition_enabled, save_dir, autoscale_enabled
    global spec_cached, spec_name

    ensure_dir(save_dir)
    paths = make_paths(save_dir)
    for p in paths.values():
        ensure_dir(p)

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.40)

    # lines (we keep all, visibility controlled by checkboxes)
    line_raw,    = ax.plot([], [], label="Raw")
    line_smooth, = ax.plot([], [], label="Smoothed")
    line_corr,   = ax.plot([], [], label="Corrected (T)")
    line_abs,    = ax.plot([], [], label="Absorbance (A)")

    # ROI vertical lines (for visual feedback)
    roi_line_lo = ax.axvline(roi_min_nm, linestyle="--", linewidth=1)
    roi_line_hi = ax.axvline(roi_max_nm, linestyle="--", linewidth=1)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Value (a.u.)")
    ax.grid(True)
    ax.legend(loc="upper right")

    # status box
    info_box = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes, va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.35")
    )

    # paused overlay
    paused_text = ax.text(
        0.5, 0.5, "",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=20, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"),
        alpha=0.85
    )

    # help overlay
    help_text = ax.text(
        0.02, 0.02,
        "HELP (h)\n"
        "Keys: d(Dark) w(White) s(Save) a(Auto) o(Folder) p(Pause) x(Autoscale) q(Quit)\n"
        "Use checkboxes to show/hide curves. ROI sliders control peak search range.",
        transform=ax.transAxes, va="bottom",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.35"),
        visible=False
    )

    # ---- sliders: integration + smoothing ----
    ax_int = plt.axes([0.12, 0.28, 0.55, 0.035])
    s_int = Slider(ax=ax_int, label="Integration \ntime (ms)",
                   valmin=MIN_INT_TIME_MS, valmax=MAX_INT_TIME_MS,
                   valinit=DEFAULT_INT_TIME_MS, valstep=1)

    ax_sm = plt.axes([0.12, 0.22, 0.55, 0.035])
    s_sm = Slider(ax=ax_sm, label="Smoothing \nwindow (points)",
                  valmin=MIN_SMOOTH_WIN, valmax=MAX_SMOOTH_WIN,
                  valinit=DEFAULT_SMOOTH_WIN, valstep=2)

    # ---- sliders: ROI min/max ----
    ax_roi_min = plt.axes([0.12, 0.16, 0.55, 0.035])
    s_roi_min = Slider(ax=ax_roi_min, label="ROI min (nm)",
                       valmin=MIN_ROI, valmax=MAX_ROI,
                       valinit=DEFAULT_ROI_MIN, valstep=1)

    ax_roi_max = plt.axes([0.12, 0.10, 0.55, 0.035])
    s_roi_max = Slider(ax=ax_roi_max, label="ROI max (nm)",
                       valmin=MIN_ROI, valmax=MAX_ROI,
                       valinit=DEFAULT_ROI_MAX, valstep=1)

    def on_int(val):
        global integration_time_us, spec_cached
        integration_time_us = int(val * 1000)
        # apply immediately if device connected
        if spec_cached is not None:
            try:
                spec_cached.integration_time_micros(integration_time_us)
            except Exception:
                pass
        print(f"[INFO] Integration time set to {val:.0f} ms")

    def on_sm(val):
        global smoothing_window
        v = int(round(val))
        if v % 2 == 0:
            v += 1
        v = max(MIN_SMOOTH_WIN, min(MAX_SMOOTH_WIN, v))
        smoothing_window = v
        print(f"[INFO] Smoothing window set to {smoothing_window} points")

    def on_roi_min(val):
        global roi_min_nm
        roi_min_nm = float(val)

    def on_roi_max(val):
        global roi_max_nm
        roi_max_nm = float(val)

    s_int.on_changed(on_int)
    s_sm.on_changed(on_sm)
    s_roi_min.on_changed(on_roi_min)
    s_roi_max.on_changed(on_roi_max)

    # ---- checkboxes: which curves to show ----
    # Default: raw+smoothed ON, corrected+absorbance OFF until refs exist
    ax_checks = plt.axes([0.72, 0.12, 0.25, 0.18])
    labels = ["Raw", "Smoothed", "Corrected (T)", "Absorbance (A)"]
    actives = [True, True, False, False]
    checks = CheckButtons(ax_checks, labels, actives)

    show = {
        "Raw": True,
        "Smoothed": True,
        "Corrected (T)": False,
        "Absorbance (A)": False,
    }

    def on_check(label):
        show[label] = not show[label]

    checks.on_clicked(on_check)

    # ---- buttons ----
    btn_y = 0.02
    btn_w = 0.10
    btn_h = 0.055
    x0 = 0.12
    gap = 0.012

    ax_btn_pause = plt.axes([x0 + 0*(btn_w+gap), btn_y, btn_w, btn_h])
    ax_btn_auto  = plt.axes([x0 + 1*(btn_w+gap), btn_y, btn_w, btn_h])
    ax_btn_save  = plt.axes([x0 + 2*(btn_w+gap), btn_y, btn_w, btn_h])
    ax_btn_dark  = plt.axes([x0 + 3*(btn_w+gap), btn_y, btn_w, btn_h])
    ax_btn_white = plt.axes([x0 + 4*(btn_w+gap), btn_y, btn_w, btn_h])
    ax_btn_fold  = plt.axes([x0 + 5*(btn_w+gap), btn_y, btn_w, btn_h])

    b_pause = Button(ax_btn_pause, "Pause")
    b_auto  = Button(ax_btn_auto,  "Auto: OFF")
    b_save  = Button(ax_btn_save,  "Save")
    b_dark  = Button(ax_btn_dark,  "Dark")
    b_white = Button(ax_btn_white, "White")
    b_fold  = Button(ax_btn_fold,  "Folder")

    def refresh_paths():
        nonlocal paths
        paths = make_paths(save_dir)
        for p in paths.values():
            ensure_dir(p)

    def meta_common(source: str):
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "spectrometer": spec_name if spec_name else "N/A",
            "integration_time_ms": f"{integration_time_us/1000:.0f}",
            "smoothing_window": str(smoothing_window),
            "roi_min_nm": f"{min(roi_min_nm, roi_max_nm):.0f}",
            "roi_max_nm": f"{max(roi_min_nm, roi_max_nm):.0f}",
            "dark_ref_set": "yes" if dark_ref is not None else "no",
            "white_ref_set": "yes" if white_ref is not None else "no",
        }

    def do_toggle_pause(_=None):
        global acquisition_enabled
        acquisition_enabled = not acquisition_enabled
        b_pause.label.set_text("Pause" if acquisition_enabled else "Resume")
        print(f"[INFO] Acquisition {'RESUMED' if acquisition_enabled else 'PAUSED'}")

    def do_toggle_auto(_=None):
        global auto_save_enabled, last_save_time
        auto_save_enabled = not auto_save_enabled
        b_auto.label.set_text("Auto: ON" if auto_save_enabled else "Auto: OFF")
        last_save_time = time.time()
        print(f"[INFO] Auto-save {'ON' if auto_save_enabled else 'OFF'} (RAW only)")

    def do_folder(_=None):
        global save_dir
        picked = choose_output_folder(save_dir)
        if picked:
            save_dir = picked
            ensure_dir(save_dir)
            refresh_paths()
            print(f"[INFO] Output folder set to: {save_dir}")
        else:
            print("[INFO] Output folder selection canceled.")

    def do_dark(_=None):
        global last_raw, last_wavelengths, dark_ref
        if last_raw is None:
            print("[WARN] No data yet to store dark reference.")
            return
        dark_ref = last_raw.copy()
        path = os.path.join(paths["references"], "dark_reference.csv")
        write_csv_with_metadata(path, last_wavelengths, dark_ref, {**meta_common("capture"), "type": "dark_reference"})
        print("[INFO] Dark reference captured.")

    def do_white(_=None):
        global last_raw, last_wavelengths, white_ref
        if last_raw is None:
            print("[WARN] No data yet to store white reference.")
            return
        white_ref = last_raw.copy()
        path = os.path.join(paths["references"], "white_reference.csv")
        write_csv_with_metadata(path, last_wavelengths, white_ref, {**meta_common("capture"), "type": "white_reference"})
        print("[INFO] White reference captured.")

    def do_save(_=None):
        global last_raw, last_wavelengths
        if last_raw is None:
            print("[WARN] No data to save yet.")
            return
        t = ts_now()

        # Save what is currently selected/available (RAW always possible)
        raw_path = os.path.join(paths["raw"], f"manual_raw_{t}.csv")
        write_csv_with_metadata(raw_path, last_wavelengths, last_raw, {**meta_common("manual_save"), "type": "raw"})

        # Corrected + Abs only if refs exist
        if dark_ref is not None and white_ref is not None:
            T = compute_corrected(last_raw, dark_ref, white_ref)
            corr_path = os.path.join(paths["corrected"], f"manual_corrected_T_{t}.csv")
            write_csv_with_metadata(corr_path, last_wavelengths, T, {**meta_common("manual_save"), "type": "corrected_T"})

            A = compute_absorbance(T)
            abs_path = os.path.join(paths["absorbance"], f"manual_absorbance_A_{t}.csv")
            write_csv_with_metadata(abs_path, last_wavelengths, A, {**meta_common("manual_save"), "type": "absorbance_A"})

            print("[INFO] Saved raw + corrected(T) + absorbance(A).")
        else:
            print("[INFO] Saved raw. (Corrected/Absorbance require dark+white references.)")

    b_pause.on_clicked(do_toggle_pause)
    b_auto.on_clicked(do_toggle_auto)
    b_save.on_clicked(do_save)
    b_dark.on_clicked(do_dark)
    b_white.on_clicked(do_white)
    b_fold.on_clicked(do_folder)

    # ---- key handler ----
    def on_key(event):
        global running, autoscale_enabled

        if event.key == "q":
            running = False
            print("[INFO] Stopping.")

        elif event.key == "p":
            do_toggle_pause()

        elif event.key == "a":
            do_toggle_auto()

        elif event.key == "s":
            do_save()

        elif event.key == "d":
            do_dark()

        elif event.key == "w":
            do_white()

        elif event.key == "o":
            do_folder()

        elif event.key == "x":
            autoscale_enabled = not autoscale_enabled
            print(f"[INFO] Autoscale {'ON' if autoscale_enabled else 'OFF (locked)'}")

        elif event.key == "h":
            help_text.set_visible(not help_text.get_visible())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Live mode started.")
    print("Keys: d=dark, w=white, s=save, a=autosave, o=folder, p=pause, x=autoscale, h=help, q=quit")
    print("Curve visibility: use checkboxes (Raw/Smoothed/Corrected/Absorbance).")

    # ---- loop ----
    last_save_time = time.time()
    locked_xlim = None
    locked_ylim = None

    while running:
        if not acquisition_enabled:
            paused_text.set_text("PAUSED")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)
            continue
        else:
            paused_text.set_text("")

        # acquire
        try:
            wl, raw = acquire_real_spectrum()
            source = "real_spectrometer"
        except Exception as e:
            wl, raw = acquire_dummy_spectrum()
            source = f"dummy ({type(e).__name__})"

        last_wavelengths, last_raw = wl, raw

        # process
        sm = smooth(raw, smoothing_window)

        # references-based
        T = None
        A = None
        refs_ok = (dark_ref is not None and white_ref is not None)
        if refs_ok:
            T = compute_corrected(raw, dark_ref, white_ref)
            A = compute_absorbance(T)

        # Update ROI lines
        mask, lo, hi = within_roi(wl, roi_min_nm, roi_max_nm)
        roi_line_lo.set_xdata([lo, lo])
        roi_line_hi.set_xdata([hi, hi])

        # plot data according to checkboxes + availability
        line_raw.set_data(wl, raw)
        line_raw.set_visible(show["Raw"])

        line_smooth.set_data(wl, sm)
        line_smooth.set_visible(show["Smoothed"])

        if refs_ok:
            line_corr.set_data(wl, T)
            line_corr.set_visible(show["Corrected (T)"])
            line_abs.set_data(wl, A)
            line_abs.set_visible(show["Absorbance (A)"])
        else:
            line_corr.set_visible(False)
            line_abs.set_visible(False)

        # Peak in ROI: choose smoothed as peak basis (stable)
        peak_wl, peak_val, (roi_lo, roi_hi) = analyze_peak_in_roi(wl, sm, roi_min_nm, roi_max_nm)
        rgb = wavelength_to_rgb(peak_wl)
        face = tuple(c / 255.0 for c in rgb) if rgb != (0, 0, 0) else (1.0, 1.0, 1.0)

        # Update status box
        info_box.set_text(
            f"Source: {source}\n"
            f"Peak (ROI): {peak_wl:.1f} nm  |  ROI: {roi_lo:.0f}-{roi_hi:.0f} nm\n"
            f"Auto-save: {'ON' if auto_save_enabled else 'OFF'} (RAW)\n"
            f"Refs: Dark={'SET' if dark_ref is not None else 'NO'}  White={'SET' if white_ref is not None else 'NO'}\n"
            f"Int: {integration_time_us/1000:.0f} ms  |  Smooth: {smoothing_window}\n"
            f"Folder: {os.path.basename(save_dir) or save_dir}"
        )
        info_box.set_bbox(dict(facecolor=face, edgecolor="black", boxstyle="round,pad=0.35"))

        ax.set_title("Live spectrum tool")

        # autoscale behavior
        if autoscale_enabled:
            ax.relim()
            ax.autoscale_view()
            locked_xlim = ax.get_xlim()
            locked_ylim = ax.get_ylim()
        else:
            if locked_xlim is not None and locked_ylim is not None:
                ax.set_xlim(locked_xlim)
                ax.set_ylim(locked_ylim)

        fig.canvas.draw()
        fig.canvas.flush_events()

        # auto-save RAW only
        if auto_save_enabled:
            now = time.time()
            if now - last_save_time >= SAVE_INTERVAL_SECONDS:
                t = ts_now()
                auto_path = os.path.join(paths["raw"], f"auto_raw_{t}.csv")
                write_csv_with_metadata(
                    auto_path, wl, raw,
                    {**meta_common("autosave"), "type": "raw", "autosave_interval_s": str(SAVE_INTERVAL_SECONDS)}
                )
                last_save_time = now

        plt.pause(0.1)

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()
