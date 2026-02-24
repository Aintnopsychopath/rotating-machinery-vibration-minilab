import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from datetime import datetime

# ----------------------------
# Robust column detection
# ----------------------------
def _find_col(cols, contains_any, contains_all=None):
    cols_l = [c.lower() for c in cols]
    for c, cl in zip(cols, cols_l):
        ok_any = any(s.lower() in cl for s in contains_any)
        ok_all = True if not contains_all else all(s.lower() in cl for s in contains_all)
        if ok_any and ok_all:
            return c
    return None

def load_phyphox_timeseries(csv_path):
    df = pd.read_csv(csv_path)

    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    if tcol is None:
        raise ValueError(f"[{csv_path}] Could not find a time column. Columns: {list(df.columns)}")

    # Prefer absolute acceleration; fallback to linear accel; fallback to accel
    acol = (_find_col(df.columns, ["absolute acceleration"], ["m/s"]) or
            _find_col(df.columns, ["absolute acceleration"]) or
            _find_col(df.columns, ["linear acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"]))

    if acol is None:
        raise ValueError(f"[{csv_path}] Could not find an acceleration column. Columns: {list(df.columns)}")

    t = df[tcol].to_numpy(dtype=float)
    x = df[acol].to_numpy(dtype=float)
    return t, x, tcol, acol

def resample_uniform(t, x):
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], dt)
    xu = np.interp(tu, t, x)
    return tu, xu, fs

def preprocess_vibration(x, fs, hp=5.0, lp=200.0):
    y = sp_signal.detrend(x, type="linear")
    nyq = 0.5 * fs
    lo = hp / nyq
    hi = min(lp / nyq, 0.999)
    if hi <= lo:
        return y
    b, a = sp_signal.butter(4, [lo, hi], btype="bandpass")
    return sp_signal.filtfilt(b, a, y)

def welch_psd(y, fs):
    nperseg = min(2048, max(256, len(y)//4))
    f, pxx = sp_signal.welch(y, fs=fs, nperseg=nperseg, scaling="density")
    return f, pxx

def dominant_freq_from_psd(f, pxx, fmin=10, fmax=200):
    band = (f >= fmin) & (f <= fmax)
    fb, sb = f[band], pxx[band]
    i = int(np.argmax(sb))
    return float(fb[i]), float(sb[i])

def rolling_rms(y, fs, win_s=1.0):
    n = max(5, int(win_s * fs))
    if n >= len(y):
        return np.array([np.sqrt(np.mean(y**2))]), np.array([0.0])
    kernel = np.ones(n) / n
    rms = np.sqrt(np.convolve(y**2, kernel, mode="valid"))
    t_mid = (np.arange(len(rms)) + n/2) / fs
    return rms, t_mid

# ----------------------------
# Vibration analysis per run
# ----------------------------
def analyze_vibration_run(csv_path, hp=5.0, lp=200.0, fmin=10, fmax=200):
    t_raw, x_raw, tcol, acol = load_phyphox_timeseries(csv_path)
    t, x, fs = resample_uniform(t_raw, x_raw)
    y = preprocess_vibration(x, fs, hp=hp, lp=lp)

    f, pxx = welch_psd(y, fs)
    fpk, ppk = dominant_freq_from_psd(f, pxx, fmin=fmin, fmax=min(fmax, 0.5*fs-1e-6))
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / rms) if rms > 0 else float("nan")

    return {
        "file": str(csv_path),
        "fs": float(fs),
        "duration_s": float(t[-1]-t[0]),
        "dominant_hz": fpk,
        "rpm_obs": float(fpk*60),
        "rpm_mirror": float((fs - fpk)*60),  # alias candidate
        "rms": rms,
        "crest": crest,
        "t": t, "x_raw": x, "y": y,
        "f_psd": f, "pxx": pxx
    }

def summarize_speed(runs):
    # runs = list of dicts
    dom = np.array([r["dominant_hz"] for r in runs])
    rms = np.array([r["rms"] for r in runs])
    return {
        "dominant_hz_mean": float(dom.mean()),
        "dominant_hz_std": float(dom.std(ddof=1)) if len(dom) > 1 else 0.0,
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std(ddof=1)) if len(rms) > 1 else 0.0,
        "repeatability_dom_hz_delta": float(dom.max() - dom.min()) if len(dom) > 1 else 0.0,
        "repeatability_rms_cv": float(rms.std(ddof=1)/rms.mean()) if len(rms) > 1 and rms.mean() > 0 else 0.0,
    }

# ----------------------------
# Acoustics (two formats)
# ----------------------------
def analyze_freq_history(csv_path):
    df = pd.read_csv(csv_path)
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    if tcol is None or fcol is None:
        raise ValueError(f"[{csv_path}] Expected Frequency history with Time and Frequency columns. Columns: {list(df.columns)}")
    t = df[tcol].to_numpy(dtype=float)
    f = df[fcol].to_numpy(dtype=float)
    f = f[np.isfinite(f)]
    return {"file": str(csv_path), "f_mean": float(np.mean(f)), "f_std": float(np.std(f, ddof=1)) if len(f) > 1 else 0.0}

def analyze_audio_spectrum(csv_path):
    df = pd.read_csv(csv_path)
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    acol = _find_col(df.columns, ["amplitude"]) or _find_col(df.columns, ["level"]) or _find_col(df.columns, ["power"])
    if fcol is None or acol is None:
        raise ValueError(f"[{csv_path}] Expected Audio spectrum with Frequency + Amplitude-like column. Columns: {list(df.columns)}")
    f = df[fcol].to_numpy(dtype=float)
    a = df[acol].to_numpy(dtype=float)
    i = int(np.argmax(a))
    return {"file": str(csv_path), "tone_hz": float(f[i]), "tone_amp": float(a[i])}

# ----------------------------
# Plot helpers
# ----------------------------
def savefig(out_dir, name):
    out = Path(out_dir) / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def plot_psd_overlay(speed_label, runs, out_dir):
    plt.figure(figsize=(11,6))
    for r in runs:
        plt.semilogy(r["f_psd"], r["pxx"], alpha=0.8, label=Path(r["file"]).name)
    plt.xlim(0, 300)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (a.u.^2/Hz)")
    plt.title(f"{speed_label} — PSD per run (repeatability view)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    return savefig(out_dir, f"vib_psd_{speed_label.lower()}.png")

def plot_time_overlay(speed_label, runs, out_dir):
    plt.figure(figsize=(11,5))
    for r in runs:
        t = r["t"] - r["t"][0]
        plt.plot(t, r["y"], alpha=0.6, label=Path(r["file"]).name)
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered vibration (a.u.)")
    plt.title(f"{speed_label} — Filtered vibration over time (repeatability)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    return savefig(out_dir, f"vib_time_{speed_label.lower()}.png")

def plot_freq_history_overlay(speed_label, analyses, out_dir):
    # analyses are summaries; for visuals we load again quickly
    plt.figure(figsize=(11,5))
    for a in analyses:
        df = pd.read_csv(a["file"])
        tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
        fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
        plt.plot(df[tcol], df[fcol], alpha=0.7, label=Path(a["file"]).name)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{speed_label} — Acoustic frequency history (tone tracking)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    return savefig(out_dir, f"audio_freq_history_{speed_label.lower()}.png")

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- vibration ----
    vib_results = {}
    vib_summary_rows = []

    if args.do_vibration:
        for speed in ["low", "high"]:
            files = [Path(f) for f in (args.vib_low if speed=="low" else args.vib_high)]
            runs = [analyze_vibration_run(f, hp=args.hp, lp=args.lp, fmin=args.fmin, fmax=args.fmax) for f in files]
            vib_results[speed] = runs

            s = summarize_speed(runs)
            vib_summary_rows.append({
                "speed": speed,
                "n_runs": len(runs),
                **s
            })

            plot_time_overlay(speed, runs, out_dir)
            plot_psd_overlay(speed, runs, out_dir)

        pd.DataFrame(vib_summary_rows).to_csv(out_dir/"vibration_summary.csv", index=False)

    # ---- acoustics ----
    audio_rows = []
    audio_plots = []
    if args.do_acoustics:
        if args.audio_mode == "freq_history":
            for speed in ["low", "high"]:
                files = [Path(f) for f in (args.audio_low if speed=="low" else args.audio_high)]
                analyses = [analyze_freq_history(f) for f in files]
                # summary
                fmean = np.mean([a["f_mean"] for a in analyses]) if analyses else np.nan
                fstd  = np.mean([a["f_std"] for a in analyses]) if analyses else np.nan
                audio_rows.append({"speed": speed, "mode": "freq_history", "tone_hz_mean": fmean, "tone_hz_std": fstd, "n_runs": len(analyses)})
                audio_plots.append(plot_freq_history_overlay(speed, analyses, out_dir))
        else:  # spectrum
            for speed in ["low", "high"]:
                files = [Path(f) for f in (args.audio_low if speed=="low" else args.audio_high)]
                analyses = [analyze_audio_spectrum(f) for f in files]
                tone = np.mean([a["tone_hz"] for a in analyses]) if analyses else np.nan
                audio_rows.append({"speed": speed, "mode": "spectrum", "tone_hz_mean": tone, "n_runs": len(analyses)})

        pd.DataFrame(audio_rows).to_csv(out_dir/"audio_summary.csv", index=False)

    # ---- combined “executive” summary HTML ----
    html = []
    html.append(f"<h1>Phyphox Condition Monitoring — Hair Dryer</h1>")
    html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")
    html.append("<h2>What this shows</h2>")
    html.append("<ul>"
                "<li>Repeatability: two runs per speed (low/high)</li>"
                "<li>Vibration signature: filtered signal + PSD</li>"
                "<li>Acoustic signature: tone tracking (frequency history) or dominant tone (spectrum)</li>"
                "</ul>")

    if args.do_vibration:
        html.append("<h2>Vibration summary</h2>")
        html.append(pd.DataFrame(vib_summary_rows).to_html(index=False))
        html.append("<p><b>Note:</b> Phone sampling limits (Nyquist) can cause aliasing. In industrial turbomachinery tests, a tach/keyphasor removes ambiguity.</p>")
        for name in ["vib_time_low.png","vib_psd_low.png","vib_time_high.png","vib_psd_high.png"]:
            p = out_dir/name
            if p.exists():
                html.append(f"<img src='{p.name}' style='max-width:100%;border:1px solid #eee;border-radius:8px;margin:8px 0;'/>")

    if args.do_acoustics:
        html.append("<h2>Acoustics summary</h2>")
        html.append(pd.DataFrame(audio_rows).to_html(index=False))
        for p in audio_plots:
            html.append(f"<img src='{Path(p).name}' style='max-width:100%;border:1px solid #eee;border-radius:8px;margin:8px 0;'/>")

    (out_dir/"report.html").write_text("\n".join(html), encoding="utf-8")

    print(f"\n✅ Done. Outputs written to: {out_dir}")
    print(f"Open: {out_dir/'report.html'}")

def build_parser():
    p = argparse.ArgumentParser(description="Phyphox hair-dryer condition monitoring (vibration + acoustics) with repeatability.")
    p.add_argument("--out", default="analysis_results", help="Output directory")

    # vibration
    p.add_argument("--vibration", dest="do_vibration", action="store_true", help="Run vibration analysis")
    p.add_argument("--vib-low", nargs="+", default=[], help="Low-speed vibration CSVs (2 runs recommended)")
    p.add_argument("--vib-high", nargs="+", default=[], help="High-speed vibration CSVs (2 runs recommended)")
    p.add_argument("--hp", type=float, default=5.0, help="High-pass filter cutoff (Hz)")
    p.add_argument("--lp", type=float, default=200.0, help="Low-pass filter cutoff (Hz)")
    p.add_argument("--fmin", type=float, default=10.0, help="Min freq for dominant peak search (Hz)")
    p.add_argument("--fmax", type=float, default=200.0, help="Max freq for dominant peak search (Hz)")

    # acoustics
    p.add_argument("--acoustics", dest="do_acoustics", action="store_true", help="Run acoustics analysis")
    p.add_argument("--audio-mode", choices=["freq_history","spectrum"], default="freq_history", help="Acoustics dataset type")
    p.add_argument("--audio-low", nargs="+", default=[], help="Low-speed audio CSVs (2 runs recommended)")
    p.add_argument("--audio-high", nargs="+", default=[], help="High-speed audio CSVs (2 runs recommended)")

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.do_vibration and not args.do_acoustics:
        print("Tip: run with --vibration and/or --acoustics. Example commands are shown below.")
        parser.print_help()
    else:
        run_pipeline(args)