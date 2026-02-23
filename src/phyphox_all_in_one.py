# Phyphox Hair Dryer Vibration — All-in-one analysis + HTML report
# Usage:
#   1) Put this script in the same folder as your two CSV files
#   2) Edit FILE_A and FILE_B below if needed
#   3) Run: python phyphox_all_in_one.py
#
# Output: a folder named 'phyphox_full_report' containing plots + report.html + summary_metrics.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from datetime import datetime

FILE_A = "data/Raw Data Speed 1.csv"
FILE_B = "data/Raw Data Speed 2.csv"
OUT_DIR = "phyphox_full_report"

os.makedirs(OUT_DIR, exist_ok=True)

def find_col(columns, must_any, must_all=None):
    cols = list(columns)
    low = [c.lower() for c in cols]
    for c, cl in zip(cols, low):
        if any(s.lower() in cl for s in must_any) and (True if not must_all else all(s.lower() in cl for s in must_all)):
            return c
    return None

def load_phyphox_csv(path):
    df = pd.read_csv(path)
    tcol = find_col(df.columns, must_any=["time"], must_all=["s"]) or find_col(df.columns, must_any=["time"])
    if tcol is None:
        raise ValueError(f"Time column not found in {path}. Columns: {list(df.columns)}")

    acol = (find_col(df.columns, must_any=["absolute acceleration"], must_all=["m/s"]) or
            find_col(df.columns, must_any=["absolute acceleration"]) or
            find_col(df.columns, must_any=["linear acceleration"], must_all=["m/s"]) or
            find_col(df.columns, must_any=["acceleration"], must_all=["m/s"]) or
            find_col(df.columns, must_any=["acceleration"]))
    if acol is None:
        raise ValueError(f"Acceleration column not found in {path}. Columns: {list(df.columns)}")

    t = df[tcol].to_numpy(dtype=float)
    x = df[acol].to_numpy(dtype=float)
    return t, x, tcol, acol

def resample_uniform(t, x):
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    tu = np.arange(t[0], t[-1], dt)
    xu = np.interp(tu, t, x)
    return tu, xu, fs

def preprocess(x, fs, hp=5.0, lp=200.0):
    y = signal.detrend(x, type="linear")
    nyq = 0.5 * fs
    lo = hp / nyq
    hi = min(lp / nyq, 0.999)
    if hi <= lo:
        return y
    b, a = signal.butter(4, [lo, hi], btype="bandpass")
    return signal.filtfilt(b, a, y)

def fft_amp(y, fs):
    n = len(y)
    win = np.hanning(n)
    Y = np.fft.rfft(y * win)
    f = np.fft.rfftfreq(n, d=1/fs)
    amp = (2.0 / np.sum(win)) * np.abs(Y)
    return f, amp

def welch_psd(y, fs):
    nperseg = min(2048, max(256, len(y)//4))
    f, pxx = signal.welch(y, fs=fs, nperseg=nperseg, scaling="density")
    return f, pxx

def dominant_peak(f, s, fmin=10, fmax=None):
    if fmax is None:
        fmax = f.max()
    band = (f >= fmin) & (f <= fmax)
    fb, sb = f[band], s[band]
    i = int(np.argmax(sb))
    return float(fb[i]), float(sb[i])

def band_energy(f, pxx, f1, f2):
    m = (f >= f1) & (f <= f2)
    if np.sum(m) < 2:
        return float("nan")
    y = pxx[m]
    x = f[m]
    area_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(area_fn(y, x))

def rolling_rms(x, fs, win_s=1.0):
    n = max(5, int(win_s * fs))
    if n >= len(x):
        return np.array([np.sqrt(np.mean(x**2))]), np.array([0.0])
    sq = x**2
    kernel = np.ones(n) / n
    rms = np.sqrt(np.convolve(sq, kernel, mode="valid"))
    t_mid = (np.arange(len(rms)) + n/2) / fs
    return rms, t_mid

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def spectrogram_plot(y, fs, title, fname):
    f, t, Sxx = signal.spectrogram(y, fs=fs, nperseg=min(1024, len(y)//4), scaling="density", mode="psd")
    plt.figure(figsize=(11,5))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-20), shading="auto")
    plt.ylim(0, 300)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Power (dB, relative)")
    return savefig(fname)

def analyze(path, label):
    t_raw, x_raw, tcol, acol = load_phyphox_csv(path)
    t, x, fs = resample_uniform(t_raw, x_raw)
    y = preprocess(x, fs)
    f_fft, amp = fft_amp(y, fs)
    f_psd, pxx = welch_psd(y, fs)
    fpk, apk = dominant_peak(f_fft, amp, fmin=10, fmax=min(0.5*fs-1e-6, 300))

    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / rms) if rms > 0 else float("nan")

    return {
        "label": label,
        "file": os.path.basename(path),
        "fs": float(fs),
        "nyq": float(0.5*fs),
        "duration": float(t[-1]-t[0]),
        "t": t, "x": x, "y": y,
        "f_fft": f_fft, "amp": amp,
        "f_psd": f_psd, "pxx": pxx,
        "fpk": float(fpk),
        "rpm_obs": float(fpk*60),
        "rpm_mirror": float((fs - fpk)*60),
        "rms": rms,
        "crest": crest,
        "energy_10_200": band_energy(f_psd, pxx, 10, min(200, 0.5*fs))
    }

A = analyze(FILE_A, "Run A")
B = analyze(FILE_B, "Run B")

# Plots
plt.figure(figsize=(11,4)); plt.plot(A["t"]-A["t"][0], A["x"]); plt.title("Run A — Raw acceleration"); plt.xlabel("Time (s)"); plt.ylabel("m/s²"); savefig("01_runA_time_raw.png")
plt.figure(figsize=(11,4)); plt.plot(B["t"]-B["t"][0], B["x"]); plt.title("Run B — Raw acceleration"); plt.xlabel("Time (s)"); plt.ylabel("m/s²"); savefig("02_runB_time_raw.png")

plt.figure(figsize=(11,4)); plt.plot(A["t"]-A["t"][0], A["y"]); plt.title("Run A — Filtered vibration (bandpassed)"); plt.xlabel("Time (s)"); plt.ylabel("a.u."); savefig("03_runA_time_filtered.png")
plt.figure(figsize=(11,4)); plt.plot(B["t"]-B["t"][0], B["y"]); plt.title("Run B — Filtered vibration (bandpassed)"); plt.xlabel("Time (s)"); plt.ylabel("a.u."); savefig("04_runB_time_filtered.png")

rA, tA = rolling_rms(A["y"], A["fs"]); plt.figure(figsize=(11,4)); plt.plot(tA, rA); plt.title("Run A — Rolling RMS (vibration strength)"); plt.xlabel("Time (s)"); plt.ylabel("RMS"); savefig("05_runA_rolling_rms.png")
rB, tB = rolling_rms(B["y"], B["fs"]); plt.figure(figsize=(11,4)); plt.plot(tB, rB); plt.title("Run B — Rolling RMS (vibration strength)"); plt.xlabel("Time (s)"); plt.ylabel("RMS"); savefig("06_runB_rolling_rms.png")

plt.figure(figsize=(9,4)); plt.hist(A["y"], bins=80); plt.title("Run A — Histogram (distribution)"); plt.xlabel("Filtered accel"); plt.ylabel("Count"); savefig("07_runA_hist.png")
plt.figure(figsize=(9,4)); plt.hist(B["y"], bins=80); plt.title("Run B — Histogram (distribution)"); plt.xlabel("Filtered accel"); plt.ylabel("Count"); savefig("08_runB_hist.png")

plt.figure(figsize=(11,6))
plt.plot(A["f_fft"], A["amp"], label=f"Run A peak≈{A['fpk']:.1f} Hz")
plt.plot(B["f_fft"], B["amp"], label=f"Run B peak≈{B['fpk']:.1f} Hz")
plt.xlim(0, 300); plt.grid(True, alpha=0.3)
plt.title("FFT spectrum (frequency content)"); plt.xlabel("Hz"); plt.ylabel("Amplitude")
plt.legend(); plt.axvline(A["fpk"], linestyle="--", linewidth=1); plt.axvline(B["fpk"], linestyle="--", linewidth=1)
savefig("09_fft_overlay.png")

plt.figure(figsize=(11,6))
plt.semilogy(A["f_psd"], A["pxx"], label="Run A")
plt.semilogy(B["f_psd"], B["pxx"], label="Run B")
plt.xlim(0, 300); plt.grid(True, alpha=0.3)
plt.title("PSD (cleaner energy view)"); plt.xlabel("Hz"); plt.ylabel("PSD")
plt.legend(); savefig("10_psd_overlay.png")

spectrogram_plot(A["y"], A["fs"], "Run A — Spectrogram", "11_runA_spectrogram.png")
spectrogram_plot(B["y"], B["fs"], "Run B — Spectrogram", "12_runB_spectrogram.png")

# Summary CSV
summary = pd.DataFrame([
    {"Run": A["label"], "File": A["file"], "fs_Hz": A["fs"], "Nyquist_Hz": A["nyq"], "DominantPeak_Hz": A["fpk"], "RPM_observed": A["rpm_obs"], "RPM_ifAliased_fsMinusPeak": A["rpm_mirror"], "RMS": A["rms"], "CrestFactor": A["crest"], "Energy_10_200": A["energy_10_200"]},
    {"Run": B["label"], "File": B["file"], "fs_Hz": B["fs"], "Nyquist_Hz": B["nyq"], "DominantPeak_Hz": B["fpk"], "RPM_observed": B["rpm_obs"], "RPM_ifAliased_fsMinusPeak": B["rpm_mirror"], "RMS": B["rms"], "CrestFactor": B["crest"], "Energy_10_200": B["energy_10_200"]},
])
summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)

# Simple HTML report
now = datetime.now().strftime("%Y-%m-%d %H:%M")
html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Phyphox Vibration Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
img {{ max-width: 100%; border: 1px solid #eee; border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f6f6f6; }}
.small {{ color: #555; }}
</style></head>
<body>
<h1>Vibration report (phone accelerometer + Python)</h1>
<p class="small">Generated: {now}</p>
<div class="card">
<h2>Quick numbers</h2>
<table>
<tr><th>Run</th><th>fs</th><th>Nyquist</th><th>Dominant peak</th><th>RPM (observed)</th><th>RPM if aliased (fs-peak)</th><th>RMS</th></tr>
<tr><td>{A["label"]}</td><td>{A["fs"]:.1f} Hz</td><td>{A["nyq"]:.1f} Hz</td><td>{A["fpk"]:.2f} Hz</td><td>{A["rpm_obs"]:.0f}</td><td>{A["rpm_mirror"]:.0f}</td><td>{A["rms"]:.2f}</td></tr>
<tr><td>{B["label"]}</td><td>{B["fs"]:.1f} Hz</td><td>{B["nyq"]:.1f} Hz</td><td>{B["fpk"]:.2f} Hz</td><td>{B["rpm_obs"]:.0f}</td><td>{B["rpm_mirror"]:.0f}</td><td>{B["rms"]:.2f}</td></tr>
</table>
<p class="small">If the true rotation frequency is higher than Nyquist (~fs/2), the phone may show an aliased (mirrored) peak. Industrial setups use a tach/keyphasor to avoid this.</p>
</div>
<div class="card"><h2>Plots</h2>
<p><b>Raw time:</b></p>
<img src="01_runA_time_raw.png"><img src="02_runB_time_raw.png">
<p><b>Filtered time:</b></p>
<img src="03_runA_time_filtered.png"><img src="04_runB_time_filtered.png">
<p><b>Vibration strength (rolling RMS):</b></p>
<img src="05_runA_rolling_rms.png"><img src="06_runB_rolling_rms.png">
<p><b>Frequency:</b></p>
<img src="09_fft_overlay.png"><img src="10_psd_overlay.png">
<p><b>Spectrogram:</b></p>
<img src="11_runA_spectrogram.png"><img src="12_runB_spectrogram.png">
</div>
</body></html>"""
with open(os.path.join(OUT_DIR, "report.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("Done! Open phyphox_full_report/report.html")
