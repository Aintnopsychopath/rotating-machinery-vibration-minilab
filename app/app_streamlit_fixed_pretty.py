# app_streamlit.py
# Phyphox Hair Dryer — Interactive Vibration + Acoustics Dashboard (Streamlit)
#
# Run:
#   pip install streamlit numpy pandas scipy matplotlib
#   streamlit run app_streamlit.py
#
# Optional (recommended for interactive plots):
#   pip install plotly

import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal as sp_signal
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Optional Plotly ----------
PLOTLY_OK = False
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------- Styling (prettier UI) ----------
st.set_page_config(page_title="Hair Dryer Vibration + Acoustics", layout="wide")

CUSTOM_CSS = """
<style>
/* overall */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.small { color: #6b7280; font-size: 0.95rem; }

/* cards */
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 16px 16px;
  background: rgba(255,255,255,0.65);
  backdrop-filter: blur(6px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.05);
  margin-bottom: 14px;
}
.card h3 { margin-top: 0.2rem; margin-bottom: 0.2rem; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(99,102,241,0.12);
  border: 1px solid rgba(99,102,241,0.28);
  color: #3730a3;
  font-size: 0.85rem;
}
hr { margin: 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Helpers ----------
def _find_col(cols, contains_any, contains_all=None):
    cols = list(cols)
    low = [c.lower() for c in cols]
    for c, cl in zip(cols, low):
        ok_any = any(s.lower() in cl for s in contains_any)
        ok_all = True if not contains_all else all(s.lower() in cl for s in contains_all)
        if ok_any and ok_all:
            return c
    return None

def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def load_vibration_df(df: pd.DataFrame):
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    if tcol is None:
        raise ValueError(f"Could not find a time column. Columns: {list(df.columns)}")

    acol = (_find_col(df.columns, ["absolute acceleration"], ["m/s"]) or
            _find_col(df.columns, ["absolute acceleration"]) or
            _find_col(df.columns, ["linear acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"], ["m/s"]) or
            _find_col(df.columns, ["acceleration"]))
    if acol is None:
        raise ValueError(f"Could not find an acceleration column. Columns: {list(df.columns)}")

    t = df[tcol].to_numpy(dtype=float)
    x = df[acol].to_numpy(dtype=float)
    return t, x

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
    return float(fb[i])

def spectrogram_db(y, fs):
    f, t, S = sp_signal.spectrogram(
        y, fs=fs,
        nperseg=min(1024, max(128, len(y)//6)),
        scaling="density", mode="psd"
    )
    SdB = 10*np.log10(S + 1e-20)
    return f, t, SdB

def analyze_vibration_file(uploaded_file, hp, lp, fmin, fmax):
    df = _read_csv(uploaded_file)
    t_raw, x_raw = load_vibration_df(df)
    t, x, fs = resample_uniform(t_raw, x_raw)
    y = preprocess_vibration(x, fs, hp=hp, lp=lp)
    f, pxx = welch_psd(y, fs)
    nyq = 0.5 * fs

    fpk = dominant_freq_from_psd(f, pxx, fmin=fmin, fmax=min(fmax, nyq-1e-6))
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / rms) if rms > 0 else float("nan")

    return {
        "name": uploaded_file.name,
        "fs": float(fs),
        "nyq": float(nyq),
        "duration_s": float(t[-1]-t[0]),
        "dominant_hz": float(fpk),
        "rpm_obs": float(fpk*60),
        "rpm_mirror": float((fs - fpk)*60),
        "rms": rms,
        "crest": crest,
        "t": t, "y": y,
        "f_psd": f, "pxx": pxx,
    }

def summarize_runs(runs):
    dom = np.array([r["dominant_hz"] for r in runs], dtype=float)
    rms = np.array([r["rms"] for r in runs], dtype=float)
    return {
        "n_runs": len(runs),
        "dominant_hz_mean": float(dom.mean()) if len(dom) else np.nan,
        "dominant_hz_std": float(dom.std(ddof=1)) if len(dom) > 1 else 0.0,
        "repeatability_dom_hz_delta": float(dom.max()-dom.min()) if len(dom) > 1 else 0.0,
        "rms_mean": float(rms.mean()) if len(rms) else np.nan,
        "rms_std": float(rms.std(ddof=1)) if len(rms) > 1 else 0.0,
        "repeatability_rms_cv": float(rms.std(ddof=1)/rms.mean()) if len(rms) > 1 and rms.mean() > 0 else 0.0,
    }

# ---------- Acoustics ----------
def detect_audio_type(df: pd.DataFrame):
    tcol = _find_col(df.columns, ["time"], ["s"]) or _find_col(df.columns, ["time"])
    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    if tcol and fcol:
        return "freq_history", tcol, fcol, None

    fcol = _find_col(df.columns, ["frequency"], ["hz"]) or _find_col(df.columns, ["frequency"])
    acol = _find_col(df.columns, ["amplitude"]) or _find_col(df.columns, ["level"]) or _find_col(df.columns, ["power"])
    if fcol and acol:
        return "spectrum", None, fcol, acol

    raise ValueError(f"Could not detect audio CSV type. Columns: {list(df.columns)}")

def analyze_audio_file(uploaded_file):
    df = _read_csv(uploaded_file)
    mode, tcol, fcol, acol = detect_audio_type(df)

    if mode == "freq_history":
        t = df[tcol].to_numpy(dtype=float)
        f = df[fcol].to_numpy(dtype=float)
        f2 = f[np.isfinite(f)]
        return {
            "name": uploaded_file.name,
            "mode": "freq_history",
            "f_mean": float(np.mean(f2)) if len(f2) else np.nan,
            "f_std": float(np.std(f2, ddof=1)) if len(f2) > 1 else 0.0,
            "t_series": t,
            "f_series": f,
        }

    freq = df[fcol].to_numpy(dtype=float)
    amp = df[acol].to_numpy(dtype=float)
    i = int(np.argmax(amp))
    return {
        "name": uploaded_file.name,
        "mode": "spectrum",
        "tone_hz": float(freq[i]),
        "tone_amp": float(amp[i]),
        "freq": freq,
        "amp": amp,
    }

# ---------- Plotting ----------
def render_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)

def render_mpl(fig):
    st.pyplot(fig, clear_figure=True)

def plot_psd_overlay(runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in runs:
            fig.add_trace(go.Scatter(x=r["f_psd"], y=r["pxx"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="PSD", yaxis_type="log")
        render_plotly(fig)
    else:
        fig = plt.figure(figsize=(10,5))
        for r in runs:
            plt.semilogy(r["f_psd"], r["pxx"], label=r["name"], alpha=0.85)
        plt.xlim(0, 300)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        render_mpl(fig)

def plot_time_overlay(runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in runs:
            t = r["t"] - r["t"][0]
            fig.add_trace(go.Scatter(x=t, y=r["y"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Filtered vibration (a.u.)")
        render_plotly(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in runs:
            t = r["t"] - r["t"][0]
            plt.plot(t, r["y"], label=r["name"], alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Filtered vibration (a.u.)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        render_mpl(fig)

def plot_spectrogram(run, title):
    f, t, SdB = spectrogram_db(run["y"], run["fs"])
    m = f <= 300
    f = f[m]
    SdB = SdB[m, :]

    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(x=t, y=f, z=SdB, colorbar=dict(title="dB (rel.)")))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        render_plotly(fig)
    else:
        fig = plt.figure(figsize=(10,4.8))
        plt.pcolormesh(t, f, SdB, shading="auto")
        plt.ylim(0, 300)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="dB (relative)")
        render_mpl(fig)

def plot_freq_history_overlay(audio_runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in audio_runs:
            fig.add_trace(go.Scatter(x=r["t_series"], y=r["f_series"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        render_plotly(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in audio_runs:
            plt.plot(r["t_series"], r["f_series"], alpha=0.8, label=r["name"])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        render_mpl(fig)

def plot_spectrum_overlay(audio_runs, title):
    if PLOTLY_OK:
        fig = go.Figure()
        for r in audio_runs:
            fig.add_trace(go.Scatter(x=r["freq"], y=r["amp"], mode="lines", name=r["name"]))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        render_plotly(fig)
    else:
        fig = plt.figure(figsize=(10,4))
        for r in audio_runs:
            plt.plot(r["freq"], r["amp"], alpha=0.8, label=r["name"])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        render_mpl(fig)

# ---------- Plain-English dynamic blocks ----------
def stakeholder_vibration_explainer(low_sum, high_sum, vib_low, vib_high):
    st.markdown("""
**What the plots mean (always true):**
- **Time plot**: how strong the shaking is over time.
- **PSD plot**: which frequencies carry most of the vibration energy (the machine’s “signature”).
- **Repeatability**: if two runs at the same speed look similar, the measurement is more trustworthy.
""")

    stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"
    hz_low = low_sum["dominant_hz_mean"]
    hz_high = high_sum["dominant_hz_mean"]
    rpm_low = hz_low * 60
    rpm_high = hz_high * 60

    rep_low = low_sum["repeatability_dom_hz_delta"]
    rep_high = high_sum["repeatability_dom_hz_delta"]

    def rep_label(delta_hz):
        if delta_hz < 2.0: return "good"
        if delta_hz < 5.0: return "okay"
        return "needs improvement"

    nyq = np.nanmin([
        np.nanmin([r["nyq"] for r in vib_low]) if vib_low else np.nan,
        np.nanmin([r["nyq"] for r in vib_high]) if vib_high else np.nan
    ])

    st.markdown(f"""
**Your results (based on your uploads):**
- **{stronger}** shows higher vibration strength (RMS).
- Dominant vibration frequency changed from **Low ≈ {hz_low:.1f} Hz** to **High ≈ {hz_high:.1f} Hz**
  (roughly **{rpm_low:,.0f} → {rpm_high:,.0f} RPM** if that peak corresponds to 1X).
- Repeatability looks **{rep_label(rep_low)}** at low speed (peak spread ≈ **{rep_low:.1f} Hz**) and **{rep_label(rep_high)}** at high speed (≈ **{rep_high:.1f} Hz**).

**Important phone limitation (why engineers use a tach in industry):**
- Phone sensors are limited to about **Nyquist ≈ {nyq:.0f} Hz**. Higher-speed components may appear as mirrored/aliased peaks.
""")

def stakeholder_acoustics_explainer():
    st.markdown("""
**What the plots mean (always true):**
- **Tone** is like the pitch you hear.
- If the tone changes between low and high speed, it usually means the fan/motor speed changed.
- Engineers compare these signatures across operating points to detect abnormal behaviour.
""")

# ---------- UI ----------
st.markdown("""<div class="card">
  <span class="badge">Portfolio demo</span>
  <h2 style="margin-top:8px;">Hair Dryer Condition Monitoring</h2>
  <div class="small">Vibration + acoustics signature comparison using phone sensors (Phyphox) and signal processing.</div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Mode")
    persona = st.radio("Who are you?", ["Stakeholder", "Engineer"], index=0)

    st.markdown("---")
    st.subheader("Upload data")
    vib_low_files = st.file_uploader("Vibration — Low speed (2 runs recommended)", type=["csv"], accept_multiple_files=True)
    vib_high_files = st.file_uploader("Vibration — High speed (2 runs recommended)", type=["csv"], accept_multiple_files=True)

    audio_low_files = st.file_uploader("Acoustics — Low speed (optional)", type=["csv"], accept_multiple_files=True)
    audio_high_files = st.file_uploader("Acoustics — High speed (optional)", type=["csv"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Vibration settings")
    hp = st.slider("High-pass filter (Hz)", 0.0, 30.0, 5.0, 0.5)
    lp = st.slider("Low-pass filter (Hz)", 30.0, 250.0, 200.0, 5.0)
    fmin = st.slider("Peak search min (Hz)", 0, 50, 10, 1)
    fmax = st.slider("Peak search max (Hz)", 50, 250, 200, 5)

    st.markdown("---")
    if PLOTLY_OK:
        st.caption("Plotly detected: charts are zoomable/hoverable.")
    else:
        st.caption("Tip: `pip install plotly` for better interactivity.")

tabs = st.tabs(["Overview", "Vibration", "Acoustics", "Combined"])

# ---------- Vibration analysis (computed once per run) ----------
vib_low, vib_high = [], []
vib_ready = False
vib_errors = []

if vib_low_files or vib_high_files:
    for uf in vib_low_files:
        try:
            vib_low.append(analyze_vibration_file(uf, hp, lp, fmin, fmax))
        except Exception as e:
            vib_errors.append((uf.name, str(e)))
    for uf in vib_high_files:
        try:
            vib_high.append(analyze_vibration_file(uf, hp, lp, fmin, fmax))
        except Exception as e:
            vib_errors.append((uf.name, str(e)))
    vib_ready = bool(vib_low and vib_high)

# ---------- Acoustics analysis ----------
audio_low, audio_high = [], []
audio_ready = False
audio_errors = []

if audio_low_files or audio_high_files:
    for uf in audio_low_files:
        try:
            audio_low.append(analyze_audio_file(uf))
        except Exception as e:
            audio_errors.append((uf.name, str(e)))
    for uf in audio_high_files:
        try:
            audio_high.append(analyze_audio_file(uf))
        except Exception as e:
            audio_errors.append((uf.name, str(e)))
    audio_ready = bool(audio_low and audio_high)

# ---------- Overview ----------
with tabs[0]:
    st.subheader("At a glance")
    st.caption("Upload your CSVs in the sidebar. This page shows the key headline results.")

    if vib_errors:
        st.warning("Some vibration files could not be parsed:")
        for n, e in vib_errors:
            st.write(f"- {n}: {e}")
    if audio_errors:
        st.warning("Some audio files could not be parsed:")
        for n, e in audio_errors:
            st.write(f"- {n}: {e}")

    if vib_ready:
        low_sum = summarize_runs(vib_low)
        high_sum = summarize_runs(vib_high)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Low RMS", f"{low_sum['rms_mean']:.2f}")
        c2.metric("High RMS", f"{high_sum['rms_mean']:.2f}")
        c3.metric("Dominant freq (Low)", f"{low_sum['dominant_hz_mean']:.1f} Hz")
        c4.metric("Dominant freq (High)", f"{high_sum['dominant_hz_mean']:.1f} Hz")

        stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"
        st.success(f"Headline: **{stronger}** vibrates more (based on RMS).")

        metrics_df = pd.DataFrame([
            {"Speed": "Low", **low_sum},
            {"Speed": "High", **high_sum},
        ])
        st.download_button(
            "Download vibration summary CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="vibration_summary.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload at least one low-speed and one high-speed vibration CSV to see the overview.")

    if audio_ready:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) == 1:
            mode = list(modes)[0]
            if mode == "freq_history":
                low_mean = float(np.mean([r["f_mean"] for r in audio_low]))
                high_mean = float(np.mean([r["f_mean"] for r in audio_high]))
                st.write(f"Acoustics (tone mean): **Low ≈ {low_mean:.1f} Hz**, **High ≈ {high_mean:.1f} Hz**")
            else:
                low_tone = float(np.mean([r['tone_hz'] for r in audio_low]))
                high_tone = float(np.mean([r['tone_hz'] for r in audio_high]))
                st.write(f"Acoustics (dominant tone): **Low ≈ {low_tone:.1f} Hz**, **High ≈ {high_tone:.1f} Hz**")

# ---------- Vibration tab ----------
with tabs[1]:
    st.subheader("Vibration")
    if not vib_ready:
        st.info("Upload low + high vibration CSVs in the sidebar.")
    else:
        low_sum = summarize_runs(vib_low)
        high_sum = summarize_runs(vib_high)

        if persona == "Stakeholder":
            stronger = "High speed" if high_sum["rms_mean"] > low_sum["rms_mean"] else "Low speed"
            st.success(f"**{stronger}** shows higher vibration strength (RMS) in these measurements.")
        else:
            st.info("Engineer view: repeatability, PSD overlays, and optional spectrograms.")

        # Summary table + download
        metrics_df = pd.DataFrame([
            {"Speed": "Low", **low_sum},
            {"Speed": "High", **high_sum},
        ])
        st.dataframe(metrics_df, use_container_width=True)
        st.download_button(
            "Download vibration summary CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="vibration_summary.csv",
            mime="text/csv"
        )

        c1, c2 = st.columns(2)
        with c1:
            plot_time_overlay(vib_low, "Low speed — filtered vibration (repeatability)")
        with c2:
            plot_time_overlay(vib_high, "High speed — filtered vibration (repeatability)")

        c3, c4 = st.columns(2)
        with c3:
            plot_psd_overlay(vib_low, "Low speed — PSD (frequency content per run)")
        with c4:
            plot_psd_overlay(vib_high, "High speed — PSD (frequency content per run)")

        if persona == "Engineer":
            st.subheader("Time–frequency view (spectrogram)")
            s1, s2 = st.columns(2)
            with s1:
                plot_spectrogram(vib_low[0], f"Low speed — spectrogram ({vib_low[0]['name']})")
            with s2:
                plot_spectrogram(vib_high[0], f"High speed — spectrogram ({vib_high[0]['name']})")

        if persona == "Stakeholder":
            with st.expander("What do these plots mean? (plain English)"):
                stakeholder_vibration_explainer(low_sum, high_sum, vib_low, vib_high)

# ---------- Acoustics tab ----------
with tabs[2]:
    st.subheader("Acoustics")
    if not audio_ready:
        st.info("Upload low + high acoustics CSVs in the sidebar (Frequency history OR Audio spectrum).")
    else:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) > 1:
            st.error("Mixed audio file types detected (Frequency history + Spectrum). Upload only one type at a time.")
        else:
            mode = list(modes)[0]

            if mode == "freq_history":
                low_mean = float(np.mean([r["f_mean"] for r in audio_low]))
                high_mean = float(np.mean([r["f_mean"] for r in audio_high]))
                aud_df = pd.DataFrame([
                    {"Speed": "Low", "Tone mean (Hz)": low_mean, "Runs": len(audio_low)},
                    {"Speed": "High", "Tone mean (Hz)": high_mean, "Runs": len(audio_high)},
                ])
                st.dataframe(aud_df, use_container_width=True)
                st.download_button("Download acoustics summary CSV", aud_df.to_csv(index=False).encode("utf-8"),
                                   file_name="audio_summary.csv", mime="text/csv")

                c1, c2 = st.columns(2)
                with c1:
                    plot_freq_history_overlay(audio_low, "Low speed — tone tracking (frequency history)")
                with c2:
                    plot_freq_history_overlay(audio_high, "High speed — tone tracking (frequency history)")

            else:
                low_tone = float(np.mean([r["tone_hz"] for r in audio_low]))
                high_tone = float(np.mean([r["tone_hz"] for r in audio_high]))
                aud_df = pd.DataFrame([
                    {"Speed": "Low", "Dominant tone (Hz)": low_tone, "Runs": len(audio_low)},
                    {"Speed": "High", "Dominant tone (Hz)": high_tone, "Runs": len(audio_high)},
                ])
                st.dataframe(aud_df, use_container_width=True)
                st.download_button("Download acoustics summary CSV", aud_df.to_csv(index=False).encode("utf-8"),
                                   file_name="audio_summary.csv", mime="text/csv")

                c1, c2 = st.columns(2)
                with c1:
                    plot_spectrum_overlay(audio_low, "Low speed — audio spectrum")
                with c2:
                    plot_spectrum_overlay(audio_high, "High speed — audio spectrum")

            if persona == "Stakeholder":
                with st.expander("What does this mean? (plain English)"):
                    stakeholder_acoustics_explainer()

# ---------- Combined tab ----------
with tabs[3]:
    st.subheader("Vibration + Acoustics (combined view)")
    if not (vib_ready and audio_ready):
        st.info("Upload both vibration (low+high) and acoustics (low+high) to see a combined summary.")
    else:
        modes = {r["mode"] for r in audio_low + audio_high}
        if len(modes) > 1:
            st.error("Mixed audio file types detected. Use only one (Frequency history OR Spectrum).")
        else:
            low_sum = summarize_runs(vib_low)
            high_sum = summarize_runs(vib_high)

            mode = list(modes)[0]
            if mode == "freq_history":
                low_a = float(np.mean([r["f_mean"] for r in audio_low]))
                high_a = float(np.mean([r["f_mean"] for r in audio_high]))
            else:
                low_a = float(np.mean([r["tone_hz"] for r in audio_low]))
                high_a = float(np.mean([r["tone_hz"] for r in audio_high]))

            combo = pd.DataFrame([
                {"Speed": "Low", "Vibration dominant (Hz)": low_sum["dominant_hz_mean"], "Audio tone (Hz)": low_a},
                {"Speed": "High", "Vibration dominant (Hz)": high_sum["dominant_hz_mean"], "Audio tone (Hz)": high_a},
            ])
            st.dataframe(combo, use_container_width=True)
            st.download_button("Download combined summary CSV", combo.to_csv(index=False).encode("utf-8"),
                               file_name="combined_summary.csv", mime="text/csv")

            if persona == "Stakeholder":
                st.success("If vibration and sound shift consistently between speeds, it supports that the change is operating-condition related (not random noise).")
            else:
                st.info("For true order tracking you need a tach/keyphasor; here we focus on comparative signatures + repeatability.")

st.markdown("---")
st.caption("Phone sensors are not calibrated industrial instruments; this dashboard demonstrates methodology: measurement → repeatability → signature comparison → clear reporting.")
