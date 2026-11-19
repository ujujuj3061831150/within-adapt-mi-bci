"""Pure NumPy/SciPy/MNE causal DL trial loaders (zero PyTorch import in worker pool)."""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt, sosfilt_zi

import bcic2a_causal_coral_repro as b2a
import chang2025_causal_coral_repro as ch25
import eegbci_reviewer_ablations as eeg_ab

EA_REG_EPS = 1e-6
EIG_EPS = 1e-12


def causal_sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    zi_base = sosfilt_zi(sos)
    zi = zi_base[:, None, :] * x[:, :1][None, :, :]
    out, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return out


def sym_inv_sqrtm(mat: np.ndarray) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, EIG_EPS)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def compute_ea_subject_blocks(epochs: np.ndarray, n_chans: int, n_times: int) -> dict:
    mu = epochs.mean(axis=-1, keepdims=True)
    sd = epochs.std(axis=-1, keepdims=True) + 1e-6
    x_stat = ((epochs - mu) / sd)[:, None, :, :].astype(np.float32)

    xc = epochs - mu
    covs = np.einsum("nct,ndt->ncd", xc, xc) / float(n_times) + EA_REG_EPS * np.eye(n_chans)[None, :, :]
    r_sub = covs.mean(axis=0)
    r_inv = sym_inv_sqrtm(r_sub)
    x_tr_ea = np.einsum("ij,njt->nit", r_inv, epochs)[:, None, :, :].astype(np.float32)
    return {
        "x_stat": x_stat,
        "x_tr_ea": x_tr_ea,
        "raw": epochs.astype(np.float32),
        "covs": covs.astype(np.float64),
        "r_sub": r_sub.astype(np.float64),
    }


def dl_worker_eegbci(sid: int) -> dict:
    mne.set_log_level("WARNING")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sub_str = f"S{sid:03d}"
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=160.0, output="sos")
    s0, s1 = int(round(0.5 * 160.0)), int(round(0.5 * 160.0)) + 480
    ep_list, y_list = [], []
    for run in eeg_ab.MI_RUNS:
        raw = mne.io.read_raw_edf(eeg_ab.DATA_DIR / sub_str / f"{sub_str}R{run:02d}.edf", preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw)
        if abs(raw.info["sfreq"] - 160.0) > 1e-3:
            raw.resample(160.0, verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.pick([ch for ch in eeg_ab.CORE_CHANNELS if ch in raw.ch_names])
        events, ev_all = mne.events_from_annotations(raw, verbose=False)
        mi_id = {k: v for k, v in ev_all.items() if k in ("T1", "T2")}
        data_uv = raw.get_data() * 1e6
        vev = events[np.isin(events[:, 2], [mi_id["T1"], mi_id["T2"]])]
        vev = vev[(vev[:, 0] + s0 >= 0) & (vev[:, 0] + s1 <= data_uv.shape[1])]
        f_ca = causal_sosfilt(sos, data_uv)
        ep_list.append(np.stack([f_ca[:, ev[0] + s0 : ev[0] + s1] for ev in vev], axis=0))
        y_list.append((vev[:, 2] != mi_id["T1"]).astype(np.int64))
    epochs = np.concatenate(ep_list, 0)
    blk = compute_ea_subject_blocks(epochs, 13, 480)
    blk["y"] = np.concatenate(y_list, 0)
    return blk


def dl_worker_b2a(sid: int) -> dict:
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=250.0, output="sos")
    s0 = int(round((b2a.CUE_OFFSET_SEC + 0.5) * 250.0))
    s1 = s0 + 750
    out = {}
    for sn in ("T", "E"):
        mat = sio.loadmat(b2a.DATA_DIR / f"A{sid:02d}{sn}.mat", simplify_cells=True)
        ep_list, y_list = [], []
        for rd in mat["data"]:
            tr = np.asarray(rd["trial"], dtype=np.int64)
            if tr.size == 0:
                continue
            X = np.asarray(rd["X"], dtype=np.float64)[:, :22]
            X_car = (X - X.mean(1, keepdims=True)).T
            f_ca = causal_sosfilt(sos, X_car)
            ep_list.append(np.stack([f_ca[:, (i - 1) + s0 : (i - 1) + s1] for i in tr], 0))
            y_list.append(np.asarray(rd["y"], dtype=np.int64) - 1)
        ep = np.concatenate(ep_list, 0)
        blk = compute_ea_subject_blocks(ep, 22, 750)
        blk["y"] = np.concatenate(y_list, 0)
        out[sn] = blk
    return out


def dl_worker_chang(subj_dir: Path) -> dict:
    sid = subj_dir.name
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=1000.0, output="sos")
    s0, s1 = int(round(0.5 * 250.0)), int(round(0.5 * 250.0)) + 750
    ep_list, y_list = [], []
    for sp in sorted(subj_dir.glob(f"{sid}_MI[0-9]*.set")):
        eeg = sio.loadmat(sp, simplify_cells=True)["EEG"]
        n_ch, n_pts = int(eeg["nbchan"]), int(eeg["pnts"])
        ch_names = [c["labels"] for c in eeg["chanlocs"]]
        pick_idx = [ch_names.index(c) for c in ch25.CORE12_CHANNELS]
        raw = np.fromfile(sp.with_suffix(".fdt"), dtype=np.float32).reshape((n_ch, n_pts), order="F").astype(np.float64)
        raw = raw - raw.mean(0, keepdims=True)
        f250 = causal_sosfilt(sos, raw[pick_idx])[:, ::4]
        evs = eeg["event"]
        if isinstance(evs, dict):
            evs = [evs]
        lats, ys = [], []
        for ev in evs:
            tv = int(ev["type"])
            if tv in (1, 2, 3):
                s250 = int(round((float(ev["latency"]) - 1.0) / 4.0))
                if s250 + s1 <= f250.shape[1]:
                    lats.append(s250)
                    ys.append(tv - 1)
        ep_list.append(np.stack([f250[:, s + s0 : s + s1] for s in lats], 0))
        y_list.append(np.asarray(ys, dtype=np.int64))
    ep = np.concatenate(ep_list, 0)
    blk = compute_ea_subject_blocks(ep, 12, 750)
    blk["y"] = np.concatenate(y_list, 0)
    return blk
