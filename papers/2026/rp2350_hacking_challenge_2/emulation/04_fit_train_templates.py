#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step 4: Fit `canonical_t2_ref.npy` & Walsh/Tree-BP Templates Strictly on `train`.

Reads ONLY `<work_dir>/dataset/train/shards/*.fb` (`0` holdout shards!) and fits the two
profiling inputs expected by unmodified `end_to_end_attack.py`:
  1. `<work_dir>/profiling/canonical_t2_ref.npy` — mean of the first 512 `train` `trace2` waveforms.
  2. `<work_dir>/profiling/trace2_walsh_tree_bp_templates.pkl` — 4-mode Walsh + S-Box Tree-BP
     linear/quadratic operators fitted via 4-fold out-of-fold cross-validation on `train` keys
     using the operator primitives in `<scaaml_dir>/end_to_end_attack.py`.

Inputs:
  - `<work_dir>/dataset/train/shards/*.fb`
  - `<scaaml_dir>/end_to_end_attack.py`

Outputs:
  - `<work_dir>/profiling/canonical_t2_ref.npy`
  - `<work_dir>/profiling/trace2_walsh_tree_bp_templates.pkl`
  - `<work_dir>/profiling/profiling_summary.json`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sedpack.io.compress import CompressedFile
import sedpack.io.flatbuffer.shardfile.Shard as fbapi_Shard

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCAAML_DIR = ROOT_DIR.parent
FULL_ADC_SAMPLES = 97553


def _read_shard_labeled(sh_path: str) -> tuple[np.ndarray, ...]:
    """Read a labeled profiling shard (`train` / `test`) for template fitting."""
    comp = CompressedFile("LZ4")
    buf = comp.decompress(Path(sh_path).read_bytes())
    sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
    n = sh.ExamplesLength()
    remap_hdr_a = np.empty((n, 18), dtype=np.float32)
    remap_hdr_b = np.empty((n, 18), dtype=np.float32)
    remap_a = np.empty((n, 64, 30), dtype=np.float32)
    remap_b = np.empty((n, 64, 30), dtype=np.float32)
    pre_sbox = np.empty((n, 13, 56), dtype=np.float32)
    sbox_win = np.empty((n, 13, 240), dtype=np.float32)
    key0 = np.frombuffer(sh.Examples(0).Attributes(6).AttributeBytesAsNumpy(),
                         dtype=np.uint8).copy()
    for j in range(n):
        t2 = np.frombuffer(sh.Examples(j).Attributes(1).AttributeBytesAsNumpy(),
                           dtype=np.int16).astype(np.float32)
        t2_c = t2 - t2.mean()
        remap_hdr_a[j] = t2_c[14500:14518]
        remap_hdr_b[j] = t2_c[16461:16479]
        remap_a[j] = t2_c[14518:14518 + 64 * 30].reshape(64, 30)
        remap_b[j] = t2_c[16479:16479 + 64 * 30].reshape(64, 30)
        for m in range(13):
            pre_base = 21491 + 1440 * m
            pre_sbox[j, m] = t2_c[pre_base:pre_base + 56]
            base = 21643 + 1440 * m
            sbox_win[j, m] = t2_c[base:base + 240]
    return remap_hdr_a, remap_hdr_b, remap_a, remap_b, pre_sbox, sbox_win, key0


def _fit_d0_operator(tr_data: dict[str, Any], D_true_tr: np.ndarray,
                     tr_keys: np.ndarray, j: int) -> dict[str, Any]:
    n_act = 44 if j < 4 else 35
    act_u = np.arange(18, 18 + n_act)
    S_act = tr_data["S_m68"][:, :, :, act_u]
    G_feat = tr_data["G_feat_44"][:, :, :, :n_act, :n_act]
    H_ab, reg_h = _get_H_ab(tr_data, j)

    H_tr = H_ab[tr_keys].reshape(-1, H_ab.shape[2])
    mu_h = H_tr.mean(axis=0, keepdims=True)
    sd_h = H_tr.std(axis=0, keepdims=True) + 1e-6
    H_tr_n = (H_tr - mu_h) / sd_h

    P_j = np.einsum("km,ktmu->ktu", D_true_tr[tr_keys, :, 0, j],
                    S_act[tr_keys]).reshape(-1, n_act)
    r_scale = (len(tr_keys) * tr_data["M"]) / (128.0 * 256.0)
    K_h = H_tr_n.T @ H_tr_n + (reg_h * r_scale) * np.eye(H_tr_n.shape[1])
    W_j = np.linalg.solve(K_h, H_tr_n.T @ P_j)
    u_h, s_h, vt_h = np.linalg.svd(W_j, full_matrices=False)

    iu = np.triu_indices(n_act)
    m1_idx, m2_idx = np.triu_indices(13, k=1)
    r_key_scale = len(tr_keys) / 128.0

    X_cross_tr = G_feat[tr_keys][:, m1_idx,
                                 m2_idx][:, :, iu[0],
                                         iu[1]].reshape(-1, len(iu[0]))
    mu_c = X_cross_tr.mean(axis=0, keepdims=True)
    sd_c = X_cross_tr.std(axis=0, keepdims=True) + 1e-6
    X_cross_tr_n = (X_cross_tr - mu_c) / sd_c
    K_cross = X_cross_tr_n.T @ X_cross_tr_n + (4000.0 * r_key_scale) * np.eye(
        len(iu[0]))
    y_c_tr = (D_true_tr[tr_keys][:, m1_idx, 0, j] *
              D_true_tr[tr_keys][:, m2_idx, 0, j]).reshape(-1)
    y_c_mean = float(y_c_tr.mean())
    w_c = np.linalg.solve(K_cross, X_cross_tr_n.T @ (y_c_tr - y_c_mean))

    if j < 4:
        X_intra_tr = (np.diagonal(G_feat[tr_keys], axis1=1, axis2=2).transpose(
            0, 3, 1, 2)[:, :, iu[0], iu[1]].reshape(-1, len(iu[0])))
        mu_i = X_intra_tr.mean(axis=0, keepdims=True)
        sd_i = X_intra_tr.std(axis=0, keepdims=True) + 1e-6
        X_intra_tr_n = (X_intra_tr - mu_i) / sd_i
        K_intra = X_intra_tr_n.T @ X_intra_tr_n + (
            2000.0 * r_key_scale) * np.eye(len(iu[0]))
        y_i_tr = (D_true_tr[tr_keys, :, 0, j]**2).reshape(-1)
        y_i_mean = float(y_i_tr.mean())
        w_i = np.linalg.solve(K_intra, X_intra_tr_n.T @ (y_i_tr - y_i_mean))

        w_raw = w_c / sd_c[0]
        W_mat = np.zeros((n_act, n_act), dtype=np.float64)
        W_mat[iu[0], iu[1]] = 0.5 * w_raw
        W_mat[iu[1], iu[0]] = 0.5 * w_raw
        W_mat[np.diag_indices(n_act)] = w_raw[iu[0] == iu[1]]
        _, evecs_w = np.linalg.eigh(W_mat)
        u1 = evecs_w[:, -1].astype(np.float32)
    else:
        mu_i = sd_i = w_i = y_i_mean = None
        u1 = vt_h[0]

    return {
        "j": j,
        "n_act": n_act,
        "act_u": act_u,
        "mu_h": mu_h,
        "sd_h": sd_h,
        "u_h": u_h,
        "s_h": s_h,
        "vt_h": vt_h,
        "u1": u1,
        "iu": iu,
        "m1_idx": m1_idx,
        "m2_idx": m2_idx,
        "mu_c": mu_c,
        "sd_c": sd_c,
        "w_c": w_c,
        "y_c_mean": y_c_mean,
        "mu_i": mu_i,
        "sd_i": sd_i,
        "w_i": w_i,
        "y_i_mean": y_i_mean,
    }


def _get_H_ab(data: dict[str, Any], j: int) -> tuple[np.ndarray, float]:
    if j >= 2:
        c_q = 63 - np.arange(64)
        walsh_sign = (1.0 - 2.0 * ((c_q >> (j - 2)) & 1)).astype(np.float32)
        Ha = np.einsum("q,ktqu->ktu", walsh_sign, data["Za_ext"]) / 64.0
        Hb = np.einsum("q,ktqu->ktu", walsh_sign, data["Zb_ext"]) / 64.0
        H_ab = np.concatenate([Ha, Hb, data["Z_pro"], data["Z_r10_loop"]],
                              axis=2)
        H_ab = H_ab - H_ab.mean(axis=1, keepdims=True)
        reg_h = 5000.0
    else:
        H_ab = data["M_base"]
        reg_h = 12000.0
    return H_ab, reg_h


def _apply_d0_operator(
    data: dict[str, Any],
    op: dict[str,
             Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j = op["j"]
    n_act = op["n_act"]
    act_u = op["act_u"]
    N_K, M = data["N_K"], data["M"]
    S_act = data["S_m68"][:, :, :, act_u]
    G_feat = data["G_feat_44"][:, :, :, :n_act, :n_act]
    H_ab, _ = _get_H_ab(data, j)

    H_all_n = ((H_ab.reshape(-1, H_ab.shape[2]) - op["mu_h"]) /
               op["sd_h"]).reshape(N_K, M, -1)
    d0_dir = np.zeros((N_K, 13), dtype=np.float32)
    for r_idx in range(2):
        h_r = H_all_n @ op["u_h"][:, r_idx]
        s_r = S_act @ op["vt_h"][r_idx]
        d0_dir += op["s_h"][r_idx] * np.einsum("kt,ktm->km", h_r,
                                               s_r) / float(M)

    iu = op["iu"]
    m1_idx, m2_idx = op["m1_idx"], op["m2_idx"]
    G_all = np.zeros((N_K, 13, 13), dtype=np.float64)
    for k in range(N_K):
        feat_c = (G_feat[k, m1_idx, m2_idx][:, iu[0], iu[1]] -
                  op["mu_c"]) / op["sd_c"]
        vals_c = (feat_c @ op["w_c"]) + op["y_c_mean"]
        G_all[k, m1_idx, m2_idx] = vals_c
        G_all[k, m2_idx, m1_idx] = vals_c

    if j < 4:
        d0_eig = np.zeros((N_K, 13), dtype=np.float32)
        for k in range(N_K):
            G_k = G_all[k].copy()
            feat_i = np.array([
                (G_feat[k, m, m][iu[0], iu[1]] - op["mu_i"][0]) / op["sd_i"][0]
                for m in range(13)
            ])
            vals_i = np.maximum(0.0, (feat_i @ op["w_i"]) + op["y_i_mean"])
            np.fill_diagonal(G_k, vals_i)
            ev, evec = np.linalg.eigh(G_k)
            d0_eig[k] = evec[:, -1] * np.sqrt(max(0.0, ev[-1]))
            if np.dot(d0_eig[k], d0_dir[k]) < 0:
                d0_eig[k] = -d0_eig[k]
    else:
        d0_eig = d0_dir.copy()

    return d0_dir, d0_eig, G_all, H_all_n


def _init_v2_worker(scaaml_dir_str: str) -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if scaaml_dir_str not in sys.path:
        sys.path.insert(0, scaaml_dir_str)


def _compute_v_and_W_full(
    data: dict[str, Any],
    d0_ops: dict[int, dict[str, Any]],
    D0_est: np.ndarray,
    s_j_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N_K, M = data["N_K"], data["M"]
    v_iter_unmasked = np.zeros((N_K, M, 13, 4, 8), dtype=np.float32)
    for j in range(8):
        op = d0_ops[j]
        act_u = op["act_u"]
        S_act = data["S_m68"][:, :, :, act_u]
        Z_tilde_act = data["Z_tilde68"][:, :, :, :, act_u]
        H_ab, _ = _get_H_ab(data, j)
        H_all_n = ((H_ab.reshape(-1, H_ab.shape[2]) - op["mu_h"]) /
                   op["sd_h"]).reshape(N_K, M, -1)

        D0_j = D0_est[:, :, j]
        v_rem = np.zeros((N_K, M, 13, 4), dtype=np.float32)
        v_loo = np.zeros((N_K, M, 13, 4), dtype=np.float32)
        for r_idx in range(2):
            h_r = H_all_n @ op["u_h"][:, r_idx]
            s_r = S_act @ op["vt_h"][r_idx]
            z_r = Z_tilde_act @ op["vt_h"][r_idx]
            v_rem += op["s_h"][r_idx] * h_r[:, :, None, None] * z_r
            tot_m = np.einsum("km,ktm->kt", D0_j, s_r)[:, :, None]
            tot_m /= np.std(tot_m, axis=1, keepdims=True) + 1e-6
            v_loo += op["s_h"][r_idx] * tot_m[:, :, :, None] * z_r

        if j == 0:
            s_proj1 = S_act @ op["u1"]
            tot_m1 = np.einsum("km,ktm->kt", D0_j, s_proj1)[:, :, None]
            tot_m1 /= np.std(tot_m1, axis=1, keepdims=True) + 1e-6
            v_loo_u1 = tot_m1[:, :, :, None] * (Z_tilde_act @ op["u1"])
            v_loo_u1 /= np.std(v_loo_u1, axis=(1, 3), keepdims=True) + 1e-6
            v_rem /= np.std(v_rem, axis=(1, 3), keepdims=True) + 1e-6
            v_iter_unmasked[:, :, :, :, j] = 0.50 * v_rem + 0.50 * v_loo_u1
        else:
            v_rem /= np.std(v_rem, axis=(1, 3), keepdims=True) + 1e-6
            v_loo /= np.std(v_loo, axis=(1, 3), keepdims=True) + 1e-6
            v_iter_unmasked[:, :, :, :, j] = 0.65 * v_rem + 0.35 * v_loo

    W_iter_feat = np.einsum("ktmip,ktmij->kmpj", data["Z_tilde68"],
                            v_iter_unmasked) / float(M)
    W_pre_feat = np.einsum("ktmp,ktmij->kmipj",
                           data["Z_pre"], v_iter_unmasked).reshape(
                               N_K, 13, 4 * 56, 8) / float(M)
    W_all_feat = np.concatenate([W_iter_feat, W_pre_feat], axis=2)

    perm_u = np.array(
        list(range(10, 27)) + [35, 36, 46, 47, 48, 49, 50, 51, 52])
    Z_p_sub = data["Z_tilde68"][:, :, :, :, perm_u]
    iu_p = np.triu_indices(len(perm_u))
    Z_p_quad = Z_p_sub[:, :, :, :, iu_p[0]] * Z_p_sub[:, :, :, :, iu_p[1]]
    Z_p_quad -= Z_p_quad.mean(axis=3, keepdims=True)
    W_quad_feat = np.einsum("ktmip,ktmij->kmpj", Z_p_quad,
                            v_iter_unmasked) / float(M)

    W_full = np.concatenate([W_all_feat, W_quad_feat], axis=2)
    if s_j_vec is None:
        s_j_vec = np.array([W_full[:, :, :, j].std() + 1e-6 for j in range(8)],
                           dtype=np.float32)
    W_full_norm = W_full / s_j_vec[None, None, None, :]
    return v_iter_unmasked, W_full_norm, Z_p_quad, s_j_vec


def build_key_labels(
        keys_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ground-truth master key bytes K_true (N, 8, 4), S-box words W_true (N, 13, 4),
    and 4-mode Walsh bit parities D_true (N, 13, 4, 8).
    """
    import end_to_end_attack as v2
    n_keys = len(keys_bytes)
    rks_all = np.zeros((n_keys, 60), dtype=np.uint32)
    for k in range(n_keys):
        rks_all[k] = v2.expand_aes256_rk_words(keys_bytes[k])

    K_true = np.zeros((n_keys, 8, 4), dtype=np.uint8)
    for w_idx in range(8):
        w_val = rks_all[:, w_idx]
        for q in range(4):
            K_true[:, w_idx, q] = ((w_val >> (8 * q)) & 0xFF).astype(np.uint8)

    W_true = np.zeros((n_keys, 13, 4), dtype=np.uint8)
    X_bytes = np.zeros((n_keys, 13, 4), dtype=np.uint8)
    for m in range(13):
        w_orig = rks_all[:, 7 + 4 * m].copy()
        for q in range(4):
            W_true[:, m, q] = ((w_orig >> (8 * q)) & 0xFF).astype(np.uint8)
        w = ((w_orig >> 8) |
             (w_orig << 24)) & 0xFFFFFFFF if (m % 2 == 0) else w_orig
        for q in range(4):
            X_bytes[:, m, q] = ((w >> (8 * q)) & 0xFF).astype(np.uint8)

    B = np.zeros((n_keys, 13, 4, 8), dtype=np.float32)
    for q in range(4):
        for j in range(8):
            B[:, :, q,
              j] = 2.0 * ((X_bytes[:, :, q] >> j) & 1).astype(np.float32) - 1.0

    D_true = np.zeros((n_keys, 13, 4, 8), dtype=np.float32)
    D_true[:, :, 0, :] = 0.5 * (B[:, :, 0, :] + B[:, :, 1, :] + B[:, :, 2, :] +
                                B[:, :, 3, :])
    D_true[:, :, 1, :] = 0.5 * (B[:, :, 0, :] - B[:, :, 1, :] + B[:, :, 2, :] -
                                B[:, :, 3, :])
    D_true[:, :, 2, :] = 0.5 * (B[:, :, 0, :] + B[:, :, 1, :] - B[:, :, 2, :] -
                                B[:, :, 3, :])
    D_true[:, :, 3, :] = 0.5 * (B[:, :, 0, :] - B[:, :, 1, :] - B[:, :, 2, :] +
                                B[:, :, 3, :])
    return K_true, W_true, D_true


def _eval_psi_ptr_on_data(data: dict[str, Any], Z_p_quad: np.ndarray,
                          w_raw: np.ndarray) -> np.ndarray:
    w_iter = w_raw[:64]
    w_pre = w_raw[64:64 + 4 * 56].reshape(4, 56)
    w_quad = w_raw[64 + 4 * 56:]
    psi = ((data["Z_tilde68"] @ w_iter) +
           np.einsum("ktmp,ip->ktmi", data["Z_pre"], w_pre) +
           (Z_p_quad @ w_quad))
    psi -= psi.mean(axis=3, keepdims=True)
    psi /= np.std(psi, axis=(1, 3), keepdims=True) + 1e-6
    return psi


def _read_train_shard_wrapper(
        args_tuple: tuple[str, str]) -> tuple[np.ndarray, ...]:
    scaaml_dir_str, sh_path = args_tuple
    if scaaml_dir_str not in sys.path:
        sys.path.insert(0, scaaml_dir_str)
    return _read_shard_labeled(sh_path)


def build_train_only_t2_ref(train_shards: list[str],
                            out_npy: Path,
                            max_traces: int = 512) -> np.ndarray:
    comp = CompressedFile("LZ4")
    t2_sum = np.zeros(FULL_ADC_SAMPLES, dtype=np.float64)
    cnt = 0
    for sh_p in train_shards:
        buf = comp.decompress(Path(sh_p).read_bytes())
        sh = fbapi_Shard.Shard.GetRootAs(buf, 0)
        for i in range(sh.ExamplesLength()):
            t2_sum += np.frombuffer(
                sh.Examples(i).Attributes(1).AttributeBytesAsNumpy(),
                dtype=np.int16)
            cnt += 1
            if cnt >= max_traces:
                break
        if cnt >= max_traces:
            break
    ref = (t2_sum / float(max(cnt, 1))).astype(np.float32)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, ref)
    return ref


def prepare_split_tensors(
    hdr_a: np.ndarray,
    hdr_b: np.ndarray,
    remap_a: np.ndarray,
    remap_b: np.ndarray,
    pre_sbox: np.ndarray,
    traces: np.ndarray,
    whiten_models: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    from end_to_end_attack import fit_whiten, apply_whiten
    N_K, M = traces.shape[:2]
    if whiten_models is None:
        whiten_models = {
            "ra": fit_whiten(remap_a, max_keys=N_K),
            "rb": fit_whiten(remap_b, max_keys=N_K),
            "ha": fit_whiten(hdr_a, max_keys=N_K),
            "hb": fit_whiten(hdr_b, max_keys=N_K),
            "pre": fit_whiten(pre_sbox, max_keys=N_K),
            "tr": fit_whiten(traces, max_keys=N_K),
        }
    Za = apply_whiten(remap_a, whiten_models["ra"])
    Zb = apply_whiten(remap_b, whiten_models["rb"])
    Z_ha = apply_whiten(hdr_a, whiten_models["ha"])
    Z_hb = apply_whiten(hdr_b, whiten_models["hb"])
    Z_pre = apply_whiten(pre_sbox, whiten_models["pre"])
    Z = apply_whiten(traces, whiten_models["tr"])

    Za_ext = np.zeros((N_K, M, 64, 45), dtype=np.float32)
    Zb_ext = np.zeros((N_K, M, 64, 45), dtype=np.float32)
    Za_ext[:, :, :, :30] = Za
    Zb_ext[:, :, :, :30] = Zb
    Za_ext[:, :, :63, 30:] = Za[:, :, 1:, :15]
    Zb_ext[:, :, :63, 30:] = Zb[:, :, 1:, :15]

    Z_iter68 = np.stack(
        [Z[:, :, :, 22 + 50 * it:22 + 50 * it + 64] for it in range(4)], axis=3)
    S_m68 = Z_iter68.sum(axis=3)
    Z_tilde68 = Z_iter68 - Z_iter68.mean(axis=3, keepdims=True)

    Z_pro = Z[:, :, :, :22].mean(axis=2)
    Z_r10_loop = S_m68[:, :, :, 35:48].mean(axis=2)
    Za_mean = Za.mean(axis=2)
    Zb_mean = Zb.mean(axis=2)
    Fa = np.concatenate([Z_ha, Za_mean[:, :, 5:22]], axis=2)
    Fb = np.concatenate([Z_hb, Zb_mean[:, :, 5:22]], axis=2)
    iu_f = np.triu_indices(Fa.shape[2])
    Qa = Fa[:, :, iu_f[0]] * Fa[:, :, iu_f[1]]
    Qb = Fb[:, :, iu_f[0]] * Fb[:, :, iu_f[1]]
    M_base = np.concatenate([Z_pro, Z_r10_loop, Fa, Fb, Qa, Qb], axis=2)
    M_base -= M_base.mean(axis=1, keepdims=True)

    act_u_44 = np.arange(18, 62)
    S_act_44 = S_m68[:, :, :, act_u_44]
    Z_iter_44 = Z_iter68[:, :, :, :, act_u_44]
    G_feat_44 = np.einsum("ktmu,ktnv->kmnuv", S_act_44, S_act_44) / float(M)
    G_feat_44 = 0.5 * (G_feat_44 + np.swapaxes(G_feat_44, 3, 4))
    diag_corr = np.einsum("ktmiu,ktmiv->kmuv", Z_iter_44, Z_iter_44) / float(M)
    for m in range(13):
        G_feat_44[:, m, m] = 0.5 * (G_feat_44[:, m, m] * 2.0 - diag_corr[:, m])

    data = {
        "N_K": N_K,
        "M": M,
        "Z_pre": Z_pre,
        "Z_iter68": Z_iter68,
        "S_m68": S_m68,
        "Z_tilde68": Z_tilde68,
        "Za_ext": Za_ext,
        "Zb_ext": Zb_ext,
        "Z_pro": Z_pro,
        "Z_r10_loop": Z_r10_loop,
        "M_base": M_base,
        "G_feat_44": G_feat_44,
    }
    return data, whiten_models


def fit_train_only_templates(
    train_shards: list[str],
    scaaml_dir: Path,
    out_pkl: Path,
    workers: int,
) -> dict[str, Any]:
    scaaml_str = str(scaaml_dir)
    if scaaml_str not in sys.path:
        sys.path.insert(0, scaaml_str)
    os.environ[
        "PYTHONPATH"] = f"{scaaml_str}:{os.environ.get('PYTHONPATH', '')}"
    import end_to_end_attack as v2

    n_tr = len(train_shards) // 4
    if n_tr < 16:
        raise ValueError(
            "Need at least 16 train keys (64 shards) for 4-fold operator "
            f"fitting; got {n_tr}")
    labeled_shards = train_shards[:4 * n_tr]

    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_v2_worker,
            initargs=(scaaml_str,),
    ) as ex:
        res_tr = list(
            ex.map(_read_train_shard_wrapper,
                   [(scaaml_str, p) for p in labeled_shards],
                   chunksize=4))

    hdr_a_tr = np.empty((n_tr, 256, 18), dtype=np.float32)
    hdr_b_tr = np.empty((n_tr, 256, 18), dtype=np.float32)
    remap_a_tr = np.empty((n_tr, 256, 64, 30), dtype=np.float32)
    remap_b_tr = np.empty((n_tr, 256, 64, 30), dtype=np.float32)
    pre_sbox_tr = np.empty((n_tr, 256, 13, 56), dtype=np.float32)
    traces_tr = np.empty((n_tr, 256, 13, 240), dtype=np.float32)
    keys_tr_bytes = np.empty((n_tr, 32), dtype=np.uint8)
    for k in range(n_tr):
        keys_tr_bytes[k] = res_tr[4 * k][6]
        for b in range(4):
            hdr_a_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][0]
            hdr_b_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][1]
            remap_a_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][2]
            remap_b_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][3]
            pre_sbox_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][4]
            traces_tr[k, 64 * b:64 * (b + 1)] = res_tr[4 * k + b][5]
    del res_tr

    _, _, D_true_tr = build_key_labels(keys_tr_bytes)
    tr_data, whiten_models = prepare_split_tensors(hdr_a_tr, hdr_b_tr,
                                                   remap_a_tr, remap_b_tr,
                                                   pre_sbox_tr, traces_tr)
    del hdr_a_tr, hdr_b_tr, remap_a_tr, remap_b_tr, pre_sbox_tr, traces_tr

    idx_all = np.arange(n_tr)
    folds = [
        (idx_all[idx_all % 4 != f], idx_all[idx_all % 4 == f]) for f in range(4)
    ]

    D_oof = np.zeros((n_tr, 13, 4, 8), dtype=np.float32)
    G_oof = np.zeros((n_tr, 13, 13, 8), dtype=np.float64)
    d0_ops_full: dict[int, dict[str, Any]] = {}
    w_blend_all: dict[int, np.ndarray] = {}

    for j in range(8):
        oof_dir = np.zeros((n_tr, 13), dtype=np.float32)
        oof_eig = np.zeros((n_tr, 13), dtype=np.float32)
        for tr_k, val_k in folds:
            op_f = _fit_d0_operator(tr_data, D_true_tr, tr_k, j)
            val_data = {
                k_name:
                    (val_v[val_k] if isinstance(val_v, np.ndarray) else val_v)
                for k_name, val_v in tr_data.items()
            }
            val_data["N_K"] = len(val_k)
            d_dir_v, d_eig_v, G_v, _ = _apply_d0_operator(val_data, op_f)
            oof_dir[val_k] = d_dir_v
            oof_eig[val_k] = d_eig_v
            G_oof[val_k, :, :, j] = G_v

        if j < 4:
            A_oof = np.stack([oof_dir.reshape(-1), oof_eig.reshape(-1)], axis=1)
            w_blend = np.linalg.lstsq(A_oof,
                                      D_true_tr[:, :, 0, j].reshape(-1),
                                      rcond=None)[0]
        else:
            slope_oof = float(
                np.dot(oof_dir.reshape(-1), D_true_tr[:, :, 0, j].reshape(-1)) /
                (np.dot(oof_dir.reshape(-1), oof_dir.reshape(-1)) + 1e-9))
            w_blend = np.array([slope_oof, 0.0], dtype=np.float32)

        w_blend_all[j] = w_blend
        D_oof[:, :, 0, j] = w_blend[0] * oof_dir + w_blend[1] * oof_eig
        d0_ops_full[j] = _fit_d0_operator(tr_data, D_true_tr, idx_all, j)

    v_iter_tr, W_full_norm_tr, Z_p_quad_tr, s_j_vec = _compute_v_and_W_full(
        tr_data, d0_ops_full, D_oof[:, :, 0, :], s_j_vec=None)

    def fit_d123_on_keys(tr_keys: np.ndarray) -> tuple[np.ndarray, ...]:
        X_f_tr = (W_full_norm_tr[tr_keys].transpose(0, 1, 3, 2).reshape(
            -1, W_full_norm_tr.shape[2]))
        mu_f = X_f_tr.mean(axis=0, keepdims=True)
        sd_f = X_f_tr.std(axis=0, keepdims=True) + 1e-6
        X_f_tr_n = (X_f_tr - mu_f) / sd_f
        reg_f = 4000.0 * (len(tr_keys) / 128.0)
        K_f = X_f_tr_n.T @ X_f_tr_n + reg_f * np.eye(W_full_norm_tr.shape[2])
        w1 = np.linalg.solve(
            K_f, X_f_tr_n.T @ D_true_tr[tr_keys, :, 1, :].reshape(-1))
        w2 = np.linalg.solve(
            K_f, X_f_tr_n.T @ D_true_tr[tr_keys, :, 2, :].reshape(-1))
        w3 = np.linalg.solve(
            K_f, X_f_tr_n.T @ D_true_tr[tr_keys, :, 3, :].reshape(-1))
        return mu_f, sd_f, w1, w2, w3

    for tr_k, val_k in folds:
        mu_f, sd_f, w1, w2, w3 = fit_d123_on_keys(tr_k)
        for j in range(8):
            X_j_n = (W_full_norm_tr[val_k, :, :, j].reshape(
                -1, W_full_norm_tr.shape[2]) - mu_f) / sd_f
            D_oof[val_k, :, 1, j] = (X_j_n @ w1).reshape(len(val_k), 13)
            D_oof[val_k, :, 2, j] = (X_j_n @ w2).reshape(len(val_k), 13)
            D_oof[val_k, :, 3, j] = (X_j_n @ w3).reshape(len(val_k), 13)

    mu_f_full, sd_f_full, w1_full, w2_full, w3_full = fit_d123_on_keys(idx_all)
    w1_raw = w1_full / sd_f_full[0]
    w2_raw = w2_full / sd_f_full[0]
    w3_raw = w3_full / sd_f_full[0]

    D_unb_init_tr = np.zeros_like(D_oof, dtype=np.float64)
    sigma2_init = np.zeros((4, 8), dtype=np.float64)
    slopes_init = np.zeros((4, 8), dtype=np.float64)
    for j in range(8):
        for a in range(4):
            tr_p = D_oof[:, :, a, j].reshape(-1).astype(np.float64)
            tr_t = D_true_tr[:, :, a, j].reshape(-1).astype(np.float64)
            sl = float(np.dot(tr_p, tr_t) / (np.dot(tr_t, tr_t) + 1e-9))
            slopes_init[a, j] = max(sl, 0.1)
            D_unb_init_tr[:, :, a, j] = D_oof[:, :, a, j] / slopes_init[a, j]
            sigma2_init[a, j] = float(np.var((tr_p / slopes_init[a, j]) - tr_t))

    D_post_16_tr = v2.compute_16state_posterior_D(D_unb_init_tr, sigma2_init)
    psi1_ptr_tr = _eval_psi_ptr_on_data(tr_data, Z_p_quad_tr, w1_raw)
    psi2_ptr_tr = _eval_psi_ptr_on_data(tr_data, Z_p_quad_tr, w2_raw)
    psi3_ptr_tr = _eval_psi_ptr_on_data(tr_data, Z_p_quad_tr, w3_raw)

    psi1_k4_tr = psi2_ptr_tr * psi3_ptr_tr
    psi1_k4_tr -= psi1_k4_tr.mean(axis=3, keepdims=True)
    psi2_k4_tr = psi1_ptr_tr * psi3_ptr_tr
    psi2_k4_tr -= psi2_k4_tr.mean(axis=3, keepdims=True)
    psi3_k4_tr = psi1_ptr_tr * psi2_ptr_tr
    psi3_k4_tr -= psi3_k4_tr.mean(axis=3, keepdims=True)

    D_k4_tr = np.zeros((n_tr, 13, 3, 8), dtype=np.float32)
    for idx, pk in enumerate([psi1_k4_tr, psi2_k4_tr, psi3_k4_tr]):
        D_k4_tr[:, :, idx, :] = np.einsum(
            "ktmi,ktmij->kmj", pk, v_iter_tr) / (256.0 * s_j_vec[None, None, :])

    v_norm_tr = v_iter_tr / (np.std(v_iter_tr, axis=(1, 3), keepdims=True) +
                             1e-6)
    r_oof_bits = np.array(
        [
            np.mean([
                np.corrcoef(
                    D_post_16_tr[:, :, a, l].reshape(-1),
                    D_true_tr[:, :, a, l].reshape(-1),
                )[0, 1]**2 for a in (1, 2, 3)
            ]) for l in range(8)
        ],
        dtype=np.float32,
    )

    psi_tot_tr = {
        a:
            np.einsum("l,kml,ktmil->ktmi", r_oof_bits, D_post_16_tr[:, :, a, :],
                      v_norm_tr) for a in (1, 2, 3)
    }
    w_ca_models: dict[tuple[int, int], np.ndarray] = {}
    for a_idx, a in enumerate((1, 2, 3)):
        mb, mc = [m_a for m_a in (1, 2, 3) if m_a != a]
        for j in range(8):
            psi_a_bits = (
                psi_tot_tr[a] -
                r_oof_bits[j] * D_post_16_tr[:, :, a, j][:, None, :, None] *
                v_norm_tr[:, :, :, :, j])
            psi_b_bits = (
                psi_tot_tr[mb] -
                r_oof_bits[j] * D_post_16_tr[:, :, mb, j][:, None, :, None] *
                v_norm_tr[:, :, :, :, j])
            psi_c_bits = (
                psi_tot_tr[mc] -
                r_oof_bits[j] * D_post_16_tr[:, :, mc, j][:, None, :, None] *
                v_norm_tr[:, :, :, :, j])
            psi_bc_k4 = psi_b_bits * psi_c_bits
            psi_bc_k4 -= psi_bc_k4.mean(axis=3, keepdims=True)

            feat_bits = np.einsum("ktmi,ktmi->km", psi_a_bits,
                                  v_norm_tr[:, :, :, :, j]) / (256.0 * 4.0)
            feat_bc_k4 = np.einsum("ktmi,ktmi->km", psi_bc_k4,
                                   v_norm_tr[:, :, :, :, j]) / (256.0 * 4.0)
            feat_ptr_k4 = D_k4_tr[:, :, a_idx, j]

            A_tr = np.stack(
                [
                    D_oof[:, :, a, j].reshape(-1),
                    feat_bits.reshape(-1),
                    feat_ptr_k4.reshape(-1),
                    feat_bc_k4.reshape(-1),
                ],
                axis=1,
            )
            w_ca = np.linalg.lstsq(A_tr,
                                   D_true_tr[:, :, a, j].reshape(-1),
                                   rcond=None)[0]
            w_ca_models[(a, j)] = w_ca
            D_oof[:, :, a, j] = (A_tr @ w_ca).reshape(n_tr, 13)

    slopes_p1 = np.zeros((4, 8), dtype=np.float64)
    sigma2_modes = np.zeros((4, 8), dtype=np.float64)
    D_unbiased_tr = np.zeros_like(D_oof, dtype=np.float64)
    for j in range(8):
        for a in range(4):
            tr_pred = D_oof[:, :, a, j].reshape(-1).astype(np.float64)
            tr_true = D_true_tr[:, :, a, j].reshape(-1).astype(np.float64)
            slope = float(
                np.dot(tr_pred, tr_true) / (np.dot(tr_true, tr_true) + 1e-9))
            slopes_p1[a, j] = max(slope, 0.1)
            D_unbiased_tr[:, :, a, j] = D_oof[:, :, a, j] / slopes_p1[a, j]
            sigma2_modes[a, j] = float(
                np.var(D_unbiased_tr[:, :, a, j].reshape(-1) - tr_true))

    D_bp_post_tr = np.zeros((n_tr, 13, 4, 8), dtype=np.float32)
    for k in range(n_tr):
        post_ll = v2.run_coupled_walsh_tree_bp(D_unbiased_tr[k],
                                                  sigma2_modes,
                                                  n_outer=4)
        p_w = np.exp(v2.log_normalize_256(post_ll))
        B_exp = np.zeros((13, 4, 8), dtype=np.float32)
        for m in range(13):
            for q in range(4):
                q_w = (q + 1) % 4 if (m % 2 == 0) else q
                B_exp[m, q] = np.dot(p_w[m, q_w], v2.BYTE_BITS_PM1)
        D_bp_post_tr[k, :, 0, :] = 0.5 * (B_exp[:, 0, :] + B_exp[:, 1, :] +
                                          B_exp[:, 2, :] + B_exp[:, 3, :])

    w_b2_models: dict[int, np.ndarray] = {}
    slopes_p2 = np.zeros(8, dtype=np.float64)
    sigma2_p2 = sigma2_modes.copy()
    for j in range(8):
        G_j = G_oof[:, :, :, j]
        d0_g_proj = np.zeros((n_tr, 13), dtype=np.float32)
        for m in range(13):
            ref_m = [mp for mp in v2.M_TREE_STRONG if mp != m]
            num = np.einsum("kn,kn->k", G_j[:, m, ref_m], D_bp_post_tr[:, ref_m,
                                                                       0, j])
            den = np.sum(D_bp_post_tr[:, ref_m, 0, j]**2, axis=1) + 1e-4
            d0_g_proj[:, m] = num / den
        A_tr = np.stack([D_oof[:, :, 0, j].reshape(-1),
                         d0_g_proj.reshape(-1)],
                        axis=1)
        w_b2 = np.linalg.lstsq(A_tr,
                               D_true_tr[:, :, 0, j].reshape(-1),
                               rcond=None)[0]
        w_b2_models[j] = w_b2
        d0_p2 = (A_tr @ w_b2).reshape(n_tr, 13)
        tr_p = d0_p2.reshape(-1).astype(np.float64)
        tr_t = D_true_tr[:, :, 0, j].reshape(-1).astype(np.float64)
        sl = float(np.dot(tr_p, tr_t) / (np.dot(tr_t, tr_t) + 1e-9))
        slopes_p2[j] = max(sl, 0.1)
        sigma2_p2[0, j] = float(np.var((tr_p / slopes_p2[j]) - tr_t))

    mdl = {
        "whiten_models": whiten_models,
        "d0_ops_full": d0_ops_full,
        "w_blend_all": w_blend_all,
        "s_j_vec": s_j_vec,
        "mu_f_full": mu_f_full,
        "sd_f_full": sd_f_full,
        "w1_full": w1_full,
        "w2_full": w2_full,
        "w3_full": w3_full,
        "w1_raw": w1_raw,
        "w2_raw": w2_raw,
        "w3_raw": w3_raw,
        "slopes_init": slopes_init,
        "sigma2_init": sigma2_init,
        "r_oof_bits": r_oof_bits,
        "w_ca_models": w_ca_models,
        "slopes_p1": slopes_p1,
        "sigma2_modes": sigma2_modes,
        "w_b2_models": w_b2_models,
        "slopes_p2": slopes_p2,
        "sigma2_p2": sigma2_p2,
    }
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f_out:
        pickle.dump(mdl, f_out)
    return mdl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit `train`-only templates for `end_to_end_attack.py`.")
    parser.add_argument("--work-dir",
                        type=Path,
                        default=Path("test_run"),
                        help="Working directory (default: test_run)")
    parser.add_argument("--scaaml-dir",
                        type=Path,
                        default=DEFAULT_SCAAML_DIR,
                        help="Path to `end_to_end_attack.py`")
    parser.add_argument("--workers",
                        type=int,
                        default=min(32,
                                    os.cpu_count() or 8))
    args = parser.parse_args()

    t0 = time.time()
    work_dir = args.work_dir.resolve()
    scaaml_dir = args.scaaml_dir.resolve()
    if not (scaaml_dir / "end_to_end_attack.py").is_file():
        raise FileNotFoundError(
            f"Missing `end_to_end_attack.py` in `{scaaml_dir}`.")

    train_shards = [
        str(p) for p in sorted((work_dir / "dataset" / "train" /
                                "shards").glob("*.fb"))
    ]
    if not train_shards:
        raise FileNotFoundError("No train shards found in "
                                f"`{work_dir / 'dataset/train/shards'}`.")

    prof_dir = work_dir / "profiling"
    prof_dir.mkdir(parents=True, exist_ok=True)
    out_ref = prof_dir / "canonical_t2_ref.npy"
    out_pkl = prof_dir / "trace2_walsh_tree_bp_templates.pkl"

    print("[*] Building `canonical_t2_ref.npy` from first 512"
          " `train` traces (0 holdout)...")
    ref = build_train_only_t2_ref(train_shards, out_ref, max_traces=512)

    print(
        f"[*] Fitting Walsh/Tree-BP templates strictly on {len(train_shards)} "
        f"`train` shards ({len(train_shards) // 4} keys, 0 holdout)...")
    mdl = fit_train_only_templates(train_shards,
                                   scaaml_dir,
                                   out_pkl,
                                   workers=args.workers)

    summary = {
        "train_shards_used": len(train_shards),
        "train_keys_used": len(train_shards) // 4,
        "holdout_shards_used": 0,
        "canonical_t2_ref_path": str(out_ref),
        "canonical_t2_ref_mean": round(float(np.mean(ref)), 3),
        "templates_pkl_path": str(out_pkl),
        "r_oof_bits": [round(float(x), 4) for x in mdl["r_oof_bits"]],
        "wall_time_sec": round(time.time() - t0, 2),
    }
    (prof_dir / "profiling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Fitted strict `train`-only templates in "
          f"{summary['wall_time_sec']:.1f}s -> `{out_pkl}`")


if __name__ == "__main__":
    main()
