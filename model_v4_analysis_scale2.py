# real, imag 바꾸려면 True-> False로 바꿔야함 두 군데


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def extract_segments(y01: np.ndarray):
    """
    y01: (L,) np.ndarray, values in {0,1}
    return: list of (start, end, length)
    """
    segments = []
    in_seg = False
    for i, v in enumerate(y01):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            end = i - 1
            segments.append((start, end, end - start + 1))
            in_seg = False
    if in_seg:
        end = len(y01) - 1
        segments.append((start, end, end - start + 1))
    return segments


@torch.no_grad()
def collect_segment_scale_attention(
    model,
    loader,
    task_idx=1,          # 1: protein, 2: rna, 3: dna
    use_real=False,       # True: att_R, False: att_I
    device="cuda"
):
    """
    return:
        seg_lens:  List[int]
        seg_atts:  np.ndarray (N_segments, S)
    """
    model.eval()

    seg_lens = []
    seg_atts = []

    for batch in loader:
        H = batch["emb"].to(device)        # (B,L,d)
        Y = batch["label"].to(device)      # (B,L,4)
        M = batch["mask"].to(device)       # (B,L)

        outputs = model(H, M)
        att = outputs["att_R"] if use_real else outputs["att_I"]  # (B,L,S)

        B, L, S = att.shape

        for b in range(B):
            y = Y[b, :, task_idx]
            valid = M[b] & (y > -50)

            if not valid.any():
                continue

            y01 = (y[valid] == 1).long().cpu().numpy()
            att_b = att[b][valid].detach().cpu().numpy()  # (L_valid, S)

            segments = extract_segments(y01)

            for (st, ed, seg_len) in segments:
                seg_att = att_b[st:ed+1].mean(axis=0)  # (S,)
                seg_lens.append(seg_len)
                seg_atts.append(seg_att)

    if len(seg_atts) == 0:
        return [], np.empty((0, S))

    return seg_lens, np.vstack(seg_atts)

def assign_segment_groups(seg_lens, q33=33, q66=66):
    """
    return: list of 'short' | 'mid' | 'long'
    """
    q1 = np.percentile(seg_lens, q33)
    q2 = np.percentile(seg_lens, q66)

    groups = []
    for l in seg_lens:
        if l <= q1:
            groups.append("short")
        elif l <= q2:
            groups.append("mid")
        else:
            groups.append("long")
    return groups, q1, q2

def group_scale_attention(seg_atts, groups):
    """
    seg_atts: (N_segments, S)
    groups:   list[str] length N_segments
    """
    out = {"short": [], "mid": [], "long": []}
    for att, g in zip(seg_atts, groups):
        out[g].append(att)

    for k in out:
        if len(out[k]) > 0:
            out[k] = np.vstack(out[k])  # (N_k, S)
        else:
            out[k] = np.empty((0, seg_atts.shape[1]))
    return out

def plot_segment_scale_boxplot(
    grouped_atts,
    scales,
    out_path,
    title="Segment-level Scale Attention"
):
    import matplotlib.pyplot as plt

    groups = ["short", "mid", "long"]
    colors = {"short": "#1f77b4", "mid": "#ff7f0e", "long": "#2ca02c"}

    S = len(scales)
    width = 0.22
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(2.6 * S, 6))

    data, positions, box_colors = [], [], []

    for s_idx in range(S):
        for g_idx, g in enumerate(groups):
            if grouped_atts[g].shape[0] == 0:
                continue
            data.append(grouped_atts[g][:, s_idx])
            positions.append(s_idx + 1 + offsets[g_idx])
            box_colors.append(colors[g])

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.9,
        showfliers=False,
        patch_artist=True
    )

    for box, c in zip(parts["boxes"], box_colors):
        box.set_facecolor(c)
        box.set_alpha(0.7)
        box.set_edgecolor("black")

    for med in parts["medians"]:
        med.set_color("black")
        med.set_linewidth(2)

    ax.set_xticks(range(1, S + 1))
    ax.set_xticklabels(scales)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Scale Attention Weight")
    ax.set_title(title)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=colors[g], label=g) for g in groups],
        title="Segment length"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_pr_curves(test_metrics, out_dir, suffix=""):
    names = ['disorder', 'protein', 'rna', 'dna']
    roc_raw = test_metrics['roc_raw']

    best_f1_per_task = []

    for t_idx, name in enumerate(names):
        y_true = roc_raw['y_true'][t_idx]
        y_prob = roc_raw['y_prob'][t_idx]

        if len(y_true) == 0 or len(np.unique(y_true)) == 1:
            print(f"[WARN] Cannot plot ROC/PR for task {name} (no positive or negative examples)")
            continue

        # ----------------- ROC -----------------
        roc_auc = roc_auc_score(y_true, y_prob)
        # fpr, tpr, _ = roc_curve(y_true, y_prob)
        # roc_auc_val = auc(fpr, tpr)

        # plt.figure(figsize=(5,5))
        # plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
        # plt.plot([0,1],[0,1],'k--')
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title(f"ROC Curve - {name}")
        # plt.legend(loc="lower right")
        # plt.tight_layout()
        # plt.savefig(f"{out_dir}/roc_{name}{suffix}.png", dpi=200)
        # plt.close()

        # ----------------- PR Curve -----------------
        P, R, TH = precision_recall_curve(y_true, y_prob)

        if len(TH) > 0:
            F = 2 * P[1:] * R[1:] / (P[1:] + R[1:] + 1e-12)
            best_idx = np.argmax(F)
            best_f1 = F[best_idx]
            best_th = TH[best_idx]
            best_p = P[best_idx + 1]
            best_r = R[best_idx + 1]
        else:
            best_f1 = best_th = best_p = best_r = float('nan')

        pr_auc_val = auc(R, P)

        plt.figure(figsize=(5,5))
        plt.plot(R, P, label=f"AUC={roc_auc:.4f}, AUPR={pr_auc_val:.4f}, F1={best_f1:.4f}, Rec={best_r:.4f}")

        if not np.isnan(best_f1):
            plt.scatter(best_r, best_p, color='red', s=40, label=f"Best-F1@th={best_th:.3f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_{name}_{suffix}rna.png", dpi=200)
        plt.close()

        print(f"[PLOT] Saved PR curve for '{name}{suffix}' (Best F1={best_f1:.4f})")
        best_f1_per_task.append(best_f1)
    return best_f1_per_task



def log(s: str):
    print(f'[LOG] {s}')

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ids(npz_ids):
    n = len(npz_ids)
    idx = np.arange(n)
    n_val = 0
    train_ids = [npz_ids[i] for i in range(n)]

    return train_ids


def parse_thresholds(s: str, default=0.5):
    """
    예) 'disorder=0.4,protein=0.6,rna=0.35,dna=0.55'
    빠진 키는 default로 채움.
    """
    names = ['disorder', 'protein', 'rna', 'dna']
    out = {k: default for k in names}

    if not s:
        return out
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split('=', 1)
        k = k.strip().lower()
        v = float(v.strip())
        if k in out:
            out[k] = v
    return out





class IDREmbDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        label_files: Dict[str,str], # keys: disorder, protein, rna, dna
        id_key: str="ids",
        per_res_key: str= "per_residue",
        max_len: Optional[int] =None,
        include_ids: Optional[set] =None,
    ):
        """
        npz must have keys: 'ids' (object array of str), 'per_residue' (object array of arrays (L_i, d))
        label_files: dict mapping task -> path
        """
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=True)
        self.ids: List[str] = list(map(str,self.npz[id_key]))
        self.per_residue = list(self.npz[per_res_key])# list of (L_i, d) float32
        self.dim = int(self.npz.get('dim', self.per_residue[0].shape[-1]))
        self.max_len = max_len
        self.include_ids = set(include_ids) if include_ids is not None else None

        # Load all labels into dict: task->{id:np.ndarray[L]}
        self.tasks = ['disorder', 'protein', 'rna', 'dna']
        for t in self.tasks:
            if t not in label_files:
                raise ValueError(f"Missinglabel file for task '{t}'. Provided keys: {list(label_files.keys())}")
        self.label_maps: Dict[str, Dict[str, np.ndarray]] ={
            t: self._load_label_txt(label_files[t]) for t in self.tasks
        }

        self.id_to_rawidx = {sid: i for i, sid in enumerate(self.ids)}


        # Sanity check lengths & gather indices that are fully labeled
        self.valid_indices: List[int] =[]
        dropped: List[Tuple[str, int, Dict[str, int]]]=[]
        for i, sid in enumerate(self.ids):
            if self.include_ids is not None and sid not in self.include_ids:
                continue
            L = self.per_residue[i].shape[0]
            ok = True
            mis = {}
            for t in self.tasks:
                arr = self.label_maps[t].get(sid)
                if arr is None:
                    ok = False
                    mis[t] = -1
                elif len(arr) != L:
                    ok = False
                    mis[t] = len(arr)
            if ok:
                self.valid_indices.append(i)
            else:
                dropped.append((sid, L, mis))
        if dropped:
            log(f'Dropped {len(dropped)} sequences with missing/length-mismatched labels (showing up to 3): {dropped[:30]}')
        log(f'Dataset ready. kept={len(self.valid_indices)} / total={len(self.ids)}, dim={self.dim}')


    @staticmethod
    def _parse_label_line(line: str)-> Optional[List[int]]:
        line = line.strip()
        if not line:
            return None
        if set(line) <= set("01") and len(line) >1:
            return [int(c) for c in line]
        else:
            return None

    def _load_label_txt(self, path:str) -> Dict[str, np.ndarray]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'label file not found: {path}')
        mp: Dict[str, np.ndarray] ={}
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f]
        for ln in lines:
            if not ln.strip():
                continue
            sid, payload = ln.split('\t',1)
            arr = self._parse_label_line(payload)
            if arr is None:
                raise ValueError(f'Could not parse labels in TSV line: {ln}')
            mp[sid] = np.asarray(arr, dtype=np.int64)
        return mp


    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx:int):
        i = self.valid_indices[idx]
        sid = self.ids[i]
        H = self.per_residue[i].astype(np.float32) #(L, d)
        L, d = H.shape
        y = np.zeros((L,4), dtype=np.float32)
        for j, t in enumerate(self.tasks):
            y[:, j] = self.label_maps[t][sid].astype(np.float32)
        if self.max_len and L > self.max_len:
            start = (L - self.max_len) //2
            H = H[start:start+self.max_len]
            y = y[start:start+self.max_len]
            L = self.max_len
        return {
            "id": sid,
            "emb": torch.from_numpy(H), # (L, d)
            "label": torch.from_numpy(y), #(L, 4)
            "len": L,
        }


def collate_pad(batch: List[dict]):
    # pad to max L within batch
    Ls = [b["len"] for b in batch]
    Lmax = max(Ls)
    d = batch[0]['emb'].shape[-1]
    B = len(batch)
    
    emb = torch.zeros(B, Lmax, d, dtype=torch.float32)
    lab = torch.full((B, Lmax, 4), fill_value=-100.0, dtype=torch.float32) # for BCE ignore-mask via mask multiply
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    ids = []
    for i, b in enumerate(batch):
        L = b["len"]
        emb[i, :L] = b['emb']
        lab[i, :L] = b['label']
        mask[i, :L] = True
        ids.append(b['id'])
    return {'ids': ids, 'emb':emb, 'label': lab, 'mask':mask}




def build_loader(cfg, use_ids=None, test_flag=None):
    if test_flag is None:
        npz = np.load(cfg.npz_path, allow_pickle=True)
    else:
        npz = np.load(cfg.test_npz_path, allow_pickle=True)
    
    ids = [str(x) for x in npz['ids']]
    per_res = list(npz['per_residue']) #각 (L_i, d)
    d = int(npz['dim']) if 'dim' in npz else per_res[0].shape[1]

    select = list(range(len(ids)))
    selected_ids = list(range(len(ids)))
    if use_ids is not None:
        wanted = set(use_ids)
        select = [i for i, sid in enumerate(ids) if sid in wanted]
        selected_ids = [sid for i, sid in enumerate(ids) if sid in wanted]

    if test_flag is None:
        ds = IDREmbDataset(
            npz_path = cfg.npz_path,
            label_files={
                "disorder": cfg.disorder_txt,
                "protein": cfg.protein_txt,
                "rna":cfg.rna_txt,
                "dna": cfg.dna_txt,
            },
            max_len = cfg.max_len,
            include_ids = set(selected_ids)
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, collate_fn=collate_pad, pin_memory=True)
    else:
        ds = IDREmbDataset(
            npz_path = cfg.test_npz_path,
            label_files={
                "disorder": cfg.test_disorder_txt,
                "protein": cfg.test_protein_txt,
                "rna":cfg.test_rna_txt,
                "dna": cfg.test_dna_txt,
            },
            max_len = cfg.max_len,
            include_ids = set(selected_ids)
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pad, pin_memory=True)
    

class ComplexCWT(nn.Module):
    """
    Continuous Wavelet Transform (Morlet) implemented via conv1d.
    Produces complex output: (real, imag)
    """
    def __init__(self, in_dim, scales, wavelet_dim, kernel_size=5):
        super().__init__()
        self.scales = scales
        self.D_w = wavelet_dim
        self.kernel_size = kernel_size

        # 1×1 conv to reduce PLM embedding dim → wavelet channel dim
        self.dim_reduce = nn.Conv1d(in_dim, wavelet_dim, kernel_size=1)

        # Correct kernel shape:
        # (num_scales, out_channels=D_w, in_channels=D_w, kernel_size)
        self.kernels_real = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )
        self.kernels_imag = nn.Parameter(
            torch.randn(len(scales), wavelet_dim, wavelet_dim, kernel_size)
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)   # → (B, D, L)
        x = self.dim_reduce(x)  # → (B, D_w, L)

        real_out = []
        imag_out = []

        for i, s in enumerate(self.scales):
            ker_r = self.kernels_real[i]  # (D_w, D_w, K)
            ker_i = self.kernels_imag[i]

            r = F.conv1d(x, ker_r, padding=(self.kernel_size//2)*s, dilation=s)  # (B, D_w, L)
            m = F.conv1d(x, ker_i, padding=(self.kernel_size//2)*s, dilation=s)

            real_out.append(r)
            imag_out.append(m)

        # stack scales
        real_out = torch.stack(real_out, dim=2)  # (B, D_w, S, L)
        imag_out = torch.stack(imag_out, dim=2)

        return real_out, imag_out


class ScaleAttention(nn.Module):
    def __init__(self, D_plm, D_att, D_w):
        super().__init__()
        self.W_Q = nn.Linear(D_plm, D_att)
        self.W_K = nn.Linear(D_w, D_att)
        self.W_V = nn.Linear(D_w, D_att)

    def forward(self, E, W_red):
        # E: (B, L, D_plm)
        # W_red: (B,D_w, S,L)

        B, D_w, S, L = W_red.shape
        D_plm = E.shape[-1]

        Q = self.W_Q(E) # (B, L, D_att)

        W_red_perm = W_red.permute(0, 3, 2, 1)
        W_red_flat = W_red_perm.reshape(B*L, S, D_w)

        K = self.W_K(W_red_flat)  # (B*L, S, D_att)
        V = self.W_V(W_red_flat)

        Q_flat = Q.reshape(B*L, 1, -1)
        att = torch.softmax((Q_flat @ K.transpose(1, 2)) / math.sqrt(K.shape[-1]), dim=-1)
        ctx = att @ V

        ctx = ctx.reshape(B, L, -1)

        return ctx, att.reshape(B,L,S)


class ComplexInteractionBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.a = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))
        self.fuse = nn.Linear(2*D, D)

    def forward(self, R, I):
        R_ = self.a * R - self.b * I
        I_ = self.b * R + self.a * I
        out = torch.cat([R_, I_], dim=-1)
        return self.fuse(out)






class TinyMamba(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(D, 4*D),
            nn.GELU(),
            nn.Linear(4*D, D)
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x):
        return self.norm(x + self.ff(x))

class MambaLite(nn.Module):
    """
    Lightweight Mamba-like selective SSM layer
    Pure PyTorch implementation (no CUDA kernels required)

    Shape:
        x: (B, L, D)
    """
    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=4):
        super().__init__()

        hidden = d_model * expand

        # 1) selective gating
        self.gate_proj = nn.Linear(d_model, hidden)
        self.x_proj = nn.Linear(d_model, hidden)

        # 2) Convolutional mixing (1D depthwise conv)
        self.dwconv = nn.Conv1d(
            hidden, hidden, kernel_size=conv_kernel,
            groups=hidden, padding=conv_kernel //2
        )

        # 3) State projection(SSM-style recurrent kernel, simplified)
        self.ss_out = nn.Linear(hidden, d_model)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B,L,D)
        B, L, D = x.shape

        gate = torch.sigmoid(self.gate_proj(x)) # (B, L, H)
        u = self.x_proj(x) # (B, L, H)

        # conv mixing (depthwise)
        h = u.transpose(1, 2) # (B, H, L)
        h = self.dwconv(h)
        h = h.transpose(1, 2) # (B, L, H)

        # selective gating
        h = h * gate

        # project back
        out = self.ss_out(h) # (B, L, D)

        # residual + norm
        return self.norm(x+out)



class MambaPython(nn.Module):
    """
    Stable Pure-Python Mamba (Selective SSM)
    - NaN 방지 (A 안정화 + dt 제한 + 안전한 recurrence)
    - CUDA kernel 없이 PyTorch만으로 작동
    """

    def __init__(self, d_model, d_state=16, expand=2, conv_kernel=3):
        super().__init__()

        self.d_model = d_model
        hidden = d_model * expand   # H
        self.hidden = hidden
        self.d_state = d_state

        # 1) main projections
        self.x_proj = nn.Linear(d_model, hidden)
        self.g_proj = nn.Linear(d_model, hidden)

        # 2) depthwise conv (mixing)
        self.dwconv = nn.Conv1d(
            hidden, hidden,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=hidden
        )

        # 3) Raw parameters for A, B, C
        # A_raw → stable A = -softplus(A_raw)
        self.A_raw = nn.Parameter(torch.randn(hidden, d_state))
        self.B = nn.Parameter(torch.randn(hidden, d_state))
        self.C = nn.Parameter(torch.randn(hidden, d_state))

        # 4) dt projection
        self.dt_proj = nn.Linear(d_model, hidden)

        # 5) output projection
        self.out_proj = nn.Linear(hidden, d_model)

        self.norm = nn.LayerNorm(d_model)

    def discretize(self, A, B, dt):
        """
        Stable discretization:
        A_d = exp(A * dt)
        B_d = (exp(A*dt) - 1) / A * B
        """
        # A: (H,S)
        # dt: (H,) → (H,1)
        dt = dt.unsqueeze(-1)  # (H,1)

        A_dt = A * dt              # (H,S)
        A_d = torch.exp(A_dt)      # (H,S)

        # avoid division by zero
        A_safe = A + 1e-6

        B_d = ((A_d - 1.0) / A_safe) * B  # (H,S)

        return A_d, B_d

    def selective_scan(self, A_d, B_d, C, u):
        """
        SSM recurrence:
        x_{t+1} = A_d ⊙ x_t + B_d ⊙ u_t
        y_t = C^T x_t

        A_d, B_d, C: (H,S)
        u: (B,L,H)
        return: (B,L,H)
        """
        B_, L, H = u.shape
        S = A_d.shape[1]

        # hidden state
        x = torch.zeros(B_, H, S, device=u.device)
        outputs = []

        for t in range(L):
            u_t = u[:, t, :].unsqueeze(-1)  # (B,H,1)
            x = A_d.unsqueeze(0) * x + B_d.unsqueeze(0) * u_t

            y_t = torch.sum(x * C.unsqueeze(0), dim=-1)  # (B,H)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B,L,H)

    def forward(self, x):
        """
        x: (B,L,D)
        """
        B, L, D = x.shape

        # 1) projections
        u = self.x_proj(x)                   # (B,L,H)
        g = torch.sigmoid(self.g_proj(x))    # (B,L,H)

        # 2) depthwise conv mixing
        h = u.transpose(1, 2)                # (B,H,L)
        h = self.dwconv(h)
        h = h.transpose(1, 2)                # (B,L,H)

        # 3) gating
        u = u * g + h * (1 - g)              # (B,L,H)

        # 4) dt selection (stabilized)
        dt_full = F.softplus(self.dt_proj(x))    # (B,L,H)
        dt = dt_full.mean(dim=(0, 1))            # (H,)
        dt = torch.clamp(dt, 0.001, 2.0)         # ★ stabilize exp()

        # 5) stable A
        A = -F.softplus(self.A_raw)              # A < 0 ensures decay

        # 6) discretization
        A_d, B_d = self.discretize(A, self.B, dt)

        # 7) selective SSM scan
        y = self.selective_scan(A_d, B_d, self.C, u)  # (B,L,H)

        # 8) output projection + residual
        out = self.out_proj(y)                       # (B,L,D)

        return self.norm(x + out)











# scales 원래 [2, 4, 6, 8] 였는데, [1, 2, 4, 8, 16]로 바꿈
# scales 원래 ~였는데 scales = [1, 2, 3, 4, 6, 8]로 바꿈

class IDRModel(nn.Module):
    def __init__(self, d, D_att=128, D_w=128, scales=[1, 2, 3, 4, 6, 8, 16]):
        super().__init__()
        self.d_plm = d
        self.D_w = D_w
        self.D_att = D_att
        self.heads = nn.ModuleDict({
            "disorder": nn.Linear(d,1),
            "protein": nn.Linear(d,1),
            "rna":nn.Linear(d,1),
            "dna": nn.Linear(d,1),
        })

        # 1. CWT
        self.cwt = ComplexCWT(in_dim=d, scales=scales, wavelet_dim=D_w)
        
        # 2. Residue-wise Scale Cross-Attention
        self.real_att = ScaleAttention(d, D_att, D_w)
        self.imag_att = ScaleAttention(d, D_att, D_w)

        # 3. CIB
        self.cib = ComplexInteractionBlock(D_att)

        # 4.Residual fusion
        self.out_proj = nn.Linear(D_att, d)

        # 5. Mamba
        # self.mamba = TinyMamba(d)
        # self.mamba = MambaLite(d_model=d, d_state=16, expand=2, conv_kernel=3) # conv_kernel은 무조건 홀수로
        # self.mamba = MambaPython(d_model=d, d_state=16, expand=2, conv_kernel=3)
        self.mamba_blocks = nn.ModuleList([
            MambaPython(d_model=d, d_state=16, expand=2, conv_kernel=3)
            for _ in range(3)
        ])


    

    def forward(self, emb, mask=None):
        # emb: (B, L, d)
        real, imag = self.cwt(emb) #(B, D_w, S, L)

        ctx_R, att_R = self.real_att(emb, real) #(B, L, D_att)
        ctx_I, att_I = self.imag_att(emb, imag)

        fused = self.cib(ctx_R, ctx_I)

        E_fused = emb + self.out_proj(fused)

        # H = self.mamba(E_fused)
        H = E_fused
        for blk in self.mamba_blocks:
            H = blk(H)

        out = {k: self.heads[k](H).squeeze(-1) for k in self.heads} # logits (B,L)
        return {
            "logits": out,
            "att_R": att_R,
            "att_I": att_I
        }





def masked_bce_with_logits(logits:torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight=None) -> torch.Tensor:
    # logits/targets: (B,L), mask: (B,L)
    logits = logits[mask]
    targets= targets[mask]
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, gamma: float=2.0, alpha: float=0.75)-> torch.Tensor:
    # Manual focal in masked
    logits = logits[mask]
    targets = targets[mask]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p=torch.sigmoid(logits)
    pt = targets * p + (1-targets) * (1-p)
    # added 
    alpha_added = targets * alpha + (1-targets) *(1-alpha)

    focal = (alpha_added * (1-pt) ** gamma) * bce
    return focal.mean()

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, mask:torch.Tensor, eps:float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs[mask]
    targets = targets[mask]
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + eps
    dice = 1 -(2 * inter + eps) /denom
    return dice






def train_one_epoch(model, loader, optimizer, cfg, epoch:int):
    model.train()
    total=0.0
    for step, batch in enumerate(loader):
        H = batch['emb'].to(cfg.device) # (B,L,d)
        Y = batch['label'].to(cfg.device) # (B,L,4) with -100 for pads
        M = batch['mask'].to(cfg.device) # (B,L)

        logits = model(H,M) # dict of (B,L)
        loss_sum = 0.0
        names = ['disorder', 'protein', 'rna', 'dna']
        # task_weights = {
        #         "disorder": 1.0,
        #         "protein": 3.0,
        #         "rna": 15.0,
        #         "dna": 10.0,
        #     }

        task_weights = {
                "disorder": 1.0,
                "protein": 2.43,
                "rna": 27.61,
                "dna": 17.63,
            }

        pos_weight_map = {
            "disorder": torch.tensor(3.51, device=cfg.device),
            "protein":  torch.tensor(9.98, device=cfg.device),
            "rna":      torch.tensor(123.63, device=cfg.device),
            "dna":      torch.tensor(78.59, device=cfg.device),
        }

        for j, name in enumerate(names):
            if name =='disorder':
                continue
            if name =='protein':
                continue
            if name =='dna':
                continue
            y = Y[:,:,j]
            mask = M &(y > -50) # sanity

            pw = pos_weight_map[name]
            l = masked_bce_with_logits(logits[name], y, mask, pos_weight=pw)

            if cfg.use_focal:
                # print('use focal')
                l = l + focal_bce_with_logits(logits[name], y, mask, alpha=cfg.alphas[name])
            if cfg.use_dice:
                # print('use dice')
                l = l + dice_loss(logits[name], y, mask)
            task_w = task_weights[name]
            loss_sum = loss_sum + l* task_w
        optimizer.zero_grad(set_to_none=True)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total +=loss_sum.item()
        # if (step + 1)% 50 ==0:
        #     log(f'epoch {epoch} step {step+1}/{len(loader)} loss={total/(step+1):.4f}')
    return total/max(len(loader),1)




@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    total_loss = 0.0
    nsteps = 0

    names = ['disorder', 'protein', 'rna', 'dna']
    n_tasks = len(names)

    scale = len(model.cwt.scales) # S
    pos_att_R = {t: [] for t in ['disorder','protein','rna','dna']}
    pos_att_I = {t: [] for t in ['disorder','protein','rna','dna']}

    # for PRF
    tp = torch.zeros(4, device=cfg.device)
    tn = torch.zeros(4, device=cfg.device)
    fp = torch.zeros(4, device=cfg.device)
    fn = torch.zeros(4, device=cfg.device)

    # for AUC (collect all valid probs/labels per task)
    y_true_all = [[] for _ in range(n_tasks)]
    y_prob_all = [[] for _ in range(n_tasks)]

    for batch in loader:
        H = batch['emb'].to(cfg.device)     # (B,L,d)
        Y = batch['label'].to(cfg.device)   # (B,L,4) with -100 for pads
        M = batch['mask'].to(cfg.device)    # (B,L), bool

        outputs = model(H, M)                # dict: name -> (B,L)
        logits = outputs['logits']
        att_R = outputs['att_R'] # (B,L,S)
        att_I = outputs['att_I']

        loss_sum = 0.0

        for j, name in enumerate(names):
            y = Y[:, :, j]                  # (B,L) 0/1 or -100
            valid = M & (y > -50)           # (B,L) 유효 위치
            
            pos_mask = (y == 1) & valid # (B,L)
            if pos_mask.any():
                # (B,L,S) → (N_pos,S)
                att_r_pos_vals = att_R[pos_mask]   # (N_pos, S)
                att_i_pos_vals = att_I[pos_mask]

                # 리스트에 저장
                pos_att_R[name].append(att_r_pos_vals.detach().cpu().numpy())
                pos_att_I[name].append(att_i_pos_vals.detach().cpu().numpy())



            l = masked_bce_with_logits(logits[name], y, valid)
            if cfg.use_focal:
                l = l + focal_bce_with_logits(logits[name], y, valid, alpha=cfg.alphas[name])
            if cfg.use_dice:
                l = l + dice_loss(logits[name], y, valid)
            loss_sum = loss_sum + l

            # ---- metrics (0.5 threshold) ----
            if valid.any():
                prob = torch.sigmoid(logits[name])          # (B,L)
                # ------PRF-------
                # pred = (prob >= 0.5).float()[valid]         # (N_valid,)
                thr = cfg.thresholds.get(name, 0.5)
                pred = (prob >= thr).float()[valid]
                tgt  = y[valid].float()                     # (N_valid,)
                tp[j] += (pred * tgt).sum()
                fp[j] += (pred * (1 - tgt)).sum()
                fn[j] += ((1 - pred) * tgt).sum()
                total_valid = valid.sum()
                tn[j] += total_valid - (pred * tgt).sum() - (pred * (1- tgt)).sum() - ((1 - pred) * tgt).sum()
                # -----AUC-------
                y_true_all[j].append(tgt.detach().cpu().numpy())
                y_prob_all[j].append(prob[valid].detach().cpu().numpy())

        total_loss += loss_sum.item()
        nsteps += 1
    test_loss = total_loss / max(nsteps, 1)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    


    # per-task AUC
    auc = []
    best_thresh_youden =[]
    best_thresh_fbeta=[]

    for j in range(n_tasks):
        if len(y_true_all[j]) == 0:
            auc.append(float('nan'))
            continue
        y_true = np.concatenate(y_true_all[j], axis=0)
        y_prob = np.concatenate(y_prob_all[j], axis=0)
        # AUC은 양/음성 모두 있어야 정의됨
        if (y_true.max() == y_true.min()):
            auc.append(float('nan'))
        else:
            try:
                auc.append(float(roc_auc_score(y_true, y_prob)))
                # for testing
                fpr_t, tpr_t, thr_t = roc_curve(y_true, y_prob)
                j_scores_t = tpr_t - fpr_t
                best_thresh_youden.append(thr_t[np.argmax(j_scores_t)])
            except Exception:
                auc.append(float('nan'))

        pos_cnt = int(y_true.sum())
        if pos_cnt == 0 or pos_cnt == y_true.size:
            # 모두 음성 또는 모두 양성 → PR 임계값 무의미
            best_thresh_fbeta.append(np.nan)
        else:
            try:
                P, R, TH = precision_recall_curve(y_true, y_prob)
                # TH 길이는 len(P)-1 == len(R)-1 이므로, P[1:], R[1:]와 정렬
                if len(TH) == 0:
                    best_thresh_fbeta.append(float(default_thresh))
                else:
                    F = (2.0 * P[1:] * R[1:]) / (P[1:] + R[1:] + 1e-12)
                    idx = int(np.nanargmax(F))
                    best_thresh_fbeta.append(float(TH[idx]))
            except Exception:
                best_thresh_fbeta.append(np.nan)
    

    # ----- macro/micro over multi-label (protein, rna, dna) -----
    # indices 1,2,3
    idxs = [1, 2, 3]
    # macro: arithmetic mean over tasks
    macro_sens = float(recall[idxs].mean().detach().cpu())
    macro_f1   = float(f1[idxs].mean().detach().cpu())

    # micro: aggregate TP/FP/FN then compute
    TP = float(tp[idxs].sum().detach().cpu())
    FP = float(fp[idxs].sum().detach().cpu())
    FN = float(fn[idxs].sum().detach().cpu())
    micro_sens = TP / (TP + FN + 1e-8)
    micro_f1   = 2 * TP / (2 * TP + FP + FN + 1e-8)

    final_attR = {}
    final_attI = {}
    final_attR_dist = {}
    final_attI_dist = {}

    for task in names:
        if len(pos_att_R[task]) > 0:
            all_r = np.vstack(pos_att_R[task])  # (N_pos, S)
            all_i = np.vstack(pos_att_I[task])
            final_attR[task] = all_r.mean(axis=0)
            final_attI[task] = all_i.mean(axis=0)

            final_attR_dist[task] = all_r        # ★ 분포 그대로
            final_attI_dist[task] = all_i
        else:
            final_attR[task] = np.zeros(scale)
            final_attI[task] = np.zeros(scale)
            final_attR_dist[task] = np.empty((0, scale))
            final_attI_dist[task] = np.empty((0, scale))


    return {
        'loss': float(test_loss),
        'precision': precision.detach().cpu().tolist(),   # [dis, pro, rna, dna]
        'recall':    recall.detach().cpu().tolist(),      # sensitivity
        'f1':        f1.detach().cpu().tolist(),
        'auc':       auc,                                  # per-task AUC
        'macro': {
            'sensitivity': macro_sens,
            'f1':          macro_f1,
        },
        'micro': {
            'sensitivity': micro_sens,
            'f1':          micro_f1,
        },
        'best_thr': best_thresh_youden,
        'best_thr_f1':best_thresh_fbeta,
        'confusion': {
            'tp': tp.detach().cpu().tolist(),
            'fp': fp.detach().cpu().tolist(),
            'fn': fn.detach().cpu().tolist(),
            'tn': tn.detach().cpu().tolist(),
        },
        'actual': {
            'pos': (tp + fn).detach().cpu().tolist(),
            'neg': (tn + fp).detach().cpu().tolist(),
        },
        'roc_raw': {
            'y_true': [np.concatenate(y_true_all[j]) for j in range(4)],
            'y_prob': [np.concatenate(y_prob_all[j]) for j in range(4)],
        },
        'scale_attention': {
            'real_mean': final_attR,
            'imag_mean': final_attI,
            'real_dist': final_attR_dist,
            'imag_dist': final_attI_dist,
        },
    }








def main():
    p=argparse.ArgumentParser()
    p.add_argument('--npz_path', required=True)
    p.add_argument('--test_npz_path', required=True)
    p.add_argument('--disorder_txt', required=True)
    p.add_argument('--protein_txt', required=True)
    p.add_argument('--rna_txt', required=True)
    p.add_argument('--dna_txt', required=True)
    p.add_argument('--test_disorder_txt', required=True)
    p.add_argument('--test_protein_txt', required=True)
    p.add_argument('--test_rna_txt', required=True)
    p.add_argument('--test_dna_txt', required=True)
    p.add_argument('--out_dir', default='./checkpoints')
    p.add_argument('--batch_size', type=int, default=3)
    p.add_argument('--max_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--max_len', type=int, default=0, help='optional crop length (0=no crop)')
    p.add_argument('--no_focal', action='store_true')
    p.add_argument('--no_dice', action='store_true')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--thresholds', type=str, default=None, help="클래스별 임계값. 예: 'disorder=0.4,protein=0.6,rna=0.35,dna=0.55' (없으면 전부 0.5)")
    p.add_argument('--alphas', type=str, default=None, help="클래스별 알파값. 예: 예: 'disorder=0.4,protein=0.6,rna=0.35,dna=0.55' (없으면 전부 0.5)")
    args = p.parse_args()

    class c: pass
    cfg = c()
    cfg.npz_path = args.npz_path
    cfg.test_npz_path = args.test_npz_path
    cfg.disorder_txt = args.disorder_txt
    cfg.protein_txt = args.protein_txt
    cfg.rna_txt = args.rna_txt
    cfg.dna_txt = args.dna_txt
    cfg.test_disorder_txt = args.test_disorder_txt
    cfg.test_protein_txt = args.test_protein_txt
    cfg.test_rna_txt = args.test_rna_txt
    cfg.test_dna_txt = args.test_dna_txt
    cfg.out_dir = args.out_dir
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.max_len = None if args.max_len in (0, None) else int(args.max_len)
    cfg.use_focal = not args.no_focal
    cfg.use_dice = not args.no_dice
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.thresholds = parse_thresholds(args.thresholds, default=0.5)
    cfg.alphas = parse_thresholds(args.alphas, default=0.5)
    log(f'Using thresholds: {cfg.thresholds}')
    

    # 실제 돌릴때는 지우기
    set_seed(args.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    # train file load
    npz = np.load(cfg.npz_path, allow_pickle=True)
    npz_ids = [str(x) for x in npz['ids']]

    # for testing dataset
    test_npz = np.load(cfg.test_npz_path, allow_pickle=True)
    test_npz_ids = [str(x) for x in test_npz['ids']]

    train_ids = make_ids(npz_ids)
    test_ids = make_ids(test_npz_ids)

    train_loader = build_loader(cfg, use_ids=train_ids)
    test_loader = build_loader(cfg, use_ids=test_ids, test_flag=1)

    # infer d from one batch
    sample = next(iter(train_loader))
    d = sample['emb'].shape[-1]

    model = IDRModel(d=d)
    model.to(cfg.device)

    tasks = ['disorder', 'protein', 'rna', 'dna']

    task_colors = {
        'disorder': '#1f77b4',
        'protein':  '#ff7f0e',
        'rna':      '#2ca02c',
        'dna':      '#d62728',
    }

    task_to_idx = {
        "disorder": 0,
        "protein":  1,
        "rna":      2,
        "dna":      3,
    }

    for task in tasks:
        # 1) task별 best model load
        best_path = os.path.join(cfg.out_dir, f"{task}_best.pt")
        print(f"[LOAD] {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=cfg.device))

        # 2) 해당 task의 segment 기준으로 scale attention 수집
        seg_lens, seg_atts = collect_segment_scale_attention(
            model,
            test_loader,
            task_idx=task_to_idx[task],
            use_real=False,
            device=cfg.device
        )

        if len(seg_lens) == 0:
            print(f"[WARN] No segments found for task {task}")
            continue

        # 3) segment length grouping
        groups, q1, q2 = assign_segment_groups(seg_lens)
        grouped = group_scale_attention(seg_atts, groups)

        print(f"[{task}] Segment length cut: Q33={q1:.2f}, Q66={q2:.2f}")

        # 4) task별로 다른 파일명으로 저장
        out_path = os.path.join(
            cfg.out_dir,
            f"segment_level_scale_attention_{task}.png"
        )

        plot_segment_scale_boxplot(
            grouped,
            scales=model.cwt.scales,
            out_path=out_path,
            title=f"Segment-level Scale Attention ({task})"
        )

        # metrics = evaluate(model, test_loader, cfg)
        # # dist = metrics['scale_attention']['imag_dist'][task]  # (N_pos, S)
        # dist = metrics['scale_attention']['real_dist'][task]  # (N_pos, S)

        # if dist.shape[0] == 0:
        #     continue

        # S = dist.shape[1]
        # for s in range(S):
        #     if s not in scale_data:
        #         scale_data[s] = {}
        #     scale_data[s][task] = dist[:, s]
    # # best.pt이걸 목적별로 고쳐야함
    # best_path = os.path.join(cfg.out_dir, "disorder_best.pt")
    # if os.path.exists(best_path):
    #     print(f"[LOAD] Loading pretrained model from {best_path}")
    #     model.load_state_dict(torch.load(best_path, map_location=cfg.device))
    # else:
    #     raise FileNotFoundError(f"No best.pt found in {cfg.out_dir}")

    # test_metrics = evaluate(model, test_loader, cfg)

    # scales = model.cwt.scales   # [1,2,3,4,6,8,16]
    # n_scales = len(scales)
    # n_tasks = len(tasks)

    # fig, ax = plt.subplots(figsize=(2.5 * n_scales, 6))

    # positions = []
    # data = []
    # colors = []

    # width = 0.18
    # offsets = np.linspace(-0.27, 0.27, n_tasks)

    # for s_idx, s_val in enumerate(scales):
    #     for t_idx, task in enumerate(tasks):
    #         if task not in scale_data[s_idx]:
    #             continue
    #         data.append(scale_data[s_idx][task])
    #         positions.append(s_idx + 1 + offsets[t_idx])
    #         colors.append(task_colors[task])

    # parts = ax.violinplot(
    #     data,
    #     positions=positions,
    #     widths=width,
    #     showmeans=False,
    #     showmedians=True,
    #     showextrema=False
    # )
    # parts = ax.boxplot(
    #     data,
    #     positions=positions,
    #     widths=width*0.4,
    #     showfliers=False,
    #     patch_artist=True
    # )

    # all_vals = np.concatenate(data)
    # ymin, ymax = np.percentile(all_vals, [5, 95])   # or [1, 99]
    # ax.set_ylim(ymin, ymax)

    # for body, color in zip(parts['bodies'], colors):
    #     body.set_facecolor(color)
    #     body.set_edgecolor('black')
    #     body.set_alpha(0.7)

    # for body, color in zip(parts['boxes'], colors):
    #     body.set_facecolor(color)
    #     body.set_edgecolor('black')
    #     body.set_alpha(0.7)
    # # (선택) median 선도 보기 좋게
    # for med in parts['medians']:
    #     med.set_color('black')
    #     med.set_linewidth(2.0)

    # ax.set_xticks(range(1, n_scales + 1))
    # ax.set_xticklabels(scales)
    # ax.set_xlabel("Scale")
    # ax.set_ylabel("Scale Attention Weight")
    # # ax.set_title("Scale-wise Attention Distribution (Imaginary part)")
    # ax.set_title("Scale-wise Attention Distribution (Real part)")

    # # legend
    # from matplotlib.patches import Patch
    # legend_patches = [
    #     Patch(color=task_colors[t], label=t) for t in tasks
    # ]
    # ax.legend(handles=legend_patches, title="Task")

    # plt.tight_layout()
    # plt.savefig(
    #     # os.path.join(cfg.out_dir, "scale_attention_violin_by_scale_imag.png"),
    #     os.path.join(cfg.out_dir, "scale_attention_violin_by_scale_real.png"),
    #     dpi=200
    # )
    # plt.show()

    # v1=test_metrics['scale_attention']['real_mean']['disorder']
    # v2=test_metrics['scale_attention']['imag_mean']['disorder']
    # v3=test_metrics['scale_attention']['real_mean']['protein']
    # v4=test_metrics['scale_attention']['imag_mean']['protein']
    # v5=test_metrics['scale_attention']['real_mean']['rna']
    # v6=test_metrics['scale_attention']['imag_mean']['rna']
    # v7=test_metrics['scale_attention']['real_mean']['dna']
    # v8=test_metrics['scale_attention']['imag_mean']['dna']
    # print(f'----> scale attention averaged (disorder): {v1}')
    # print(f'----> scale attention averaged (disorder): {v2}')
    # print(f'----> scale attention averaged (protein): {v3}')
    # print(f'----> scale attention averaged (protein): {v4}')
    # print(f'----> scale attention averaged (rna): {v5}')
    # print(f'----> scale attention averaged (rna): {v6}')
    # print(f'----> scale attention averaged (dna): {v7}')
    # print(f'----> scale attention averaged (dna): {v8}')



if __name__=='__main__':
    main()
