import os
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# Utility: run-length computation
# --------------------------------------------------
def get_run_lengths(binary_array):
    """
    binary_array: 1D np.array of 0/1
    return: list of lengths of consecutive 1s
    """
    runs = []
    run = 0
    for v in binary_array:
        if v == 1:
            run += 1
        else:
            if run > 0:
                runs.append(run)
                run = 0
    if run > 0:
        runs.append(run)
    return runs


# --------------------------------------------------
# Dataset (GT only, no model / no embedding usage)
# --------------------------------------------------
class GTDataset(Dataset):
    def __init__(self, npz_path, label_files):
        """
        npz_path: test_npz_path
        label_files: dict with keys ['disorder','protein','rna','dna']
        """
        self.npz = np.load(npz_path, allow_pickle=True)
        self.ids = list(map(str, self.npz['ids']))
        self.per_res = list(self.npz['per_residue'])  # only for length

        self.tasks = ['disorder', 'protein', 'rna', 'dna']
        self.labels = {t: self._load_label_txt(label_files[t]) for t in self.tasks}

        # keep only fully matched sequences
        self.valid_ids = []
        for i, sid in enumerate(self.ids):
            L = self.per_res[i].shape[0]
            ok = True
            for t in self.tasks:
                if sid not in self.labels[t] or len(self.labels[t][sid]) != L:
                    ok = False
                    break
            if ok:
                self.valid_ids.append(sid)

        print(f"[INFO] Loaded {len(self.valid_ids)} valid test sequences")

    def _load_label_txt(self, path):
        mp = {}
        with open(path, 'r') as f:
            for ln in f:
                if not ln.strip():
                    continue
                sid, payload = ln.strip().split('\t')
                mp[sid] = np.array([int(c) for c in payload], dtype=np.int64)
        return mp

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        sid = self.valid_ids[idx]
        y = np.stack([self.labels[t][sid] for t in self.tasks], axis=1)  # (L,4)
        return y


def collate_fn(batch):
    # batch: list of (L_i,4) arrays → keep variable length
    return batch


# --------------------------------------------------
# Main: compute GT run-length statistics
# --------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_npz_path', required=True)
    p.add_argument('--test_disorder_txt', required=True)
    p.add_argument('--test_protein_txt', required=True)
    p.add_argument('--test_rna_txt', required=True)
    p.add_argument('--test_dna_txt', required=True)
    args = p.parse_args()

    label_files = {
        'disorder': args.test_disorder_txt,
        'protein':  args.test_protein_txt,
        'rna':      args.test_rna_txt,
        'dna':      args.test_dna_txt,
    }

    ds = GTDataset(args.test_npz_path, label_files)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    tasks = ['disorder', 'protein', 'rna', 'dna']
    task_runs = {t: [] for t in tasks}

    for batch in loader:
        # batch[0]: (L,4)
        y = batch[0]
        for t_idx, task in enumerate(tasks):
            gt = (y[:, t_idx] == 1).astype(int)
            runs = get_run_lengths(gt)
            task_runs[task].extend(runs)

    print("\n=== GT contiguous segment length statistics (TEST set) ===")
    for task in tasks:
        runs = np.array(task_runs[task])
        if len(runs) == 0:
            print(f"{task:8s} | no positive segments")
            continue
        print(
            f"{task:8s} | "
            f"mean={runs.mean():.2f}, "
            f"median={np.median(runs):.2f}, "
            f"min={runs.min()}, "
            f"max={runs.max()}, "
            f"#segments={len(runs)}"
        )


if __name__ == "__main__":
    main()
