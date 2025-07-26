import argparse, os, re, yaml, json
import torch
import pandas as pd
from pathlib import Path
from importlib import import_module

import ast, yaml
from pathlib import Path

def smart_read_metrics(path):
    """
    Robustly parse metrics/*.txt into a dict.

    Handles:
      • YAML / JSON blocks
      • Plain key: value  (or key = value)
      • Banners / comments lines are ignored
    """
    if path is None or not Path(path).exists():
        return {}

    raw = Path(path).read_text().strip()
    if not raw:
        return {}

    # ---------- 1) Try YAML / JSON first ----------------------------
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # ---------- 2) Fallback line-by-line ----------------------------
    data = {}
    for line in raw.splitlines():
        line = line.strip()
        if (
            not line                         # blank
            or line.startswith("#")          # comment
            or line.startswith("=")          # banner =====
        ):
            continue

        if ":" in line:
            key, val = line.split(":", 1)
        elif "=" in line:
            key, val = line.split("=", 1)
        else:
            continue                         # skip malformed

        key, val = key.strip(), val.strip()

        # Cast numbers / lists / dicts if possible
        try:
            val_cast = ast.literal_eval(val)
        except Exception:
            val_cast = val                  # leave as string

        data[key] = val_cast

    return data

def count_params(best_model_dir):
    """
    Return the number of *trainable* parameters stored in best_model/.
    Priority order:
        1. model_dict.pth  (state-dict only – light-weight, fastest)
        2. model.pth       (full model object)
        3. first *.pth     (fallback)
    """
    best_model_dir = Path(best_model_dir)

    # ---- choose which file to open ---------------------------------
    candidates = [
        best_model_dir / "model_dict.pth",
        best_model_dir / "model.pth"
    ]
    # grab any other *.pth if the expected names aren’t present
    candidates += sorted(best_model_dir.glob("*.pth"))

    fp = next((p for p in candidates if p.exists()), None)
    if fp is None:
        return None

    obj = torch.load(fp, map_location="cpu")

    # ---- extract the state-dict no matter how it was saved ---------
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        state_dict = obj                       # bare state-dict ✅
    elif isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]         # packaged dict
    else:
        try:                                   # full nn.Module
            state_dict = obj.state_dict()
        except AttributeError:
            return None                        # unknown format

    # ---- count parameters -----------------------------------------
    return sum(p.numel() for p in state_dict.values())


def parse_lr(name):
    m = re.search(r"lr([0-9.]+)", name)
    return float(m.group(1)) if m else None

def crawl(run_dir):
    rows = []
    run_dir = Path(run_dir)
    for trial_norm in run_dir.iterdir():                           # single_bump_minmax
        if not trial_norm.is_dir(): continue
        *trial_tokens, norm_type = trial_norm.name.split('_')
        trial_type = '_'.join(trial_tokens)
        for activation_dir in trial_norm.iterdir():               # leakyrelu
            if not activation_dir.is_dir(): continue
            activation = activation_dir.name
            for arch_dir in activation_dir.iterdir():             # 10x10
                if not arch_dir.is_dir(): continue
                architecture = arch_dir.name
                for job_dir in arch_dir.iterdir():                # FCN_single_…
                    if not job_dir.is_dir(): continue
                    job_name  = job_dir.name
                    model_type = job_name.split('_')[0]           # FCN / KAN
                    lr = parse_lr(job_name)
                    metrics_dir = job_dir / "metrics"
                    metrics_file = next(iter(metrics_dir.glob("*.txt")), None)  # first .txt found
                    metrics = smart_read_metrics(metrics_file)  # {} if None
                    best_model = job_dir / "best_model"
                    # --- parameter counting -----------------
                    param_cnt  = count_params(best_model)
                    # ----------------------------------------
                    row = {
                        "job_name"    : job_name,
                        "model_type"  : model_type,
                        "trial_type"  : trial_type,
                        "norm_type"   : norm_type,
                        "activation"  : activation,
                        "architecture": architecture,
                        "learning_rate": lr,
                        "param_count" : param_cnt,
                    }
                    row.update(metrics)          # append all metric columns
                    rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Top-level sweep directory (e.g. outputs/FCN_single_v2)")
    ap.add_argument("--out_csv", default=None,
                    help="Optional path for the summary CSV. "
                         "If omitted, saves run_dir/results.csv")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "results.csv"

    df = crawl(run_dir)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(df)} rows → {out_csv}")

if __name__ == "__main__":
    main()
