import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from collections import defaultdict
from scipy import sparse
from scipy.sparse import save_npz, load_npz
from huggingface_hub import snapshot_download

from huggingface_hub import login

login(token="your_hf_token_here")



# Download only the npz_Baseline folder from the repo
snapshot_download(
    repo_id="IQSeC-Lab/LAMDA",
    repo_type="dataset",
    allow_patterns="NPZ_Version/npz_Baseline/**",
    local_dir="LAMDA_dataset"  # optional: where to save locally
)


# Spliting all the npz files into monthwise for further training and testing

input_root = "./LAMDA_dataset/NPZ_Version"  # Directory containing var_npz_* folders
output_root = "./LAMDA_dataset/NPZ_Version"  # Output root for monthwise splits
os.makedirs(output_root, exist_ok=True)

years = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
splits = ["train", "test"]

list_dir = ['npz_Baseline']

for var_dir in list_dir:
    var_path = os.path.join(input_root, var_dir)
    if not os.path.isdir(var_path):
        continue

    print(f"\nProcessing directory: {var_dir}")
    for split in splits:
        split_output_dir = os.path.join(output_root, f"{var_dir}_monthwise", split)
        os.makedirs(split_output_dir, exist_ok=True)
        summary_records = []
        skipped_samples = 0

        for year in years:
            try:
                X_file = os.path.join(var_path, f"{year}_X_{split}.npz")
                meta_file = os.path.join(var_path, f"{year}_meta_{split}.npz")
                if not (os.path.exists(X_file) and os.path.exists(meta_file)):
                    print(f"Missing files for {year} {split} in {var_dir}")
                    continue

                X = load_npz(X_file)
                meta = np.load(meta_file, allow_pickle=True)
                y = meta["y"]
                family = meta["family"]
                vt_count = meta["vt_count"]
                year_month = meta["year_month"]
                hashes = meta["hash"]

                # assert len(y) == X.shape[0] == len(family) == len(vt_count) == len(year_month)

                # Group by year_month
                ym_indices = defaultdict(list)
                for idx, ym in enumerate(year_month):
                    if ym != "unknown":
                        ym_indices[ym].append(idx)
                    else:
                        skipped_samples += 1

                # Save each group
                for ym, indices in ym_indices.items():
                    ym_X = X[indices]
                    ym_y = y[indices]
                    ym_family = family[indices]
                    ym_vt = vt_count[indices]
                    ym_ym = year_month[indices]
                    ym_hash = hashes[indices]

                    # Save files
                    save_npz(os.path.join(split_output_dir, f"{ym}_X_{split}.npz"), ym_X)
                    np.savez_compressed(
                        os.path.join(split_output_dir, f"{ym}_meta_{split}.npz"),
                        y=ym_y, family=ym_family, vt_count=ym_vt, year_month=ym_ym, hash=ym_hash
                    )

                    summary_records.append({
                        "year_month": ym,
                        "total": len(ym_y),
                        "malware": int(np.sum(ym_y == 1)),
                        "benign": int(np.sum(ym_y == 0))
                    })

                    print(f"Saved {var_dir} {split} {ym} - Total: {len(ym_y)}, Malware: {np.sum(ym_y==1)}, Benign: {np.sum(ym_y==0)}")

            except Exception as e:
                print(f"Skipping {year} {split} in {var_dir} due to error: {e}")

        # Save summary for this split
        summary_df = pd.DataFrame(summary_records)
        summary_df = summary_df.sort_values(by="year_month")
        summary_df.to_csv(os.path.join(split_output_dir, "year_month_split_summary.csv"), index=False)
        print(f"Saved summary to: {os.path.join(split_output_dir, 'year_month_split_summary.csv')}")

        # Log skipped
        if skipped_samples > 0:
            print(f"Skipped {skipped_samples} samples due to 'unknown' year_month in {var_dir} {split}.")
        else:
            print(f"No skipped samples in {var_dir} {split} — all entries had valid year_month.")
