import os
import glob
import re
import pandas as pd

# === SETTINGS ===
LOG_DIR = r"/home/tadim/gdrl/notebooks/chapter_12"
OUTPUT_XLSX = os.path.join(LOG_DIR, "drl_results.xlsx")

# Regex to match: Estimated Rolling Intrinsic + Extrinsic Value: 3.4022
pattern = re.compile(r"Estimated Rolling Intrinsic \+ Extrinsic Value:\s*([+-]?\d+(?:\.\d+)?)")

data = []

# Go through all output_hyper_*.log files in the folder
for filepath in sorted(glob.glob(os.path.join(LOG_DIR, "output_hyper_*.log"))):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    matches = pattern.findall(content)

    # Extract RIV (first) and DRL (last)
    riv_val = float(matches[0]) if len(matches) >= 1 else None
    drl_val = float(matches[-1]) if len(matches) >= 2 else None

    # Extract combination number from filename, e.g., output_hyper_37.log -> 37
    combo_id = int(os.path.splitext(os.path.basename(filepath))[0].split("_")[-1])

    data.append({
        "Combination_ID": combo_id,
        "File_Name": os.path.basename(filepath),
        "RIV_Value": riv_val,
        "DRL_Value": drl_val
    })

# Create DataFrame and save to Excel
df = pd.DataFrame(data).sort_values("Combination_ID")
df.to_excel(OUTPUT_XLSX, index=False)

print(f" Extracted {len(df)} log files.")
print(f" Results saved to: {OUTPUT_XLSX}")
