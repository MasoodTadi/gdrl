import os
import glob
import re
from collections import deque
import pandas as pd

# === SETTINGS ===
LOG_DIR = r"/home/tadim/gdrl/notebooks/chapter_12"   # <-- change if needed
PATTERN = "output_*.log"
OUTPUT_XLSX = os.path.join(LOG_DIR, "riv_drl_results.xlsx")

# Matches:  Estimated Rolling Intrinsic + Extrinsic Value: 3.4022
val_re = re.compile(r"Estimated Rolling Intrinsic \+ Extrinsic Value:\s*([+-]?\d+(?:\.\d+)?)")
# Extract numeric id from filename like output_123.log
id_re = re.compile(r"output_(\d+)\.log$", re.IGNORECASE)

rows = []

for path in sorted(glob.glob(os.path.join(LOG_DIR, PATTERN))):
    fname = os.path.basename(path)
    # Keep only the last two values we see while streaming the file
    last_two = deque(maxlen=2)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = val_re.search(line)
                if m:
                    last_two.append(float(m.group(1)))
    except Exception as e:
        rows.append({
            "Combination_ID": int(id_re.search(fname).group(1)) if id_re.search(fname) else None,
            "File_Name": fname,
            "RIV_Value": None,
            "DRL_Value": None,
            "Note": f"Read error: {e}"
        })
        continue

    # Interpret last_two:
    #   if two values found: [RIV, DRL] in order near the end
    #   if one value found: treat as RIV only (DRL missing)
    if len(last_two) == 2:
        riv_val, drl_val = last_two[0], last_two[1]
        note = ""
    elif len(last_two) == 1:
        riv_val, drl_val = last_two[0], None
        note = "Only one value found (RIV)."
    else:
        riv_val = drl_val = None
        note = "No matching lines found."

    rows.append({
        "Combination_ID": int(id_re.search(fname).group(1)) if id_re.search(fname) else None,
        "File_Name": fname,
        "RIV_Value": riv_val,
        "DRL_Value": drl_val,
        "Note": note
    })

# Build DataFrame and save
df = pd.DataFrame(rows)
# Sort by combination id (if present), then by filename for stability
df = df.sort_values(by=["Combination_ID","File_Name"], na_position="last")
df.to_excel(OUTPUT_XLSX, index=False)
print(f"✅ Processed {len(df)} files")
print(f"💾 Saved to: {OUTPUT_XLSX}")
