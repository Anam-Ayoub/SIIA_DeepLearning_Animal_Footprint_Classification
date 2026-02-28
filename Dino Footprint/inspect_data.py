import numpy as np
import openpyxl
import os

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Load names
names = np.load(os.path.join(data_path, 'names.npy'), allow_pickle=True)

# Load tracks
wb = openpyxl.load_workbook(os.path.join(data_path, 'tracks.xlsx'))
ws = wb.active

# Build mapping: normalize both sides to a common format
def normalize_key(s):
    """Normalize a string for matching: lowercase, strip .png, replace spaces with underscores"""
    s = str(s).lower().strip()
    s = s.replace('.png', '')
    # Don't replace spaces with underscores yet - let's try both ways
    return s

# Build id_to_group from tracks.xlsx
id_to_group = {}
id_to_group_normalized = {}
for row in ws.iter_rows(min_row=2):
    if row[0].value is not None:
        track_id = str(row[0].value).strip()
        group = str(row[1].value).strip() if row[1].value else 'Unknown'
        
        # Store with original key
        id_to_group[track_id.lower()] = group
        
        # Also store with normalized key (replace space with underscore)
        norm_key = track_id.lower().replace(' ', '_')
        id_to_group_normalized[norm_key] = group

# Try matching names
matched = 0
unmatched_names = []
labels = []

for name in names:
    name_str = str(name).strip()
    name_lower = name_str.lower()
    name_no_ext = name_lower.replace('.png', '')
    
    # Try direct match with underscore replacement
    if name_no_ext in id_to_group_normalized:
        labels.append(id_to_group_normalized[name_no_ext])
        matched += 1
    # Try matching name as-is (some names have spaces)
    elif name_no_ext in id_to_group:
        labels.append(id_to_group[name_no_ext])
        matched += 1
    # Try replacing underscores with spaces in the name
    elif name_no_ext.replace('_', ' ') in id_to_group:
        labels.append(id_to_group[name_no_ext.replace('_', ' ')])
        matched += 1
    else:
        unmatched_names.append(name_str)
        labels.append(None)

print(f'Total images: {len(names)}')
print(f'Total tracks entries: {len(id_to_group)}')
print(f'Matched: {matched}')
print(f'Unmatched: {len(unmatched_names)}')

if unmatched_names:
    print(f'\nFirst 20 unmatched names:')
    for n in unmatched_names[:20]:
        print(f'  "{n}"')
    
# Count label distribution for matched
from collections import Counter
label_counts = Counter([l for l in labels if l is not None])
print(f'\n=== LABEL DISTRIBUTION (matched only) ===')
for k, v in sorted(label_counts.items()):
    print(f'  {k}: {v}')
