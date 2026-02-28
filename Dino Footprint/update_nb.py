import json

nb_path = 'dino_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix 1: Update header markdown (cell 0)
nb['cells'][0]['source'][-1] = '**Classes (3 dinosaur groups):** Ornithopoda, Stegosauria, Theropoda'

# Fix 2: Update SELECTED_CLASSES in config cell (cell 3)
src = nb['cells'][3]['source']
for i, line in enumerate(src):
    if 'SELECTED_CLASSES' in line and '=' in line:
        src[i] = "SELECTED_CLASSES = ['Ornithopoda', 'Stegosauria', 'Theropoda']\n"
    if 'Select only the 5' in line:
        src[i] = '# Select only the 3 most relevant dinosaur groups\n'

# Fix 3: Update final markdown cell - change Dense(5) to Dense(3) and remove excluded classes
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        full = ''.join(cell['source'])
        if 'Dense(5, softmax)' in full:
            cell['source'] = [s.replace('Dense(5, softmax)', 'Dense(3, softmax)') for s in cell['source']]
            cell['source'] = [s for s in cell['source'] if 'Ankylosauria' not in s and 'Sauropoda' not in s]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print('Done! Updated to 3 classes: Ornithopoda, Stegosauria, Theropoda')
