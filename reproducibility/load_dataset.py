from datasets import load_dataset
import os
from tqdm import tqdm

# Load the dataset
def download_dataset(src: str, tgt: str, directory: str):
    os.makedirs(directory, exist_ok=True)

    dataset = load_dataset('haoranxu/WMT22-Test', f'{src}-{tgt}')
    with open(f'{directory}/src.txt', 'w') as src_file, open(f'{directory}/tgt.txt', 'w') as tgt_file:
        for entry in tqdm(dataset['test']):
            src_text = entry[f'{src}-{tgt}'][src]
            tgt_text = entry[f'{src}-{tgt}'][tgt]

            src_file.write(src_text + '\n')
            tgt_file.write(tgt_text + '\n')

LANG_MAP = {
    'en': 'english',
    'de': 'german',
    'cs': 'czech',
    'is': 'icelandic',
    'zh': 'chinese',
    'ru': 'russian',
}

# load english to other languages
for lang in list(LANG_MAP.keys())[1:]:
    download_dataset('en', lang, directory=f'data/from_english/{LANG_MAP[lang]}')

for lang in list(LANG_MAP.keys())[1:]:
    download_dataset(lang, 'en', directory=f'data/to_english/{LANG_MAP[lang]}')