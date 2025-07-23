import os
import requests

# Where to save
DICT_PATH = "frequency_dictionary_en_82_765.txt"

# Download URL from official symspellpy
URL = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"

if not os.path.exists(DICT_PATH):
    print(f"Downloading {DICT_PATH}...")
    response = requests.get(URL)
    response.raise_for_status()
    with open(DICT_PATH, "wb") as f:
        f.write(response.content)
    print(f"{DICT_PATH} downloaded and saved.")
else:
    print(f"{DICT_PATH} already exists.")
