import collections
import os
import random
import datetime
import sys
import shutil
from pathlib import Path

# start running some experiments with solo or aggregate characters
# piece together a play structure
# select character, generate text of random length, select next character
# improvements: character selection, character generations, length logic

# returns dictionary with character name, file
def get_characters(root_path):
    character_texts = {}
    for play in root_path.iterdir():
        for character in play.iterdir():
            character_name = character.name.strip(".txt")
            with open(character, "r") as data:
                character_texts[character_name] = data.read()
    return character_texts

# mode: 0 = just text for the random character
#       1 = text of the random character + text from num_extra randomly chosen characters
#       2 = all dialogue text
def generate_source(character_texts, num_characters, mode, num_extra):
    character_output = {}
    for _ in range(num_characters):
        character_name =  list(character_texts)[random.randint(0, len(character_texts) - 1)]
        character_output[character_name] = character_texts[character_name]

        if mode == 1:
            for _ in range(num_extra):
                rand_source_character = character_name
                # to make sure the randomly sourced character is not the same
                while rand_source_character != character_name:
                    rand_source_character = list(character_texts)[random.randint(0, len(character_texts) - 1)]
                character_output[character_name] = character_output[character_name] + "\n" + character_texts[rand_source_character]

        if mode == 2:
            for c in character_texts:
                if c != character_name:
                    character_output[character_name] = character_output[character_name] + "\n" + character_texts[c]
                
    return character_output    

if __name__ == "__main__":
    character_path = Path("../corpus/data/strindberg/character/")
    character_texts = get_characters(character_path)
    
    # assemble dictionary
    num_characters = int(sys.argv[1])
    mode = int(sys.argv[2])
    num_extra = int(sys.argv[3])
    model_source = generate_source(character_texts, num_characters, mode, num_extra)
    for key in model_source:
        print("%s with %i chars" % (key, len(model_source[key])) )
    
    # save the source texts
    model_folder = "character_models_strindberg"
    model_subfolder = "%d_%d_%d" % (num_characters, mode, num_extra)
    file_stem = "%s/%s" % (model_folder, model_subfolder)
    if Path(file_stem).is_dir():
        shutil.rmtree(file_stem)
    os.makedirs(file_stem)
    for key in model_source:
        sub_stem = "%s/%s" % (file_stem, key)
        os.makedirs(sub_stem)
        with open("%s/words.txt" % sub_stem, "w") as file:
            file.write(model_source[key])
