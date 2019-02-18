from textgenrnn import textgenrnn
from pathlib import Path
import datetime
import os
import sys
import random

default_temp = 0.3
default_pref = "The"
default_length_char = 200 

def load_models(model_file_stem):
    models = {}
    for character in os.listdir(model_file_stem):
        character_name = character
        model_folder = "%s/%s/%s" % (model_file_stem, character, character)
        # load in the model
        textgen = textgenrnn(weights_path='%s_weights.hdf5'%model_folder,
                             vocab_path='%s_vocab.json'%model_folder,
                             config_path='%s_config.json'%model_folder)
        models[character_name] = textgen
    return models

def generate(models, n, temp_arg):
    generated_text = ""
    characters = list(models)
    next_character = 0
    next_pref = default_pref
    temp = temp_arg
    gen_length = default_length_char

    header = "Temperature: %f\n\n" % temp
    generated_text += header
    
    # generate n pieces of dialogue
    for i in range(n):
        print("%d in %d, prefix: %s" % (i+1, n, next_pref)) 
        pref = next_pref
        character = characters[next_character]
        model = models[character]
        next_character += 1
        if next_character >= len(characters):
            next_character = 0
       
        
        generated_text += "%s: " % character
        
        # actually generate
        output = model.generate(n=1, temperature=temp, prefix=pref, return_as_list=True, max_gen_length=gen_length)
        
        # get prefix for next chunk
        words = output[0].split(" ")
        next_pref = words[len(words)-1:]
        next_pref = next_pref[0]
        for line in output:
            generated_text += line
        generated_text += "\n\n"
    return generated_text

def save_text(text, text_file_stem):
    if not Path(text_file_stem).is_dir():
        os.makedirs(text_file_stem)
    timestamp = datetime.datetime.now().strftime("%s")
    file_path = "%s/%s" % (text_file_stem, timestamp)
    with open(file_path, "w") as file:
        file.write(text)
    print("Generated text and wrote into %s" % file_path)
    
if __name__ == "__main__":
    category = sys.argv[1]
    num_characters = int(sys.argv[2])
    mode = int(sys.argv[3])
    num_extra = int(sys.argv[4])
    n = int(sys.argv[5])
    temp = float(sys.argv[6])

    model_file_stem = "trained_models/%s/%d_%d_%d" % (category, num_characters, mode, num_extra)
    print(model_file_stem)
    if not Path(model_file_stem).is_dir():
        print("Model has not been prepared for that combination of arguments")
        quit()
    
    # load in models for particular experiment as well as character names
    models = load_models(model_file_stem)
    
    # generate n pieces of dialogue
    # TODO: try to figure out how to seed or set a prefix or however textgenrnn does it
    generated_text = generate(models, n, temp)
    print(generated_text)

    # save the generated pieces of dialogue in corresponding folders with timestamps
    text_file_stem = "generated_text/%s/%d_%d_%d" % (category, num_characters, mode, num_extra)
    save_text(generated_text, text_file_stem)
