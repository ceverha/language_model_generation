import sys
import os
from textgenrnn import textgenrnn
from pathlib import Path

def load_text(file_stem):
    character_text = {}
    for character in os.listdir(file_stem):
        character_text[character] = "%s/%s/words.txt" % (file_stem, character)
        character_text[character] = os.path.abspath(character_text[character])
    return character_text

def train_char(character, text_file, batchsize):
    model_cfg = {
        'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
        'rnn_size': 128,   # number of LSTM cells of each layer (128/256 recommended)
        'rnn_layers': 3,   # number of LSTM layers (>=2 recommended)
        'rnn_bidirectional': True,   # consider text both forwards and backward, can give a training boost
        'max_length': 30,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
        'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
    }

    train_cfg = {
        'num_epochs': 40,   # set higher to train the model for longer
        'gen_epochs': 40,   # generates sample text from model after given number of epochs
        'train_size': 0.9,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
        'batch_size': batchsize,
    }

    textgen = textgenrnn(name=character)
    train_function = textgen.train_from_largetext_file
    train_function(
        file_path=text_file,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size=train_cfg['batch_size'],
        train_size=train_cfg['train_size'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=100,
        word_level=model_cfg['word_level'])
    
if __name__ == "__main__":
    category = sys.argv[1]
    num_characters = int(sys.argv[2])
    mode = int(sys.argv[3])
    num_extra = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    
    file_stem = "%s/%d_%d_%d" % (category, num_characters, mode, num_extra)
    if not Path(file_stem).is_dir():
        print("Text has not been prepared for that combination of arguments")
        quit()
    
    # load in characters and file names
    character_text = load_text(file_stem)
    print(character_text)
    
    # train the models for each character with settings based on the size of the text
    for character in character_text:
        model_path = "trained_models/%s/%s" % (file_stem, character)
        cwd = os.getcwd()
        # TODO 
        # need to remove the previous models in the folder
        if not Path(model_path).is_dir():
            os.makedirs(model_path)
        os.chdir(model_path)
        train_char(character, character_text[character], batch_size)
        os.chdir(cwd)
