import json
from autotagging import NeuralTagger


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    tagger = NeuralTagger(**config)
    tagger.run()