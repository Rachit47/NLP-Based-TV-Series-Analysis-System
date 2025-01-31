import spacy
import os
import sys
import pathlib
import pandas as pd
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from ast import literal_eval
from utils import load_subtitles_dataset
from nltk.tokenize import sent_tokenize

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()
        pass
    
    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp
    
    def get_ners_inference(self,script):
        script_sentences = sent_tokenize(script)
        ner_output = []
    
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set() # to avoid duplicate character names from a sentence
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    first_name = entity.text.split(" ")[0]
                    first_name = first_name.strip()
                    ners.add(first_name)
            ner_output.append(ners)
        # Now put the names from all the sentences into a list
        
        return ner_output

    def get_ners(self,dataset_path, save_path=None):
        
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        
        # loading the dataset
        df = load_subtitles_dataset(dataset_path)
        
        # Run inference
        df['ners'] = df['script'].apply(self.get_ners_inference)
        
        if save_path is not None:
            df.to_csv(save_path, index=False)
            
        return df
            
    