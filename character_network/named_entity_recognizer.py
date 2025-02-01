import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os 
import sys
import pathlib
folder_path = pathlib.Path().parent.resolve()
sys.path.append(os.path.join(folder_path, '../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()  # Ensure nlp_model is initialized properly

    def load_model(self):
        try:
            return spacy.load("en_core_web_trf")  # Load model correctly
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            return None  # Handle failure gracefully
    
    def get_ners_inference(self, script):
        if not self.nlp_model:
            raise ValueError("spaCy model not loaded properly.")
        
        script_sentences = sent_tokenize(script)  # Tokenize script into sentences
        ner_output = []
        
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)  # Process each sentence separately
            ners = set()  # Avoid duplicates in the same sentence
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0].strip()
                    ners.add(first_name)
            ner_output.append(ners)
        
        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        if save_path and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
        
        # Load dataset
        df = load_subtitles_dataset(dataset_path)
        
        # Run inference
        df['ners'] = df['script'].apply(self.get_ners_inference)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df
