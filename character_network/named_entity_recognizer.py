import spacy
import os
import sys
import pathlib
import pandas as pd
from ast import literal_eval
from utils import load_subtitles_dataset
from nltk.tokenize import sent_tokenize

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
        
        doc = self.nlp_model(script)  # Ensure nlp_model is callable
        ner_output = []
        
        for sentence in doc.sents:
            ners = set()  # Avoid duplicates in the same sentence
            for entity in sentence.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0].strip()
                    ners.add(first_name)
            ner_output.append(ners)
        
        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        if save_path and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: self.safe_literal_eval(x))
            return df
        
        df = load_subtitles_dataset(dataset_path)
        df['ners'] = df['script'].apply(self.get_ners_inference)
        
        if save_path:
            df.to_csv(save_path, index=False)
            
        return df

    def safe_literal_eval(self, value):
        try:
            return literal_eval(value) if isinstance(value, str) else value
        except (ValueError, SyntaxError):
            return []  
