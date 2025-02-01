## Overview
This project utilises NLP techniques and large language models (LLMs) to analyse the anime series. The goal is to produce a comprehensive dataset, extract themes, classify text, and generate a character network. The end result will be an intuitive web-based GUI created with Gradio.

## Project Features

**1. Web Scraping (crawler)**
   - Utilizes Scrapy to extract structured information from Narutopedia.
   - Builds a dataset containing essential details about Naruto characters, themes, and dialogues.

**2. Character Network (character_network)**
  - Uses spaCy's Named Entity Recognition (NER) to identify character mentions in dialogues and descriptions.
  - Constructs an interactive character network using NetworkX and PyViz, allowing visualization of relationships between characters.

**3. Text Classifier (text_classifier)**
  - Trains a model to classify dialogues into predefined categories.
  - Uses deep learning and transformer-based models to achieve high accuracy.

**4. Theme Extraction (theme_classifier)**
   - Implements Zero-shot classification with Hugging Face's transformers to identify key themes from dialogue and descriptions.
   - Helps in understanding the overarching topics present in different episodes or story arcs.
  
## Technologies Used
- Python (Primary programming language)
- Scrapy (Web scraping)
- spaCy (Named Entity Recognition)
- NetworkX & PyViz (Graph visualization)
- Hugging Face Transformers (Zero-shot classification, LLMs)
- TensorFlow/PyTorch (Text classification)

## Future Enhancements
  - Develop a character chatbot using LLMs for interactive conversations.
  - Expand web scraping to include more anime series.
  - Enhance visualization features for character relationships.
