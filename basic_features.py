import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Union
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

class BasicFeatureExtractor:
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    # lexical features 
    def extract_lexical_features(self, text: str) -> Dict[str, float]:

        # tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # stopwords
        words_no_stop = [word for word in words if word not in self.stop_words]
        
        # empty text
        if not sentences or not words:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'lexical_diversity': 0,
                'sentence_count': 0,
                'word_count': 0,
                'unique_word_ratio': 0
            }
        
        # same initial metrics as processor PLUS unique word ratio to capture difference
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        lexical_diversity = len(set(words_no_stop)) / len(words_no_stop) if words_no_stop else 0
        unique_word_ratio = len(set(words)) / len(words)
        
        features = {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'sentence_count': len(sentences),
            'word_count': len(words),
            'unique_word_ratio': unique_word_ratio
        }
        
        return features
    
    # identify common text 
    def extract_function_word_features(self, text: str) -> Dict[str, float]:
        function_words = {
            'articles': ['a', 'an', 'the'],
            'prepositions': ['in', 'on', 'at', 'of', 'to', 'with', 'by', 'for', 'from', 'about'],
            'conjunctions': ['and', 'but', 'or', 'so', 'yet', 'nor', 'because', 'although', 'since', 'unless'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
            'auxiliary_verbs': ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had', 
                              'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could']
        }
        
        # tokenize
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        
        if total_words == 0:
            # empty texts
            features = {f"{category}_freq": 0 for category in function_words}
            for category, word_list in function_words.items():
                for word in word_list:
                    features[f"{word}_freq"] = 0
            return features
        
        word_counts = Counter(words)
        features = {}
        
        for category, word_list in function_words.items():
            category_count = sum(word_counts.get(word, 0) for word in word_list)
            features[f"{category}_freq"] = category_count / total_words
            
            for word in word_list:
                features[f"{word}_freq"] = word_counts.get(word, 0) / total_words
                
        return features
        
    # captures ease of reading features
    def extract_readability_features(self, text: str) -> Dict[str, float]:

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        total_sentences = len(sentences)
        
        if total_words == 0 or total_sentences == 0:
            return {
                'avg_words_per_sentence': 0,
                'avg_sentence_length_chars': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0
            }
        
        # approximate syllables count
        def count_syllables(word):
            word = word.lower()
            if len(word) <= 3:
                return 1
            
            # remove common suffixes
            if word.endswith('es') or word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('e'):
                word = word[:-1]
                
            # count vowel groups
            vowels = 'aeiouy'
            count = 0
            prev_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_vowel:
                        count += 1
                    prev_vowel = True
                else:
                    prev_vowel = False
                    
            return max(1, count)  # assume each word has >= 1 syllable
        
        total_syllables = sum(count_syllables(word) for word in words)
        total_chars = sum(len(word) for word in words)
        
        # readability metric
        avg_words_per_sentence = total_words / total_sentences
        avg_sentence_length_chars = total_chars / total_sentences
        
        # Flesch-Kincaid level test
        flesch_kincaid = 0.39 * avg_words_per_sentence + 11.8 * (total_syllables / total_words) - 15.59
        
        # Automated Readability Index
        ari = 4.71 * (total_chars / total_words) + 0.5 * avg_words_per_sentence - 21.43
        
        features = {
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_sentence_length_chars': avg_sentence_length_chars,
            'flesch_kincaid_grade': flesch_kincaid,
            'automated_readability_index': ari
        }
        
        return features
    
    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
  
        punctuation_marks = {
            'comma': ',',
            'period': '.',
            'question_mark': '?',
            'exclamation': '!',
            'semicolon': ';',
            'colon': ':',
            'dash': '-',
            'parenthesis_open': '(',
            'parenthesis_close': ')',
            'quote': '"',
            'apostrophe': "'"
        }
        
        total_chars = len(text)
        
        if total_chars == 0:
            return {f"{name}_freq": 0 for name in punctuation_marks}
            
        features = {}
        
        for name, mark in punctuation_marks.items():
            count = text.count(mark)
            features[f"{name}_freq"] = count / total_chars
            
        return features
    
    def extract_text_structure_features(self, text: str) -> Dict[str, float]:
 
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return {
                'paragraph_count': 0,
                'avg_paragraph_length': 0,
                'avg_paragraph_length_words': 0,
                'quote_frequency': 0,
                'avg_quote_length': 0
            }
            
        # counts # of paragraphs
        paragraph_count = len(paragraphs)
        avg_paragraph_length = sum(len(p) for p in paragraphs) / paragraph_count
        
        # counts # words in para
        paragraph_word_counts = [len(re.findall(r'\b\w+\b', p)) for p in paragraphs]
        avg_paragraph_length_words = sum(paragraph_word_counts) / paragraph_count
        
        # grabs quotations
        quotes = re.findall(r'"([^"]*)"', text)
        quote_count = len(quotes)
        
        total_words = len(re.findall(r'\b\w+\b', text))
        quote_frequency = quote_count / max(1, total_words)
        avg_quote_length = sum(len(q.split()) for q in quotes) / max(1, quote_count)
        
        features = {
            'paragraph_count': paragraph_count,
            'avg_paragraph_length': avg_paragraph_length,
            'avg_paragraph_length_words': avg_paragraph_length_words,
            'quote_frequency': quote_frequency,
            'avg_quote_length': avg_quote_length
        }
        
        return features
        
    # returns dict with all the above features
    def extract_all_basic_features(self, text: str) -> Dict[str, float]:

        lexical_features = self.extract_lexical_features(text)
        function_word_features = self.extract_function_word_features(text)
        readability_features = self.extract_readability_features(text)
        punctuation_features = self.extract_punctuation_features(text)
        structure_features = self.extract_text_structure_features(text)
        
        combined_features = {
            **lexical_features,
            **function_word_features,
            **readability_features,
            **punctuation_features,
            **structure_features
        }
        
        return combined_features
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        all_features = []
        
        for text in texts:
            features = self.extract_all_basic_features(text)
            all_features.append(features)
            
        return pd.DataFrame(all_features)