import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from processor import ArticleDataProcessor
from basic_features import BasicFeatureExtractor
from adv_features import AdvancedFeatureExtractor
from sklearn.preprocessing import StandardScaler

class FeatureCombiner:

    # if want to use spaCy and n-gram features
    def __init__(self, use_spacy: bool = True, include_ngrams: bool = True):

        self.processor = ArticleDataProcessor()
        self.basic_extractor = BasicFeatureExtractor()
        self.advanced_extractor = AdvancedFeatureExtractor(use_spacy=use_spacy)
        self.use_spacy = use_spacy
        self.include_ngrams = include_ngrams
        self.scaler = StandardScaler()
        
    # grab all features for list of articles
    def extract_article_features(self, texts: List[str], preprocess: bool = True) -> pd.DataFrame:

        if preprocess:
            processed_texts = self.processor.preprocess_text(texts)
        else:
            processed_texts = texts
            
        basic_features_df = self.basic_extractor.extract_features_batch(processed_texts)
        
        advanced_features_df = self.advanced_extractor.extract_advanced_features_batch(
            processed_texts,
            include_ngrams=self.include_ngrams
        )
        
        if advanced_features_df.empty:
            combined_df = basic_features_df
        else:
            combined_df = pd.concat([basic_features_df, advanced_features_df], axis=1)
            
        return combined_df
