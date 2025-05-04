import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from datetime import datetime
from typing import List, Dict, Union, Tuple

nltk.download('punkt')
nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_md")
except:
    import os
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class ArticleDataProcessor:

    # update with csv path
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.data = None
        self.stop_words = set(stopwords.words('english'))
        
    # return df with loaded article data
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("No data path provided")
            
        self.data = pd.read_csv(self.data_path)
        
        # Check CSV columns
        print(f"CSV columns: {self.data.columns.tolist()}")
        
        # update with csv column titles, NYT being a boolean
        required_columns = [
            'Author', 'Source', 'NYT', 'Article Title', 'Article Text'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # handle date parsing with error handling
        if 'PubDate' in self.data.columns:
            try:
                self.data['PubDate'] = pd.to_datetime(self.data['PubDate'], errors='coerce')
                print(f"Date conversion successful with {self.data['PubDate'].isna().sum()} missing dates")
            except Exception as e:
                print(f"Warning: Could not parse dates: {e}")
                print("Continuing without date conversion")
        
        # check if alr boolean
        try:
            self.data['NYT'] = self.data['NYT'].astype(bool)
        except Exception as e:
            print(f"Warning: Could not convert NYT column to boolean: {e}")
            print("Continuing with NYT as original type")
        
        return self.data
    
    # preprocessed text to remove and case stop words; can consider lemmatization and tokenization
    def preprocess_text(self, text: str) -> str:

        # handle None or NaN values
        if pd.isna(text) or text is None:
            return ""
            
        # convert to string if it's not already
        text = str(text).lower()
        
        # remove URLs, special chars, and extra white space
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # TO BE SHARPENED: FEATURE EXTRACTION
    def extract_basic_features(self, text: str) -> Dict[str, float]:
        # Handle None or empty text
        if not text:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'lexical_diversity': 0,
                'sentence_count': 0,
                'word_count': 0,
            }

        # tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # cut stopwords
        words_no_stop = [word for word in words if word not in self.stop_words]
        
        # basic metrics for comparison; would run stats here
        features = {
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'lexical_diversity': len(set(words_no_stop)) / len(words_no_stop) if words_no_stop else 0,
            'sentence_count': len(sentences),
            'word_count': len(words),
        }
        
        return features
    
    # TO BE SHARPENED: STYLISTIC ELEMENTS
    def extract_stylistic_features(self, text: str) -> Dict[str, float]:
        # Handle None or empty text
        if not text or len(text.strip()) == 0:
            return {'avg_sentence_complexity': 0}

        try:
            # Limit text length to avoid spaCy processing errors
            max_len = 1000000  # 1 million chars
            if len(text) > max_len:
                text = text[:max_len]
                
            doc = nlp(text)
            
            # Check if document is empty
            if len(doc) == 0:
                return {'avg_sentence_complexity': 0}
                
            # POS tag frequencies
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                
            total_tokens = len(doc)
            pos_features = {f'pos_freq_{pos}': count / total_tokens 
                             for pos, count in pos_counts.items()}
            
            # dependency relation frequencies
            dep_counts = {}
            for token in doc:
                dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
                
            dep_features = {f'dep_freq_{dep}': count / total_tokens 
                             for dep, count in dep_counts.items()}
            
            # entity type frequencies
            entity_counts = {}
            for ent in doc.ents:
                entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
                
            entity_features = {f'ent_freq_{ent}': count / len(doc.ents) if len(doc.ents) > 0 else 0
                               for ent, count in entity_counts.items()}
            
            stylistic_features = {**pos_features, **dep_features, **entity_features}
            
            # derived features
            sentences = [sent for sent in doc.sents]
            stylistic_features['avg_sentence_complexity'] = sum(len(list(sent.noun_chunks)) for sent in sentences) / len(sentences) if sentences else 0
            
            return stylistic_features
        except Exception as e:
            print(f"Error extracting stylistic features: {e}")
            return {'avg_sentence_complexity': 0}
    
    # returns df with original data and extracted features; determine whether to preprocess or not
    def process_articles(self, preprocess: bool = True) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # copy of data to avoid modifying the original
        processed_data = self.data.copy()
        
        # preprocessed text column if requested
        if preprocess:
            processed_data['preprocessed_text'] = processed_data['Article Text'].apply(self.preprocess_text)
            text_column = 'preprocessed_text'
        else:
            text_column = 'Article Text'
            
        # Extract features for each article
        basic_features_list = []
        stylistic_features_list = []
        
        for text in processed_data[text_column]:
            basic_features = self.extract_basic_features(text)
            basic_features_list.append(basic_features)
            
            stylistic_features = self.extract_stylistic_features(text)
            stylistic_features_list.append(stylistic_features)
            
        basic_features_df = pd.DataFrame(basic_features_list)
        stylistic_features_df = pd.DataFrame(stylistic_features_list)
        
        result = pd.concat([
            processed_data, 
            basic_features_df, 
            stylistic_features_df
        ], axis=1)
        
        return result
    
    def create_author_pairs(self, n_same_author_pairs: int = None, n_diff_author_pairs: int = None) -> pd.DataFrame:
        """
        Create pairs of articles for training the author similarity model.
        
        Args:
            n_same_author_pairs: Number of same-author pairs to create (None for all possible)
            n_diff_author_pairs: Number of different-author pairs to create (None for equal to same-author)
            
        Returns:
            DataFrame containing article pairs with same_author label
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        pairs = []
        
        # group articles by author
        author_groups = self.data.groupby('Author')
        
        # create same-author pairs
        same_author_pairs = []
        for author, group in author_groups:
            if len(group) < 2:
                continue
                
            articles = group.index.tolist()
            for i in range(len(articles)):
                for j in range(i+1, len(articles)):
                    same_author_pairs.append((articles[i], articles[j], 1))
                    
        # subsample author pairs if requested
        if n_same_author_pairs and n_same_author_pairs < len(same_author_pairs):
            same_author_pairs = np.random.choice(
                same_author_pairs, size=n_same_author_pairs, replace=False).tolist()
                
        # number of different-author pairs to create
        if n_diff_author_pairs is None:
            n_diff_author_pairs = len(same_author_pairs)
            
        # create different-author pairs
        all_authors = list(author_groups.groups.keys())
        diff_author_pairs = []
        
        while len(diff_author_pairs) < n_diff_author_pairs:
            author1, author2 = np.random.choice(all_authors, size=2, replace=False)
            
            # randomly sample authors
            article1 = np.random.choice(author_groups.get_group(author1).index)
            article2 = np.random.choice(author_groups.get_group(author2).index)
            
            diff_author_pairs.append((article1, article2, 0))
            
        all_pairs = same_author_pairs + diff_author_pairs
        np.random.shuffle(all_pairs)
        
        pairs_df = pd.DataFrame(all_pairs, columns=['article1_idx', 'article2_idx', 'same_author'])
        
        # grab article details
        pairs_df['article1_title'] = pairs_df['article1_idx'].apply(lambda idx: self.data.loc[idx, 'Article Title'])
        pairs_df['article2_title'] = pairs_df['article2_idx'].apply(lambda idx: self.data.loc[idx, 'Article Title'])
        pairs_df['article1_author'] = pairs_df['article1_idx'].apply(lambda idx: self.data.loc[idx, 'Author'])
        pairs_df['article2_author'] = pairs_df['article2_idx'].apply(lambda idx: self.data.loc[idx, 'Author'])
        
        return pairs_df
    
    # create train, validation, and test sets
    def split_data(self, pairs_df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # randomize the data
        pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # calculate number of samples for each split
        n_samples = len(pairs_df)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # split the data
        train_df = pairs_df.iloc[:n_train]
        val_df = pairs_df.iloc[n_train:n_train+n_val]
        test_df = pairs_df.iloc[n_train+n_val:]
        
        return train_df, val_df, test_df