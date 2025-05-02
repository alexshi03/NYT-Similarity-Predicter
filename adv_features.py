import pandas as pd
import numpy as np
import spacy
from typing import List, Dict, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import re

# main difference is using spaCy and TF-IDF for actual similarity of word to corpus
class AdvancedFeatureExtractor:
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_md")
            except:
                print("Installing spaCy model...")
                import os
                os.system("python -m spacy download en_core_web_md")
                self.nlp = spacy.load("en_core_web_md")
                
        # TF-IDF vectorizers for word and character n-grams
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True
        )
        
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=500,
            sublinear_tf=True
        )
        
        # scaler for numerical features
        self.scaler = StandardScaler()
        
    def extract_syntax_features(self, text: str) -> Dict[str, float]:
        if not self.use_spacy:
            raise ValueError("spaCy is required for syntax feature extraction")
            
        doc = self.nlp(text)
        
        if len(doc) == 0:
            return {
                'verb_noun_ratio': 0,
                'adj_adv_ratio': 0,
                'passive_voice_freq': 0,
                'question_freq': 0,
                'complex_sentence_ratio': 0,
                'compound_sentence_ratio': 0,
                'avg_dependency_depth': 0
            }
            
        # POS tags
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        # POS tag ratios
        verb_count = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
        noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        adj_count = pos_counts.get('ADJ', 0)
        adv_count = pos_counts.get('ADV', 0)
        
        verb_noun_ratio = verb_count / max(1, noun_count)
        adj_adv_ratio = adj_count / max(1, adv_count)
        
        # find passive voice constructions
        passive_count = 0
        for token in doc:
            if token.dep_ == 'auxpass' or token.dep_ == 'nsubjpass':
                passive_count += 1
                
        passive_voice_freq = passive_count / len(doc)
        
        # questions
        question_count = 0
        for sent in doc.sents:
            if sent.text.strip().endswith('?'):
                question_count += 1
                
        sentences = list(doc.sents)
        question_freq = question_count / max(1, len(sentences))
        
        # sentence structure
        complex_sentence_count = 0
        compound_sentence_count = 0
        dependency_depths = []
        
        for sent in sentences:
            # subordinating conjunctions = complex sentences
            if any(token.dep_ == 'mark' for token in sent):
                complex_sentence_count += 1
                
            # coordinating conjunctions = maybe compound sentences
            if any(token.dep_ == 'cc' for token in sent):
                compound_sentence_count += 1
                
            # 
            depths = {}
            
            # 
            for token in sent:
                if token.dep_ == 'ROOT':
                    depths[token.i] = 0
                else:
                    depth = 1
                    current = token
                    while current.head != current:
                        current = current.head
                        depth += 1
                    depths[token.i] = depth
                    
            max_depth = max(depths.values()) if depths else 0
            dependency_depths.append(max_depth)
            
        complex_sentence_ratio = complex_sentence_count / max(1, len(sentences))
        compound_sentence_ratio = compound_sentence_count / max(1, len(sentences))
        avg_dependency_depth = sum(dependency_depths) / max(1, len(dependency_depths))
        
        features = {
            'verb_noun_ratio': verb_noun_ratio,
            'adj_adv_ratio': adj_adv_ratio,
            'passive_voice_freq': passive_voice_freq,
            'question_freq': question_freq,
            'complex_sentence_ratio': complex_sentence_ratio,
            'compound_sentence_ratio': compound_sentence_ratio,
            'avg_dependency_depth': avg_dependency_depth
        }
        
        # POS tag frequencies
        for pos, count in pos_counts.items():
            features[f'pos_{pos}_freq'] = count / len(doc)
            
        # dependency relation frequencies
        dep_counts = {}
        for token in doc:
            dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1
            
        for dep, count in dep_counts.items():
            features[f'dep_{dep}_freq'] = count / len(doc)
            
        # entity type frequencies
        ent_counts = {}
        for ent in doc.ents:
            ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
            
        if doc.ents:
            for ent_type, count in ent_counts.items():
                features[f'ent_{ent_type}_freq'] = count / len(doc.ents)
        
        return features
    
    # grab word and char n-gram features via TF-IDR
    def extract_ngram_features(self, texts: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # fit word n-grams
        word_ngram_features = self.word_vectorizer.fit_transform(texts)
        word_ngram_df = pd.DataFrame(
            word_ngram_features.toarray(),
            columns=[f'word_ngram_{i}' for i in range(word_ngram_features.shape[1])]
        )
        
        # fit char n-grams
        char_ngram_features = self.char_vectorizer.fit_transform(texts)
        char_ngram_df = pd.DataFrame(
            char_ngram_features.toarray(),
            columns=[f'char_ngram_{i}' for i in range(char_ngram_features.shape[1])]
        )
        
        return word_ngram_df, char_ngram_df
    
    def extract_advanced_features_batch(self, texts: List[str], include_ngrams: bool = True) -> pd.DataFrame:

        all_features = []
        
        # syntax
        if self.use_spacy:
            for text in texts:
                syntax_features = self.extract_syntax_features(text)
                all_features.append(syntax_features)
                
            syntax_df = pd.DataFrame(all_features)
        else:
            syntax_df = pd.DataFrame()
            
        # n-gram
        if include_ngrams:
            word_ngram_df, char_ngram_df = self.extract_ngram_features(texts)
            
            if not syntax_df.empty:
                combined_df = pd.concat([syntax_df, word_ngram_df, char_ngram_df], axis=1)
            else:
                combined_df = pd.concat([word_ngram_df, char_ngram_df], axis=1)
        else:
            combined_df = syntax_df
            
        # rescale features
        if not combined_df.empty:
            scaled_features = self.scaler.fit_transform(combined_df)
            scaled_df = pd.DataFrame(scaled_features, columns=combined_df.columns)
        else:
            scaled_df = pd.DataFrame()
            
        return scaled_df
    
    # similarity features across author-article pairs
    def extract_pair_similarity_features(self, df: pd.DataFrame, text_col: str, pair_indices: pd.DataFrame) -> pd.DataFrame:

        from sklearn.metrics.pairwise import cosine_similarity
        
        # unique article indices
        unique_indices = set(pair_indices['article1_idx'].tolist() + pair_indices['article2_idx'].tolist())
        
        texts = {idx: df.loc[idx, text_col] for idx in unique_indices}
        
        text_list = [texts[idx] for idx in unique_indices]
        index_list = list(unique_indices)
        
        # basic n-gram features
        if len(text_list) > 0:
            # fit TF-IDF vectorizers on all texts
            word_features = self.word_vectorizer.fit_transform([texts[idx] for idx in index_list])
            char_features = self.char_vectorizer.fit_transform([texts[idx] for idx in index_list])
            
            # map from index to position in feature matrix
            idx_to_pos = {idx: i for i, idx in enumerate(index_list)}
            
            pair_features_list = []
            
            for _, row in pair_indices.iterrows():
                idx1 = row['article1_idx']
                idx2 = row['article2_idx']
                
                pos1 = idx_to_pos[idx1]
                pos2 = idx_to_pos[idx2]
                
                # cosine similarity between word vectors
                word_sim = cosine_similarity(
                    word_features[pos1:pos1+1], 
                    word_features[pos2:pos2+1]
                )[0][0]
                
                # cosine similarity between character vectors
                char_sim = cosine_similarity(
                    char_features[pos1:pos1+1], 
                    char_features[pos2:pos2+1]
                )[0][0]
                
                pair_features = {
                    'article1_idx': idx1,
                    'article2_idx': idx2,
                    'word_ngram_similarity': word_sim,
                    'char_ngram_similarity': char_sim
                }
                
                pair_features_list.append(pair_features)
                
            pair_features_df = pd.DataFrame(pair_features_list)
            
            return pair_features_df
        else:
            return pd.DataFrame(columns=['article1_idx', 'article2_idx', 'word_ngram_similarity', 'char_ngram_similarity'])
            
    # transform with fitted vectorizers
    def transform_new_texts(self, texts: List[str], include_ngrams: bool = True) -> pd.DataFrame:

        all_features = []
        
        if self.use_spacy:
            for text in texts:
                syntax_features = self.extract_syntax_features(text)
                all_features.append(syntax_features)
                
            syntax_df = pd.DataFrame(all_features)
        else:
            syntax_df = pd.DataFrame()
            
        # fitted n-gram vectorizers
        if include_ngrams:
            try:
                word_ngram_features = self.word_vectorizer.transform(texts)
                char_ngram_features = self.char_vectorizer.transform(texts)
                
                word_ngram_df = pd.DataFrame(
                    word_ngram_features.toarray(),
                    columns=[f'word_ngram_{i}' for i in range(word_ngram_features.shape[1])]
                )
                
                char_ngram_df = pd.DataFrame(
                    char_ngram_features.toarray(),
                    columns=[f'char_ngram_{i}' for i in range(char_ngram_features.shape[1])]
                )
                
                if not syntax_df.empty:
                    combined_df = pd.concat([syntax_df, word_ngram_df, char_ngram_df], axis=1)
                else:
                    combined_df = pd.concat([word_ngram_df, char_ngram_df], axis=1)
            except:
                print("Vectorizers have not been fitted. Using extract_ngram_features instead.")
                if not syntax_df.empty:
                    word_ngram_df, char_ngram_df = self.extract_ngram_features(texts)
                    combined_df = pd.concat([syntax_df, word_ngram_df, char_ngram_df], axis=1)
                else:
                    combined_df = pd.DataFrame()
        else:
            combined_df = syntax_df
            
        # transform with fitted scalers
        if not combined_df.empty:
            try:
                scaled_features = self.scaler.transform(combined_df)
                scaled_df = pd.DataFrame(scaled_features, columns=combined_df.columns)
            except:
                print("Scaler has not been fitted. Using fit_transform instead.")
                scaled_features = self.scaler.fit_transform(combined_df)
                scaled_df = pd.DataFrame(scaled_features, columns=combined_df.columns)
        else:
            scaled_df = pd.DataFrame()
            
        return scaled_df