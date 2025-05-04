import pandas as pd
import numpy as np
import os
import argparse
from processor import ArticleDataProcessor
from basic_feature_extraction import BasicFeatureExtractor
from advanced_feature_extraction import AdvancedFeatureExtractor

def main(csv_path, output_dir='features_output'):

    print(f"Loading data from {csv_path}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    processor = ArticleDataProcessor(csv_path)
    basic_extractor = BasicFeatureExtractor()
    advanced_extractor = AdvancedFeatureExtractor(use_spacy=True)
    
    data = processor.load_data()
    print(f"Loaded {len(data)} articles")
    
    article_texts = data['Article Text'].tolist()
    
    print("Preprocessing texts...")
    processed_texts = [processor.preprocess_text(text) for text in article_texts]
    
    print("Extracting basic features...")
    basic_features_df = basic_extractor.extract_features_batch(processed_texts)
    basic_features_df.to_csv(f"{output_dir}/basic_features.csv", index=False)
    print(f"Saved basic features ({basic_features_df.shape[1]} features)")
    
    print("Extracting advanced features...")
    advanced_features_df = advanced_extractor.extract_advanced_features_batch(
        processed_texts, include_ngrams=True
    )
    advanced_features_df.to_csv(f"{output_dir}/advanced_features.csv", index=False)
    print(f"Saved advanced features ({advanced_features_df.shape[1]} features)")
    
    print("Combining data and features...")
    data_reset = data.reset_index()
    
    basic_features_df['index'] = data_reset.index
    advanced_features_df['index'] = data_reset.index
    
    combined_basic = pd.merge(data_reset, basic_features_df, on='index')
    if not advanced_features_df.empty:
        combined_all = pd.merge(combined_basic, advanced_features_df, on='index')
    else:
        combined_all = combined_basic
        
    combined_all = combined_all.drop('index', axis=1)
    
    combined_all.to_csv(f"{output_dir}/articles_with_features.csv", index=False)
    print(f"Saved combined data with {combined_all.shape[1]} columns")
    
    with open(f"{output_dir}/feature_summary.txt", 'w') as f:
        f.write(f"Total articles: {len(data)}\n")
        f.write(f"Basic features: {basic_features_df.shape[1]}\n")
        f.write(f"Advanced features: {advanced_features_df.shape[1]}\n\n")
        
        f.write("Basic feature names:\n")
        for col in basic_features_df.columns:
            if col != 'index':
                f.write(f"- {col}\n")
                
        f.write("\nAdvanced feature names:\n")
        for col in advanced_features_df.columns:
            if col != 'index':
                f.write(f"- {col}\n")
                
    print(f"Feature extraction complete! Output saved to {output_dir}")
    
    return combined_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from articles in a CSV file')
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--output', default='features_output', help='Output directory')
    
    args = parser.parse_args()
    
    main(args.csv, args.output)