import pandas as pd
import numpy as np
import os
import argparse
import sys
import traceback
from tqdm import tqdm  # Progress bar

def main(csv_path, output_dir='features_output'):

    print(f"Loading data from {csv_path}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from processor import ArticleDataProcessor
        processor = ArticleDataProcessor(csv_path)
        
        try:
            print("Attempting to import BasicFeatureExtractor and AdvancedFeatureExtractor...")
            
            try:
                # First try with original names
                from basic_feature_extraction import BasicFeatureExtractor
                from advanced_feature_extraction import AdvancedFeatureExtractor
                print("Successfully imported from basic_feature_extraction.py and advanced_feature_extraction.py")
            except ImportError:
                # Try with alternative names
                try:
                    from basic_features import BasicFeatureExtractor
                    from adv_features import AdvancedFeatureExtractor
                    print("Successfully imported from basic_features.py and adv_features.py")
                except ImportError:
                    # List files to help debug
                    print("Available Python files in current directory:")
                    for file in [f for f in os.listdir() if f.endswith('.py')]:
                        print(f"  - {file}")
                    
                    raise Exception("Could not import feature extractors. Please check file names.")
        
        except Exception as e:
            print(f"Error importing feature extractors: {e}")
            print("Traceback:", traceback.format_exc())
            sys.exit(1)
        
        data = processor.load_data()
        print(f"Loaded {len(data)} articles")
        
        text_column = None
        for possible_col in ['Article Text', 'article_text', 'Text', 'text', 'Content', 'content']:
            if possible_col in data.columns:
                text_column = possible_col
                break
        
        if text_column is None:
            print("Could not find text column. Available columns:", data.columns.tolist())
            text_column = input("Please enter the name of the text column: ")
        
        print(f"Using column '{text_column}' for article texts")
        
        article_texts = data[text_column].tolist()
        
        print("Preprocessing texts...")
        processed_texts = []
        for text in tqdm(article_texts, desc="Preprocessing"):
            processed_texts.append(processor.preprocess_text(text))
        
        print("Extracting basic features...")
        basic_extractor = BasicFeatureExtractor()
        basic_features_df = basic_extractor.extract_features_batch(processed_texts)
        basic_features_df.to_csv(f"{output_dir}/basic_features.csv", index=False)
        print(f"Saved basic features ({basic_features_df.shape[1]} features)")
        
        print("Extracting advanced features...")
        advanced_extractor = AdvancedFeatureExtractor(use_spacy=True)
        
        batch_size = 100
        all_advanced_features = []
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Extracting advanced features"):
            batch_texts = processed_texts[i:i+batch_size]
            batch_features = advanced_extractor.extract_advanced_features_batch(
                batch_texts, include_ngrams=True
            )
            all_advanced_features.append(batch_features)
        
        if all_advanced_features:
            advanced_features_df = pd.concat(all_advanced_features, ignore_index=True)
            advanced_features_df.to_csv(f"{output_dir}/advanced_features.csv", index=False)
            print(f"Saved advanced features ({advanced_features_df.shape[1]} features)")
        else:
            advanced_features_df = pd.DataFrame()
            print("No advanced features were extracted.")
        
        print("Combining data and features...")
        
        data_reset = data.reset_index()
        
        basic_features_df['index'] = data_reset.index
        if not advanced_features_df.empty:
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
            f.write(f"Basic features: {basic_features_df.shape[1] - 1}\n")  # Subtract 1 for the index column
            if not advanced_features_df.empty:
                f.write(f"Advanced features: {advanced_features_df.shape[1] - 1}\n\n")  # Subtract 1 for the index column
            else:
                f.write("Advanced features: 0\n\n")
            
            f.write("Basic feature names:\n")
            for col in basic_features_df.columns:
                if col != 'index':
                    f.write(f"- {col}\n")
                    
            if not advanced_features_df.empty:
                f.write("\nAdvanced feature names:\n")
                for col in advanced_features_df.columns:
                    if col != 'index':
                        f.write(f"- {col}\n")
                
        print(f"Feature extraction complete! Output saved to {output_dir}")
        
        return combined_all
    
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from articles in a CSV file')
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--output', default='features_output', help='Output directory')
    
    args = parser.parse_args()
    
    main(args.csv, args.output)