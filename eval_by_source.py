import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def create_simple_evaluation(data_path, output_dir='simple_evaluation'):
    """
    Create a simple evaluation of source differences without using the trained model.
    This bypasses feature compatibility issues by training a new simple model just for evaluation.
    
    Args:
        data_path: Path to the features CSV
        output_dir: Output directory for results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Data loaded: {len(data)} articles")
    
    # Check source distribution
    print("\nSource distribution:")
    source_counts = data['Source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} articles ({count/len(data)*100:.1f}%)")
    
    # Create test pairs
    print("\nCreating test pairs...")
    
    # Prepare data
    nyt_articles = data[data['Source'] == 'NYT']
    not_nyt_articles = data[data['Source'] == 'NotNYT']
    
    print(f"NYT articles: {len(nyt_articles)}")
    print(f"Non-NYT articles: {len(not_nyt_articles)}")
    
    # Create simple feature dataframe for visualization
    selected_features = ['avg_sentence_length', 'avg_word_length', 'lexical_diversity', 
                        'word_count', 'unique_word_ratio', 'articles_freq', 'prepositions_freq',
                        'conjunctions_freq', 'pronouns_freq', 'auxiliary_verbs_freq']
    
    # Only keep features that exist in the data
    selected_features = [f for f in selected_features if f in data.columns]
    
    if len(selected_features) < 3:
        print("Not enough basic features found. Using first 10 available features.")
        selected_features = data.columns[8:18]  # Skip metadata columns
    
    print(f"Using features: {selected_features}")
    
    # Create feature dataframe
    feature_df = pd.DataFrame()
    feature_df['Source'] = data['Source']
    for feature in selected_features:
        feature_df[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    # Fill NaN values with mean
    for feature in selected_features:
        feature_df[feature] = feature_df[feature].fillna(feature_df[feature].mean())
    
    # Create visualization for feature distributions
    print("\nCreating feature distribution visualizations...")
    
    for feature in selected_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Source', y=feature, data=feature_df)
        plt.title(f'Distribution of {feature} by Source')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_by_source.png")
        plt.close()
    
    # Create pair plots for feature relationships
    print("Creating feature relationship visualization...")
    
    # Limit to manageable number of features for pair plot
    plot_features = selected_features[:4]  # Use first 4 features to keep plot readable
    
    plt.figure(figsize=(12, 10))
    sns.pairplot(feature_df, hue='Source', vars=plot_features)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_relationships.png")
    plt.close()
    
    # Create simple classifier for source prediction
    print("\nTraining simple classifier for source prediction...")
    
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare data
    X = feature_df[selected_features].values
    y = (feature_df['Source'] == 'NYT').astype(int).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train simple model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['NotNYT', 'NYT']))
    
    # Feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature importance:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. {selected_features[idx]}: {importances[idx]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [selected_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    
    # Calculate mean feature values for each source
    print("\nMean feature values by source:")
    mean_values = feature_df.groupby('Source')[selected_features].mean()
    print(mean_values)
    
    # Save statistics to file
    with open(f"{output_dir}/statistics.txt", 'w') as f:
        f.write("NYT vs. Non-NYT Style Analysis\n")
        f.write("============================\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Total articles: {len(data)}\n")
        f.write(f"NYT articles: {len(nyt_articles)}\n")
        f.write(f"Non-NYT articles: {len(not_nyt_articles)}\n\n")
        
        f.write("Mean feature values by source:\n")
        f.write(mean_values.to_string())
        f.write("\n\n")
        
        f.write("Feature importance:\n")
        for i, idx in enumerate(indices):
            f.write(f"{i+1}. {selected_features[idx]}: {importances[idx]:.4f}\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['NotNYT', 'NYT']))
    
    print(f"\nResults saved to {output_dir}")
    print("Simple evaluation complete!")

if __name__ == "__main__":
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python simple_eval.py <data_path> [output_dir]")
        sys.exit(1)
        
    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "simple_evaluation"
    
    create_simple_evaluation(data_path, output_dir)