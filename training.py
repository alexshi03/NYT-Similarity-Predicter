import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
import pickle
import traceback
import copy

from sklearn.linear_model import SGDClassifier

def is_numeric_array(arr):
    """Check if array contains only numeric values"""
    try:
        arr.astype(float)
        return True
    except:
        return False

def count_nan_values(arr):
    """Count NaN values in a safe way"""
    if not is_numeric_array(arr):
        nan_count = 0
        total_elements = 0

        for item in arr.flat:
            total_elements += 1
            try:
                if np.isnan(float(item)):
                    nan_count += 1
            except:
                pass

        return nan_count, total_elements
    else:
        # If it's a numeric array, use numpy's isnan
        nan_count = np.isnan(arr.astype(float)).sum()
        return nan_count, arr.size

def train_with_validation(X_train, y_train, X_val, y_val, max_epochs=100, learning_rate=0.01,
                         early_stopping=True, patience=5, verbose=True):
    """
    Train a model with explicit validation and loss tracking.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        max_epochs: Maximum number of training epochs
        learning_rate: Learning rate for gradient descent
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print progress

    Returns:
        Trained model and training history
    """
    # Ensure data is numeric
    try:
        print("Converting features to numeric values...")
        X_train = np.array(X_train, dtype=float)
        X_val = np.array(X_val, dtype=float)

        # Preprocessing: Impute missing values and scale features
        print("Preprocessing features...")
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_train_processed = imputer.fit_transform(X_train)
        X_train_processed = scaler.fit_transform(X_train_processed)

        X_val_processed = imputer.transform(X_val)
        X_val_processed = scaler.transform(X_val_processed)

        print("Initializing model...")
        model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.001,
                             learning_rate='constant', eta0=0.01, random_state=42)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        best_val_loss = float('inf')
        no_improvement_count = 0
        best_model = None

        print(f"Starting training for up to {max_epochs} epochs...")
        for epoch in range(max_epochs):
            try:
                model.partial_fit(X_train_processed, y_train, classes=np.unique(y_train))

                train_proba = model.predict_proba(X_train_processed)
                train_loss = log_loss(y_train, train_proba)
                train_pred = model.predict(X_train_processed)
                train_accuracy = accuracy_score(y_train, train_pred)

                val_proba = model.predict_proba(X_val_processed)
                val_loss = log_loss(y_val, val_proba)
                val_pred = model.predict(X_val_processed)
                val_accuracy = accuracy_score(y_val, val_pred)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{max_epochs} - "
                          f"train_loss: {train_loss:.4f} - train_acc: {train_accuracy:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_count += 1

                if early_stopping and no_improvement_count >= patience:
                    if verbose:
                        print(f"Early stopping after {epoch+1} epochs")
                    break
            except Exception as e:
                print(f"Error during epoch {epoch+1}: {e}")
                traceback.print_exc()
                break

        if verbose:
            print(f"\nFinal results:")
            if best_model is not None:
                val_pred = best_model.predict(X_val_processed)
                val_accuracy = accuracy_score(y_val, val_pred)
                print(f"Best validation accuracy: {val_accuracy:.4f}")
            else:
                print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")

        try:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Training Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')

            plt.tight_layout()
            plt.savefig('training_history.png')
        except Exception as e:
            print(f"Error plotting training history: {e}")

        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies,
            'epochs': epoch + 1
        }

        return best_model if best_model is not None else model, history

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return None, {'error': str(e)}


def use_iterative_training(data_path, output_dir='iterative_model_output'):
    """
    Complete pipeline using iterative training with validation.

    Args:
        data_path: Path to feature data CSV
        output_dir: Directory to save outputs
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)

        print(f"Dataset loaded: {len(data)} rows, {len(data.columns)} columns")

        exclude_cols = ['Author', 'Source', 'NYT', 'Genre', 'PubDate',
                       'Article Title', 'Article Text', 'URL', 'index']

        exclude_cols = [col for col in exclude_cols if col in data.columns]

        feature_cols = [col for col in data.columns if col not in exclude_cols]
        print(f"Using {len(feature_cols)} feature columns")

        print(f"Sample feature columns: {feature_cols[:5]}")

        print("\nSource distribution:")
        source_counts = data['Source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} articles ({count/len(data)*100:.1f}%)")

        print(f"\nCreating article pairs...")
        same_source_pairs = []
        for source, group in data.groupby('Source'):
            if len(group) >= 2:
                articles = group.index.tolist()
                for i in range(min(50, len(articles))):
                    idx1, idx2 = np.random.choice(articles, size=2, replace=False)
                    same_source_pairs.append((idx1, idx2, 1))

        diff_source_pairs = []
        sources = data['Source'].unique()
        if len(sources) >= 2:
            for i in range(len(same_source_pairs)):
                source1, source2 = np.random.choice(sources, size=2, replace=False)
                group1 = data[data['Source'] == source1]
                group2 = data[data['Source'] == source2]
                if len(group1) > 0 and len(group2) > 0:
                    idx1 = np.random.choice(group1.index)
                    idx2 = np.random.choice(group2.index)
                    diff_source_pairs.append((idx1, idx2, 0))

        all_pairs = same_source_pairs + diff_source_pairs
        np.random.shuffle(all_pairs)

        print(f"Created {len(all_pairs)} article pairs")
        print(f"  Same-source pairs: {len(same_source_pairs)}")
        print(f"  Different-source pairs: {len(diff_source_pairs)}")

        print("Extracting pair features...")
        X = []
        y = []

        for idx1, idx2, label in all_pairs:
            try:
                if idx1 < len(data) and idx2 < len(data):
                    # Convert feature values to numeric
                    features1 = pd.to_numeric(data.loc[idx1, feature_cols], errors='coerce').values
                    features2 = pd.to_numeric(data.loc[idx2, feature_cols], errors='coerce').values

                    diff_features = np.abs(features1 - features2)

                    pair_features = diff_features

                    X.append(pair_features)
                    y.append(label)
                else:
                    print(f"Warning: Invalid indices: {idx1}, {idx2}")
            except Exception as e:
                print(f"Error extracting features for pair ({idx1}, {idx2}): {e}")

        if len(X) == 0:
            print("No valid features extracted. Check your data.")
            return None, None, 0

        print(f"Extracted features for {len(X)} pairs")

        X = np.array(X)
        y = np.array(y)

        print(f"Feature array shape: {X.shape}")

        nan_count, total_elements = count_nan_values(X)
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in features ({nan_count/total_elements*100:.2f}%)")
            print("These will be imputed during training")

        print("Splitting data into train/val/test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")

        print("Training model with validation...")
        model, history = train_with_validation(
            X_train, y_train, X_val, y_val,
            max_epochs=100,
            learning_rate=0.001,
            early_stopping=False,
            patience=10,
            verbose=True
        )

        if model is None:
            print("Training failed. Check the error messages above.")
            return None, history, 0

        print("\nEvaluating on test set...")
        try:
            X_test = np.array(X_test, dtype=float)

            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            X_train_processed = imputer.fit_transform(X_train.astype(float))
            X_train_processed = scaler.fit_transform(X_train_processed)

            X_test_processed = imputer.transform(X_test)
            X_test_processed = scaler.transform(X_test_processed)

            test_pred = model.predict(X_test_processed)
            test_accuracy = accuracy_score(y_test, test_pred)

            print(f"Test accuracy: {test_accuracy:.4f}")
        except Exception as e:
            print(f"Error during test evaluation: {e}")
            traceback.print_exc()
            test_accuracy = 0

        try:
            with open(f"{output_dir}/model.pkl", 'wb') as f:
                pickle.dump(model, f)

            with open(f"{output_dir}/history.pkl", 'wb') as f:
                pickle.dump(history, f)

            with open(f"{output_dir}/summary.txt", 'w') as f:
                f.write(f"Model training with validation\n")
                f.write(f"===============================\n\n")
                f.write(f"Dataset: {data_path}\n")
                f.write(f"Total articles: {len(data)}\n")
                f.write(f"Feature count: {len(feature_cols)}\n\n")
                f.write(f"Article pairs: {len(all_pairs)}\n")
                f.write(f"Same-source pairs: {len(same_source_pairs)}\n")
                f.write(f"Different-source pairs: {len(diff_source_pairs)}\n\n")

                if 'epochs' in history:
                    f.write(f"Training epochs: {history['epochs']}\n")
                    if 'train_accuracy' in history and len(history['train_accuracy']) > 0:
                        f.write(f"Final training accuracy: {history['train_accuracy'][-1]:.4f}\n")
                    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
                        f.write(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}\n")

                f.write(f"Test accuracy: {test_accuracy:.4f}\n")

            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error saving results: {e}")
            traceback.print_exc()

        return model, history, test_accuracy

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        traceback.print_exc()
        return None, {"error": str(e)}, 0

if __name__ == "__main__":
    print("Source Similarity Analysis - Iterative Training")
    print("=" * 50)

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        print("No data path provided.")
        data_path = input("Enter path to features CSV file: ")

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "iterative_model_output"
        print(f"Using default output directory: {output_dir}")

    print("\nStarting analysis...")
    model, history, test_accuracy = use_iterative_training(data_path, output_dir)

    if model is not None:
        print("\n" + "=" * 50)
        print("Analysis complete!")
        print(f"Final test accuracy: {test_accuracy:.4f}")
        print(f"Results saved to {output_dir}")

        if history and 'epochs' in history:
            print(f"Trained for {history['epochs']} epochs")
            print("Learning curves saved as 'training_history.png'")
    else:
        print("\n" + "=" * 50)
        print("Analysis failed. Please check the error messages above.")