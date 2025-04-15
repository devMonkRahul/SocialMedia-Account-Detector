import xgboost as xgb
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

def load_combined_data():
    # Load JSON datasets
    with open('data/fakeAccountData.json', 'r') as f:
        fake_data = json.load(f)
    
    with open('data/realAccountData.json', 'r') as f:
        real_data = json.load(f)
    
    # Load CSV dataset
    csv_data = pd.read_csv('data/final-v1.csv')
    
    # Print CSV columns to debug
    print("Available CSV columns:", csv_data.columns.tolist())
    
    # Convert JSON data to features arrays
    features = []
    labels = []
    
    # Process fake accounts from JSON
    for account in fake_data:
        features.append([
            account['userFollowerCount'],
            account['userFollowingCount'],
            account['userMediaCount'],
            account['userHasProfilPic'],
            account['usernameLength'],
            account['usernameDigitCount'],
            int(account['usernameDigitCount'] > 0),  # username_has_number
            account['usernameLength'],  # Using as full_name_length
            int(account['usernameDigitCount'] > 0),  # full_name_has_number
            account['userIsPrivate']
        ])
        labels.append(1)  # 1 for fake accounts
    
    # Process real accounts from JSON
    for account in real_data:
        features.append([
            account['userFollowerCount'],
            account['userFollowingCount'],
            account['userMediaCount'],
            account['userHasProfilPic'],
            account['usernameLength'],
            account['usernameDigitCount'],
            int(account['usernameDigitCount'] > 0),  # username_has_number
            account['usernameLength'],  # Using as full_name_length
            int(account['usernameDigitCount'] > 0),  # full_name_has_number
            account['userIsPrivate']
        ])
        labels.append(0)  # 0 for real accounts
    
    # Process CSV data
    try:
        # Try to map CSV columns to our feature set
        csv_features = []
        for _, row in csv_data.iterrows():
            csv_features.append([
                row.get('userFollowerCount', row.get('follower_count', 0)),
                row.get('userFollowingCount', row.get('following_count', 0)),
                row.get('userMediaCount', row.get('media_count', 0)),
                row.get('userHasProfilPic', row.get('has_profile_pic', 1)),
                row.get('usernameLength', len(str(row.get('username', '')))),
                row.get('usernameDigitCount', sum(c.isdigit() for c in str(row.get('username', '')))),
                row.get('username_has_number', int(any(c.isdigit() for c in str(row.get('username', ''))))),
                row.get('full_name_length', row.get('name_length', 0)),
                row.get('full_name_has_number', 0),
                row.get('userIsPrivate', row.get('is_private', 0))
            ])
        
        csv_features = np.array(csv_features)
        csv_labels = csv_data['is_fake'].values if 'is_fake' in csv_data.columns else []
        
        if len(csv_labels) > 0:
            # Combine all features and labels
            X = np.vstack([np.array(features), csv_features])
            y = np.concatenate([np.array(labels), csv_labels])
        else:
            # Use only JSON data if CSV processing fails
            X = np.array(features)
            y = np.array(labels)
    
    except Exception as e:
        print(f"Warning: Error processing CSV data: {e}")
        print("Proceeding with only JSON data...")
        X = np.array(features)
        y = np.array(labels)
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(y)}")
    print(f"Fake accounts: {np.sum(y == 1)}")
    print(f"Real accounts: {np.sum(y == 0)}")
    print(f"Number of features: {X.shape[1]}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_new_model_dir():
    # Create a timestamp-based directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models_xgb_{timestamp}'
    os.makedirs(model_dir)
    print(f"Created new directory: {model_dir}")
    return model_dir

def plot_confusion_matrix(y_true, y_pred, model_dir):
    # Create both raw and normalized confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Save confusion matrices as numpy arrays
    np.save(f'{model_dir}/confusion_matrix_raw.npy', cm_raw)
    np.save(f'{model_dir}/confusion_matrix_normalized.npy', cm_norm)
    
    # Save as CSV files for better readability
    pd.DataFrame(
        cm_raw,
        index=['Real', 'Fake'],
        columns=['Real', 'Fake']
    ).to_csv(f'{model_dir}/confusion_matrix_raw.csv')
    
    pd.DataFrame(
        cm_norm,
        index=['Real', 'Fake'],
        columns=['Real', 'Fake']
    ).to_csv(f'{model_dir}/confusion_matrix_normalized.csv')
    
    # Custom colormap (blue for real predictions, red for fake predictions)
    colors = ['#4B86B4', '#4B86B4', '#E14B60', '#E14B60']
    custom_cmap = sns.color_palette(colors, as_cmap=False)
    
    # Plot raw counts matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_raw,
                annot=True,
                fmt='d',
                cmap=custom_cmap,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                square=True,
                cbar=True,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Confusion Matrix (Raw Counts)', pad=20, fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/confusion_matrix_raw.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot normalized matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm,
                annot=True,
                fmt='.1%',
                cmap=custom_cmap,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                square=True,
                cbar=True,
                cbar_kws={'label': 'Percentage'},
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Confusion Matrix (Normalized)', pad=20, fontsize=14, weight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print and save classification report
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(f'{model_dir}/classification_report.txt', 'w') as f:
        f.write(report)

def main():
    # Create new model directory
    model_dir = get_new_model_dir()
    
    # Load and prepare the combined data
    X_train, X_test, y_train, y_test, scaler = load_combined_data()
    
    # Save the scaler in the new directory
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Calculate class weights
    total_samples = len(y_train)
    n_positive = np.sum(y_train == 1)
    n_negative = total_samples - n_positive
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'scale_pos_weight': n_negative / n_positive,  # Handle class imbalance
        'random_state': 42
    }
    
    # Train the model
    num_rounds = 200
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evallist,
        early_stopping_rounds=25,
        verbose_eval=10
    )
    
    # Save the model
    model.save_model(f'{model_dir}/xgb_model.json')
    print(f"Model saved as '{model_dir}/xgb_model.json'")
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_dir)
    
    # Calculate and print metrics
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Save a summary of the run
    with open(f'{model_dir}/run_summary.txt', 'w') as f:
        f.write(f"Model Directory: {model_dir}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("\nModel Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main() 