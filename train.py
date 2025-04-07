import tensorflow as tf
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
                row.get('userFollowerCount', row.get('follower_count', 0)),  # Try different possible column names
                row.get('userFollowingCount', row.get('following_count', 0)),
                row.get('userMediaCount', row.get('media_count', 0)),
                row.get('userHasProfilPic', row.get('has_profile_pic', 1)),
                row.get('usernameLength', len(str(row.get('username', '')))),
                row.get('usernameDigitCount', sum(c.isdigit() for c in str(row.get('username', '')))),
                row.get('username_has_number', int(any(c.isdigit() for c in str(row.get('username', ''))))),
                row.get('full_name_length', row.get('name_length', 0)),
                row.get('full_name_has_number', 0),  # Default if not available
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

def create_model(input_shape):
    with tf.device('/CPU:0'):
        inputs = tf.keras.Input(shape=(input_shape,))
        
        # Initial normalization
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        # First Layer - Feature extraction
        x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Second Hidden Layer
        x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Third Hidden Layer
        x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Output Layer with additional stability
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Simplified optimizer configuration with fixed learning rate
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Fixed initial learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5)
            ]
        )
        
        return model

def get_new_model_dir():
    # Find the next available model directory number
    i = 1
    while os.path.exists(f'models{i}'):
        i += 1
    model_dir = f'models{i}'
    os.makedirs(model_dir)
    print(f"Created new directory: {model_dir}")
    return model_dir

def plot_confusion_matrix(y_true, y_pred, model_dir):
    # Create both raw and normalized confusion matrices
    cm_raw = confusion_matrix(y_true, y_pred.round())
    cm_norm = confusion_matrix(y_true, y_pred.round(), normalize='true')
    
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
    report = classification_report(y_true, y_pred.round())
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(f'{model_dir}/classification_report.txt', 'w') as f:
        f.write(report)

def main():
    # Set memory growth to avoid CUDA errors
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
    
    # Create new model directory
    model_dir = get_new_model_dir()
    
    # Load and prepare the combined data
    X_train, X_test, y_train, y_test, scaler = load_combined_data()
    
    # Save the scaler in the new directory
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create the model
    model = create_model(X_train.shape[1])
    
    # Calculate class weights to handle imbalance
    total_samples = len(y_train)
    n_positive = np.sum(y_train == 1)
    n_negative = total_samples - n_positive
    
    class_weight = {
        0: (1 / n_negative) * (total_samples / 2),
        1: (1 / n_positive) * (total_samples / 2)
    }
    
    # Define callbacks with optimized parameters
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,  # Increased patience
            restore_best_weights=True,
            min_delta=0.001,
            mode='min'
        ),
        
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Gentler reduction
            patience=10,
            min_lr=1e-6,
            min_delta=0.001,
            mode='min',
            cooldown=3
        ),
        
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'{model_dir}/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    # Train with optimized parameters
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=256,  # Larger batch size for stability
        callbacks=callbacks,
        class_weight=class_weight,  # Add class weights
        verbose=1,
        shuffle=True
    )
    
    # Load the best model
    model = tf.keras.models.load_model(f'{model_dir}/best_model.keras')
    
    # Make predictions and plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, model_dir)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest metrics:")
    for name, value in zip(model.metrics_names, metrics):
        print(f"{name}: {value:.4f}")
    
    # Save the final model
    model.save(f'{model_dir}/instagram_model.keras')
    print(f"Model saved as '{model_dir}/instagram_model.keras'")
    
    # Save the training history
    with open(f'{model_dir}/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Training history saved as '{model_dir}/training_history.pkl'")
    
    # Plot and save training history
    metrics_to_plot = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    for metric in metrics_to_plot:
        if metric in history.history:  # Only plot available metrics
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.savefig(f'{model_dir}/training_{metric}.png')
            plt.close()
    
    # Save a summary of the run
    with open(f'{model_dir}/run_summary.txt', 'w') as f:
        f.write(f"Model Directory: {model_dir}\n")
        f.write(f"Final Test Metrics:\n")
        for name, value in zip(model.metrics_names, metrics):
            f.write(f"{name}: {value:.4f}\n")

if __name__ == "__main__":
    main() 