import tensorflow as tf
import numpy as np
from instagram import InstagramScraper
import json
import pickle

class InstagramAccountTester:
    def __init__(self):
        # Load the trained model
        self.model = tf.keras.models.load_model('models/best_model.keras')
        self.scraper = InstagramScraper()
        # Load the scaler that was used during training
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

    def prepare_features(self, user_data):
        # Extract only the features used in training (10 features as per train3.py)
        features = np.array([
            user_data['user_follower_count'],
            user_data['user_following_count'],
            user_data['user_media_count'],
            int(user_data['user_has_profil_pic']),
            user_data['username_length'],
            user_data['username_digit_count'],
            int(user_data['username_has_number']),
            user_data['full_name_length'],
            int(user_data['full_name_has_number']),
            int(user_data['is_private'])
        ]).reshape(1, -1)
        
        # Scale the features using the saved scaler
        features = self.scaler.transform(features)
        
        return features

    def predict_account(self, username):
        # Get user data from Instagram
        user_data = self.scraper.get_user_data(username)
        
        if not user_data:
            return None, None
        
        # Prepare features for prediction
        features = self.prepare_features(user_data)
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)[0][0]
        
        return prediction, user_data

def main():
    print("Instagram Account Authenticity Checker (Version 3)")
    print("===============================================")
    
    tester = InstagramAccountTester()
    
    while True:
        url_or_username = input("\nEnter Instagram URL or username (or 'quit' to exit): ")
        if url_or_username.lower() == 'quit':
            break
        
        try:
            # Handle username input (without URL)
            if 'instagram.com' not in url_or_username:
                username = url_or_username.strip()
            else:
                # Extract username from URL
                # Remove any query parameters
                url = url_or_username.split('?')[0]
                # Remove trailing slashes and split
                parts = url.rstrip('/').split('/')
                # Get the last non-empty part
                username = next((part for part in reversed(parts) if part), '')
            
            if not username:
                print("\nInvalid username or URL")
                continue
            
            prediction, user_data = tester.predict_account(username)
            
            if prediction is None:
                print(f"\nCouldn't fetch data for @{username}")
                continue
            
            print("\n=== Relevant Account Features ===")
            relevant_data = {
                'Followers': user_data['user_follower_count'],
                'Following': user_data['user_following_count'],
                'Posts': user_data['user_media_count'],
                'Has Profile Picture': user_data['user_has_profil_pic'],
                'Username Length': user_data['username_length'],
                'Numbers in Username': user_data['username_digit_count'],
                'Has Numbers in Username': user_data['username_has_number'],
                'Full Name Length': user_data['full_name_length'],
                'Has Numbers in Full Name': user_data['full_name_has_number'],
                'Is Private': user_data['is_private']
            }
            print(json.dumps(relevant_data, indent=2))
            
            print("\n=== Prediction Results ===")
            probability = prediction * 100
            print(f"Probability of being fake: {probability:.2f}%")
            print(f"Probability of being real: {100 - probability:.2f}%")
            
            # Add confidence level indicator
            confidence = "High" if abs(probability - 50) > 30 else "Medium" if abs(probability - 50) > 15 else "Low"
            print(f"Confidence Level: {confidence}")
            
            print(f"Final verdict: {'FAKE' if probability > 50 else 'REAL'} account")
            
            # Add warning for low confidence predictions
            if confidence == "Low":
                print("\nNote: This prediction has low confidence. Please use additional verification methods.")
            
        except Exception as e:
            print(f"\nError analyzing account: {str(e)}")
            print("Please try again with a valid Instagram username or URL")
            
    print("\nThank you for using Instagram Account Authenticity Checker!")

if __name__ == "__main__":
    main() 