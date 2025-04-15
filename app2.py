from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import xgboost as xgb
import numpy as np
import json
import pickle
import os
import glob
from instagram import InstagramScraper
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class InstagramAccountTester:
    def __init__(self):
        # Load TensorFlow model
        self.tf_model = tf.keras.models.load_model('models/best_model.keras')
        with open('models/scaler.pkl', 'rb') as f:
            self.tf_scaler = pickle.load(f)
        
        # Load XGBoost model
        model_dirs = glob.glob('models_xgb*')
        if not model_dirs:
            raise FileNotFoundError("No XGBoost model directory found. Please run train2.py first.")
        
        latest_model_dir = max(model_dirs, key=os.path.getctime)
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(f'{latest_model_dir}/xgb_model.json')
        with open(f'{latest_model_dir}/scaler.pkl', 'rb') as f:
            self.xgb_scaler = pickle.load(f)
        
        self.scraper = InstagramScraper()

    def prepare_features(self, user_data):
        # Extract features for both models
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
        
        # Scale features for both models
        tf_features = self.tf_scaler.transform(features)
        xgb_features = self.xgb_scaler.transform(features)
        xgb_features = xgb.DMatrix(xgb_features)
        
        return tf_features, xgb_features

    def predict_account(self, username):
        # Get user data from Instagram
        user_data = self.scraper.get_user_data(username)
        
        if not user_data:
            return None, None, None
        
        # Prepare features for both models
        tf_features, xgb_features = self.prepare_features(user_data)
        
        # Get predictions from both models
        tf_prediction = self.tf_model.predict(tf_features, verbose=0)[0][0]
        xgb_prediction = self.xgb_model.predict(xgb_features)[0]
        
        # Calculate confidence levels for both models
        tf_confidence = abs(tf_prediction - 0.5) * 2  # Convert to 0-1 range
        xgb_confidence = abs(xgb_prediction - 0.5) * 2
        
        # Weighted ensemble prediction
        total_confidence = tf_confidence + xgb_confidence
        tf_weight = tf_confidence / total_confidence
        xgb_weight = xgb_confidence / total_confidence
        
        ensemble_prediction = (tf_prediction * tf_weight + xgb_prediction * xgb_weight)
        
        return ensemble_prediction, tf_prediction, xgb_prediction, user_data

# Initialize the tester
tester = InstagramAccountTester()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'Please provide an Instagram profile URL'}), 400
        
        # Extract username from URL
        if 'instagram.com' not in url:
            username = url.strip()
        else:
            # Remove any query parameters
            url = url.split('?')[0]
            # Remove trailing slashes and split
            parts = url.rstrip('/').split('/')
            # Get the last non-empty part
            username = next((part for part in reversed(parts) if part), '')
        
        if not username:
            return jsonify({'error': 'Invalid username or URL'}), 400
        
        app.logger.info(f"Attempting to fetch data for username: {username}")
        # Get predictions from both models
        ensemble_prediction, tf_prediction, xgb_prediction, user_data = tester.predict_account(username)

        app.logger.debug(f"Raw user_data received for {username}: {json.dumps(user_data, indent=2)}")

        if ensemble_prediction is None or user_data is None:
            error_message = f"Couldn't fetch data for @{username}. The profile might be non-existent, or Instagram blocked the request."
            app.logger.warning(error_message)
            return jsonify({'error': error_message}), 404
        
        # Calculate probabilities
        fake_probability = float(ensemble_prediction * 100)  # Probability of being fake
        real_probability = float(100 - fake_probability)  # Probability of being real
        
        # Calculate individual model probabilities
        tf_fake_prob = float(tf_prediction * 100)
        xgb_fake_prob = float(xgb_prediction * 100)
        
        # Determine if account is fake based on ensemble prediction
        is_fake = fake_probability > real_probability
        
        # Calculate confidence levels
        def get_confidence_level(probability):
            if probability >= 80:
                return "Very High"
            elif probability >= 65:
                return "High"
            elif probability >= 50:
                return "Medium"
            elif probability >= 35:
                return "Low"
            else:
                return "Very Low"
        
        # Get confidence levels for both real and fake predictions
        fake_confidence = get_confidence_level(fake_probability)
        real_confidence = get_confidence_level(real_probability)
        
        # Analysis points with numeric values
        analysis_points = [
            f"Followers: {int(user_data['user_follower_count'])}",
            f"Following: {int(user_data['user_following_count'])}",
            f"Posts: {int(user_data['user_media_count'])}",
            f"Has Profile Picture: {'Yes' if bool(user_data['user_has_profil_pic']) else 'No'}",
            f"Username Length: {int(user_data['username_length'])}",
            f"Numbers in Username: {int(user_data['username_digit_count'])}",
            f"Has Numbers in Username: {'Yes' if bool(user_data['username_has_number']) else 'No'}",
            f"Full Name Length: {int(user_data['full_name_length'])}",
            f"Has Numbers in Full Name: {'Yes' if bool(user_data['full_name_has_number']) else 'No'}",
            f"Is Private: {'Yes' if bool(user_data['is_private']) else 'No'}"
        ]
        
        userData = tester.scraper.userData
        profile_image_url = str(userData.get('user_profile_pic', f'https://ui-avatars.com/api/?name={username}&size=300&background=random'))
        
        # Prepare response with model comparison
        response = {
            'profileImage': profile_image_url,
            'username': str(username),
            'bio': str(userData.get('user_biography', 'No bio available')),
            'isFake': is_fake,
            'realProbability': float(round(real_probability, 2)),
            'fakeProbability': float(round(fake_probability, 2)),
            'realConfidence': str(real_confidence),
            'fakeConfidence': str(fake_confidence),
            'analysisPoints': list(analysis_points),
            'modelComparison': {
                'tensorflow': {
                    'fakeProbability': float(round(tf_fake_prob, 2)),
                    'realProbability': float(round(100 - tf_fake_prob, 2))
                },
                'xgboost': {
                    'fakeProbability': float(round(xgb_fake_prob, 2)),
                    'realProbability': float(round(100 - xgb_fake_prob, 2))
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Unhandled exception during analysis: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 