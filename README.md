# Instagram Fake Profile Detector

A web application that analyzes Instagram profiles to determine if they are likely to be fake or genuine using machine learning.

## Features

- Analyze Instagram profiles by URL or username
- Real-time prediction using a trained machine learning model
- Detailed analysis of profile features
- Confidence level indicator
- Beautiful and responsive UI

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the following files in your project:
   - `models/best_model.keras` - The trained TensorFlow model
   - `models/scaler.pkl` - The scaler used during model training
   - `instagram.py` - The Instagram scraper module

### Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. Enter an Instagram profile URL or username and click "Analyze" to get the results.

## Project Structure

- `app.py` - Flask application with API endpoints
- `test.py` - Command-line version of the profile analyzer
- `instagram.py` - Instagram scraper module
- `templates/index.html` - HTML template for the web interface
- `static/script.js` - JavaScript for the web interface
- `models/` - Directory containing the trained model and scaler

## How It Works

The application uses a machine learning model trained on various Instagram profile features to predict whether a profile is likely to be fake or genuine. The model analyzes features such as:

- Follower and following counts
- Post count
- Profile picture presence
- Username characteristics
- Full name characteristics
- Account privacy status

The prediction is presented as a probability percentage, along with a confidence level indicator.

## License

[MIT License](LICENSE) 