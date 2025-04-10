import requests
import time
import random
import json
from typing import Dict, Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

admin_username = os.getenv("IG_USERNAME")
admin_password = os.getenv("IG_PASSWORD")

class InstagramScraper:
    def __init__(self, session_id: str = None):
        self.base_url = "https://www.instagram.com"
        self.session = requests.Session()
        self.userData = {}
        
        # Try to login if credentials are provided
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = self._perform_login(admin_username, admin_password)
            
        # Set up default headers with session ID
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'X-IG-App-ID': '936619743392459',
            'X-Requested-With': 'XMLHttpRequest',
            'X-ASBD-ID': '198387',
            'Origin': 'https://www.instagram.com',
            'Referer': 'https://www.instagram.com/',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Connection': 'keep-alive',
            'cookie': f'sessionid={self.session_id}'
        }
        
        # Update session headers
        self.session.headers.update(self.headers)

    def _perform_login(self, username: str, password: str) -> str:
        """Internal method to handle Instagram login"""
        login_url = "https://www.instagram.com/accounts/login/ajax/"
        
        # Get CSRF token
        self.session.get("https://www.instagram.com/")
        csrf_token = self.session.cookies.get("csrftoken")
        
        headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.instagram.com/accounts/login/",
            "X-CSRFToken": csrf_token
        }
        self.session.headers.update(headers)

        payload = {
            "username": username,
            "enc_password": f"#PWD_INSTAGRAM_BROWSER:0:&:{password}",
            "queryParams": {},
            "optIntoOneTap": "false"
        }

        res = self.session.post(login_url, data=payload, allow_redirects=True)

        if res.status_code == 200 and res.json().get("authenticated"):
            session_id = self.session.cookies.get("sessionid")
            print(f"[✅] Logged in successfully as {username}")
            return session_id
        else:
            print(f"[❌] Login failed: {res.text}")
            return ""

    def get_user_data(self, username: str) -> Optional[Dict]:
        """
        Fetch user data using Instagram's GraphQL API
        """
        try:
            init_url = f"{self.base_url}/{username}/"
            init_response = self.session.get(
                url=init_url,
                timeout=10
            )
            
            if init_response.status_code != 200:
                raise Exception(f"Failed to initialize: HTTP {init_response.status_code}")

            # Update headers with new CSRF token and referer
            csrf_token = init_response.cookies.get('csrftoken', '')
            self.headers.update({
                'X-CSRFToken': csrf_token,
                'Referer': f'https://www.instagram.com/{username}/',
            })
            self.session.headers.update(self.headers)

            # Add delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))

            # Make GraphQL API request
            api_url = f"{self.base_url}/api/v1/users/web_profile_info/?username={username}"
            response = self.session.get(
                url=api_url,
                timeout=10
            )

            if response.status_code == 404:
                print(f"User {username} not found")
                return None
            elif response.status_code != 200:
                raise Exception(f"API request failed: HTTP {response.status_code}")

            data = response.json()
            if not data or 'data' not in data or 'user' not in data['data']:
                raise Exception("No user data in response")

            user_data = data['data']['user']

            # Process the data
            processed_data = {
                'username': username,
                'full_name': user_data.get('full_name', ''),
                'user_media_count': user_data.get('edge_owner_to_timeline_media', {}).get('count', 0),
                'user_follower_count': user_data.get('edge_followed_by', {}).get('count', 0),
                'user_following_count': user_data.get('edge_follow', {}).get('count', 0),
                'user_has_profil_pic': bool(user_data.get('profile_pic_url')),
                'user_profile_pic': user_data.get('profile_pic_url_hd'),
                'user_is_private': user_data.get('is_private', False),
                'user_biography': user_data.get('biography', ''),
                'user_biography_length': len(user_data.get('biography', '')),
                'username_length': len(username),
                'username_digit_count': sum(c.isdigit() for c in username),
                'username_has_number': any(c.isdigit() for c in username),
                'full_name_has_number': any(c.isdigit() for c in user_data.get('full_name', '')),
                'full_name_length': len(user_data.get('full_name', '')),
                'is_private': user_data.get('is_private', False),
                'is_joined_recently': user_data.get('is_joined_recently', False),
                'has_channel': user_data.get('has_channel', False),
                'is_business_account': user_data.get('is_business_account', False),
                'has_guides': user_data.get('has_guides', False),
                'has_external_url': bool(user_data.get('external_url'))
            }
            self.userData = processed_data
            return processed_data

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

def main():
    # Create scraper instance with automatic login
    scraper = InstagramScraper()
    
    print("Instagram Profile Data Extractor")
    print("================================")
    
    while True:
        username = input("\nEnter Instagram URL or username (or 'quit' to exit): ")
        if username.lower() == 'quit':
            break
        
        # Clean up input in case full URL was provided
        if 'instagram.com' in username:
            username = username.split('/')[-1]
            if not username:
                username = username.split('/')[-2]
            username = username.split('?')[0]  # Remove query parameters
            username = username.strip('/')  # Remove trailing slashes
        
        if not username:
            print("\nInvalid username")
            continue
            
        print(f"\nFetching data for @{username}...")
        
        if scraper.session_id:  # Check if we have a valid session
            user_data = scraper.get_user_data(username)
            
            if user_data:
                print("\nExtracted Instagram Data:")
                print(json.dumps(user_data, indent=2))
            else:
                print(f"\nCouldn't fetch data for @{username}")
        else:
            print("\nNot logged in. Please check your credentials.")
        
        time.sleep(random.uniform(1, 2))
                
    print("\nThank you for using Instagram Profile Data Extractor!")

if __name__ == "__main__":
    main()