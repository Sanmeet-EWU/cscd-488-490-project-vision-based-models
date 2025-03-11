from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests

# Google OAuth Credentials
CLIENT_SECRETS_FILE = "client_secret.json" 
SCOPES = ["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"]
REDIRECT_URI = "http://localhost:8501"

def get_login_url():
    """Generate Google OAuth login URL"""
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return auth_url

from google.auth.exceptions import RefreshError
from oauthlib.oauth2 import InvalidGrantError

def get_google_info(auth_code):
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials

        id_info = id_token.verify_oauth2_token(credentials.id_token, requests.Request())

        user_info = {
            "token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "id_token": credentials.id_token,
            "email": id_info.get("email", "N/A"),
            "name": id_info.get("name", "N/A"),
            "picture": id_info.get("picture", ""),
            "sub": id_info.get("sub"),  # "sub" is a globally unique identifier (UID) assigned to each Google user.
            "chat_history": [] # Initialize as none, will be fetched later
        }
        return user_info

    except InvalidGrantError:
        print("Invalid grant error: Token expired or already used.")
        return None  # Handle logout or re-authentication
    except RefreshError:
        print("Refresh token error: User needs to log in again.")
        return None  # Handle token refresh failure
    except ValueError as e:
        print(f"Error verifying ID token: {e}")
        return None
