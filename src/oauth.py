from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests
import streamlit as st

# Load OAuth credentials from Streamlit secrets
google_secrets = st.secrets["google_oauth"]
CLIENT_ID = google_secrets["client_id"]
CLIENT_SECRET = google_secrets["client_secret"]
AUTH_URI = google_secrets["auth_uri"]
TOKEN_URI = google_secrets["token_uri"]
AUTH_PROVIDER_CERT_URL = google_secrets["auth_provider_x509_cert_url"]
REDIRECT_URI = google_secrets["redirect_uri"]
SCOPES = google_secrets["scopes"]

def get_login_url():
     """Generate Google OAuth login URL dynamically from secrets."""
     flow = Flow.from_client_config(
         {
             "web": {
                 "client_id": CLIENT_ID,
                 "client_secret": CLIENT_SECRET,
                 "auth_uri": AUTH_URI,
                 "token_uri": TOKEN_URI,
                 "auth_provider_x509_cert_url": AUTH_PROVIDER_CERT_URL,
                 "redirect_uris": [REDIRECT_URI]
             }
         },
         scopes=SCOPES,
         redirect_uri=REDIRECT_URI
     )
     auth_url, _ = flow.authorization_url(prompt="consent")
     return auth_url
 
from google.auth.exceptions import RefreshError
from oauthlib.oauth2 import InvalidGrantError
 
def get_google_info(auth_code):
    """Exchange auth code for user info."""
    try:
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "auth_uri": AUTH_URI,
                    "token_uri": TOKEN_URI,
                    "auth_provider_x509_cert_url": AUTH_PROVIDER_CERT_URL,
                    "redirect_uris": [REDIRECT_URI]
                }
            },
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(code=auth_code)
        credentials = flow.credentials

        id_info = id_token.verify_oauth2_token(
            credentials.id_token, 
            requests.Request(), 
            clock_skew_in_seconds=2  # Allow 2 seconds of clock skew
        )

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