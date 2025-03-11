import os
import streamlit as st

# Load api keys from streamlit secrets
def load_api_key(env_var_name):
    api_key = st.secrets[env_var_name]
    os.environ[env_var_name] = api_key