
import streamlit as st
import yaml
import streamlit_authenticator as stauth
import datetime
import json
import os
from pathlib import Path


class UserManager:
    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parents[2]
        config_path = BASE_DIR / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        
        if not self.config:
            raise ValueError("config.yaml is empty or invalid YAML")
        
        if "credentials" not in self.config:
            raise KeyError("Missing 'credentials' section in config.yaml")
        
        if "cookie" not in self.config:
            raise KeyError("Missing 'cookie' section in config.yaml")
        
        for key in ("name", "key", "expiry_days"):
            if key not in self.config["cookie"]:
                raise KeyError(f"Missing cookie.{key} in config.yaml")
        
        self.authenticator = stauth.Authenticate(
            self.config["credentials"],
            self.config["cookie"]["name"],
            self.config["cookie"]["key"],
            self.config["cookie"]["expiry_days"],
        )
    
    def login(self):
        self.authenticator.login(location="main")
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
        return name, authentication_status, username
    
    def logout(self):
        self.authenticator.logout(location="sidebar")
    
    def get_user_role(self, username):
        return self.config["credentials"].get("usernames", {}).get(username, {}).get("role", "viewer")
    
    def log_action(self, username, action, details=None):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "username": username,
            "action": action,
            "details": details or {},
        }
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/audit_log.json"
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    
    def get_audit_logs(self, username=None, limit=200):
        log_file = "logs/audit_log.json"
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            return []
        if username:
            logs = [l for l in logs if l["username"] == username]
        return logs[-limit:]
