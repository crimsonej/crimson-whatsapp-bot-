import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ProfileManager:
    def __init__(self, filename="user_profiles.json"):
        self.filename = filename
        self.profiles = {}
        self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.profiles = json.load(f)
            except Exception as e:
                logger.error(f"Error loading profiles: {e}")
                self.profiles = {}

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")

    def get_profile(self, user_id):
        """user_id is the phone number (string)"""
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                "phone": user_id,
                "name": None,
                "facts": [],           # list of strings
                "preferences": {},     # e.g., {"reply_length": "short"}
                "last_seen": None,
                "first_seen": datetime.now().isoformat()
            }
        return self.profiles[user_id]

    def update_profile(self, user_id, **kwargs):
        profile = self.get_profile(user_id)
        profile.update(kwargs)
        profile["last_seen"] = datetime.now().isoformat()
        self.save()

    def add_fact(self, user_id, fact):
        profile = self.get_profile(user_id)
        if fact not in profile["facts"]:
            profile["facts"].append(fact)
        self.save()

    def set_name(self, user_id, name):
        profile = self.get_profile(user_id)
        profile["name"] = name
        self.save()
