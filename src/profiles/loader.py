"""
SentinelVision — Environment Profile Loader

Loads environment-specific configuration (warehouse, construction, etc.)
"""

import os
import yaml
from typing import Optional
from loguru import logger


class ProfileLoader:
    """Loads environment profiles from YAML files."""

    def __init__(self, profiles_dir: str = "config/environments"):
        self.profiles_dir = profiles_dir
        self._profiles = {}
        self._scan()

    def _scan(self):
        """Scan for available profiles."""
        if not os.path.isdir(self.profiles_dir):
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith((".yaml", ".yml")):
                name = os.path.splitext(fname)[0]
                self._profiles[name] = os.path.join(self.profiles_dir, fname)
        logger.info(f"Found profiles: {list(self._profiles.keys())}")

    def load(self, name: str) -> Optional[dict]:
        """Load a profile by name."""
        path = self._profiles.get(name)
        if not path:
            logger.error(f"Profile '{name}' not found. Available: {self.available}")
            # Try direct file path if given
            if os.path.exists(name):
                path = name
            else:
                return None
        with open(path, 'r') as f:
            profile = yaml.safe_load(f)
        logger.info(f"Loaded profile: {profile.get('name', name)}")
        return profile

    @property
    def available(self) -> list:
        return list(self._profiles.keys())
