import time
import requests
from pathlib import Path
import hashlib

from .config import RATELIMITSECONDS, CACHEDIR

def hashURL(url: str):
    return hashlib.md5(url.encode()).hexdigest()

class FBRefFetcher:
    def __init__(self, delay: int=RATELIMITSECONDS, cacheDir: str=CACHEDIR):
        self.delay = delay
        self.lastRequestTime = 0
        self.cacheDir = Path(cacheDir)
        self.cacheDir.mkdir(parents=True, exist_ok=True)
        self.failAttempts = 0

    def fetch(self, url: str, cache: bool=True, mute: bool=False) -> str:
        filename = self.cacheDir / (f"{hashURL(url)}.html")

        if cache and filename.exists():
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        
        timeSinceRequest = time.time() - self.lastRequestTime
        if timeSinceRequest < self.delay:
            time.sleep(self.delay - timeSinceRequest)

        html = ""
        if not mute:
            print(f"Fetching: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            html = response.text
        except requests.exceptions.HTTPError:
            if self.failAttempts < 5:
                if not mute:
                    print(f"Request failed. {5 - self.failAttempts} attempts remaining. Trying again in {30 + (30 * self.failAttempts)} seconds...")
                self.failAttempts += 1
                time.sleep(30 + (30 * self.failAttempts))
                return self.fetch(url=url, cache=cache)
            return html

        self.failAttempts = 0
        if cache:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
        
        self.lastRequestTime = time.time()
        return html