import time
import requests
from pathlib import Path
import hashlib

from config import RATELIMITSECONDS, CACHEDIR

def hashURL(url: str):
    return hashlib.md5(url.encode()).hexdigest()

class FBRefFetcher:
    def __init__(self, delay: int=RATELIMITSECONDS, cacheDir: str=CACHEDIR):
        self.delay = delay
        self.lastRequestTime = 0
        self.cacheDir = Path(cacheDir)
        self.cacheDir.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str, cache: bool=True) -> str:
        filename = self.cacheDir / (f"{hashURL(url)}.html")

        if cache and filename.exists():
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        
        timeSinceRequest = time.time() - self.lastRequestTime
        if timeSinceRequest < self.delay:
            time.sleep(self.delay - timeSinceRequest)
        
        print(f"Fetching: {url}")
        response = requests.get(url)
        response.raise_for_status()

        html = response.text
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
        
        self.lastRequestTime = time.time()
        return html