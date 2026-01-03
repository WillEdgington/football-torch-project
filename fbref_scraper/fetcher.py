import time
from pathlib import Path
import hashlib
import asyncio

from rnet import Impersonate, Client

from .config import RATELIMITSECONDS, CACHEDIR

def hashURL(url: str):
    return hashlib.md5(url.encode()).hexdigest()

class FBRefFetcher:
    def __init__(self, delay: int=RATELIMITSECONDS, cacheDir: str=CACHEDIR):
        self.delay = delay
        self.lastRequestTime = 0
        self.failAttempts = 0

        self.cacheDir = Path(cacheDir)
        self.cacheDir.mkdir(parents=True, exist_ok=True)
        
        self.client = Client(impersonate=Impersonate.Firefox139)
        self._loop = None

    async def fetch_async(self, url: str, cache: bool=True, mute: bool=False) -> str:
        filename = self.cacheDir / (f"{hashURL(url)}.html")

        if cache and filename.exists():
            return filename.read_text(encoding="utf-8")

        timeSinceRequest = time.time() - self.lastRequestTime
        if timeSinceRequest < self.delay:
            await asyncio.sleep(self.delay - timeSinceRequest)

        if not mute:
            print(f"Fetching: {url}")

        try:
            response = await self.client.get(url)
            html = await response.text()
        except Exception as exc:
            if self.failAttempts < 5:
                self.failAttempts += 1
                wait = 30 + (30 * self.failAttempts)
                if not mute:
                    print(
                        f"Request failed ({exc}). "
                        f"{5 - self.failAttempts} attempts remaining. "
                        f"Retrying in {wait}s..."
                    )
                await asyncio.sleep(wait)
                return await self.fetch_async(url, cache, mute)
            return ""

        self.failAttempts = 0
        self.lastRequestTime = time.time()

        if cache:
            filename.write_text(html, encoding="utf-8")

        return html
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def fetch(self, url: str, cache: bool=True, mute: bool=False) -> str:
        loop = self._get_loop()
        return loop.run_until_complete(self.fetch_async(url=url,
                                                        cache=cache,
                                                        mute=mute))
    
# fetcher = FBRefFetcher()
# html = fetcher.fetch(url="https://fbref.com/en/matches/560a0a6c/Brentford-Tottenham-Hotspur-January-1-2026-Premier-League",
#                      cache=False)
# print(type(html), html[:100])