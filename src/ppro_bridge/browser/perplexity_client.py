import asyncio
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / ".browser_profile"
PERPLEXITY_HOSTS = {"www.perplexity.ai", "perplexity.ai"}


class PerplexityClient:
    def __init__(self) -> None:
        self._pw = None
        self._context = None
        self._page = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._pw = await async_playwright().start()

        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            no_viewport=True,
            args=["--start-maximized"],
            permissions=["clipboard-read", "clipboard-write"],
        )

        pages = self._context.pages

        self._page = next(
            (
                page
                for page in pages
                if urlparse(page.url).netloc.lower() in PERPLEXITY_HOSTS
            ),
            None,
        )

        if self._page is None:
            self._page = pages[0] if pages else await self._context.new_page()
            await self._page.goto(
            "https://www.perplexity.ai/",
            wait_until="domcontentloaded",
        )

        await self._page.locator("#ask-input").wait_for(
            state="visible",
            timeout=30_000,
        )

    async def stop(self) -> None:
        if self._context:
            await self._context.close()
        if self._pw:
            await self._pw.stop()

    async def browser_status(self) -> dict:
        if self._page is None:
            return {
                "started": False,
                "url": None,
                "title": None,
                "is_perplexity": False,
                "chat_type": "unavailable",
                "chat_id": None,
            }

        url = self._page.url
        title = await self._page.title()

        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path_parts = [part for part in parsed.path.split("/") if part]

        chat_id = None
        if len(path_parts) >= 2 and path_parts[0] == "search":
            chat_id = path_parts[1]

        is_perplexity = host in PERPLEXITY_HOSTS

        if not is_perplexity:
            chat_type = "other_page"
        elif chat_id:
            chat_type = "existing_chat"
        else:
            chat_type = "new_or_home"

        return {
            "started": True,
            "url": url,
            "title": title,
            "is_perplexity": is_perplexity,
            "chat_type": chat_type,
            "chat_id": chat_id,
        }

    async def _copy_last(self, timeout_ms: int = 1_500) -> str | None:
        button = self._page.get_by_role("button", name="Copiar", exact=True).last

        try:
            await button.wait_for(state="visible", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            return None

        try:
            await self._page.evaluate("navigator.clipboard.writeText('')")
            await button.click(force=True, timeout=5_000)
            await asyncio.sleep(0.15)
            text = await self._page.evaluate("navigator.clipboard.readText()")
            return text.strip() or None
        except PlaywrightTimeoutError:
            return None

    async def ask(self, prompt: str, timeout_s: float = 120) -> str:
        async with self._lock:
            baseline = await self._copy_last()

            editor = self._page.locator("#ask-input")
            await editor.wait_for(state="visible", timeout=30_000)
            await editor.click()
            await editor.fill(prompt)
            await editor.press("Enter")

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout_s
            previous = None
            stable_reads = 0

            while loop.time() < deadline:
                await asyncio.sleep(2)
                text = await self._copy_last()

                if not text or text == baseline or text == prompt.strip():
                    previous = None
                    stable_reads = 0
                    continue

                if text == previous:
                    stable_reads += 1
                    if stable_reads >= 1:
                        return text
                else:
                    previous = text
                    stable_reads = 0

            raise TimeoutError("Resposta não obtida dentro do tempo limite.")
