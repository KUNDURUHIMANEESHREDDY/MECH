r"""ChatGPT Web Browser Automation & Continuous Feedback Loop for Antigravity.

Enables an autonomous bidirectional loop between Antigravity and the ChatGPT Web UI / App:
1. Connects to active Chrome session via Chrome DevTools Protocol (CDP, localhost:9222)
   or launches a persistent browser context on chatgpt.com.
2. Sends Antigravity's outputs/hypotheses/prompts directly into the ChatGPT web input.
3. Waits for ChatGPT to stream its response, detects completion, and extracts the text.
4. Returns the response back to Antigravity to generate the next iteration.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)


try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext
except ImportError:
    print("Error: Playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)


class ChatGPTBrowserController:
    """Controls the ChatGPT Web UI via Playwright browser automation."""

    def __init__(self, cdp_url: str = "http://localhost:9222", user_data_dir: Optional[str] = None) -> None:
        self.cdp_url = cdp_url
        self.user_data_dir = user_data_dir or os.path.expanduser("~/.chatgpt_playwright_profile")
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def initialize(self) -> Page:
        """Connects to an existing Chrome instance or launches a persistent browser."""
        self._playwright = sync_playwright().start()

        # Try connecting via CDP first
        try:
            print(f"[Browser] Attempting connection to Chrome CDP at {self.cdp_url}...")
            self._browser = self._playwright.chromium.connect_over_cdp(self.cdp_url)
            self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
            
            # Find an active ChatGPT tab
            for page in self._context.pages:
                if "chatgpt.com" in page.url or "openai.com" in page.url:
                    self._page = page
                    break

            if not self._page:
                self._page = self._context.new_page()
                self._page.goto("https://chatgpt.com")
            print("[Browser] Connected to active Chrome session!")
        except Exception as cdp_err:
            print(f"[Browser] CDP connection failed ({cdp_err}). Launching standalone persistent browser...")
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir,
                headless=False,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"],
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._page.goto("https://chatgpt.com")

        self._page.bring_to_front()
        return self._page

    def send_prompt_and_get_response(self, prompt_text: str, timeout_seconds: int = 180) -> str:
        """Enters prompt into ChatGPT prompt area, clicks Send, and waits for complete generation."""
        if not self._page:
            self.initialize()

        page = self._page
        page.bring_to_front()

        # Input box selectors used by ChatGPT
        input_selectors = [
            '#prompt-textarea',
            'div[contenteditable="true"]',
            'textarea[placeholder*="Message"]',
            'div[data-placeholder*="Message"]',
        ]

        found_input = None
        for sel in input_selectors:
            try:
                page.wait_for_selector(sel, timeout=4000)
                found_input = sel
                break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)
                continue

        if not found_input:
            raise RuntimeError(
                "Could not locate ChatGPT prompt textarea. Please ensure you are logged into https://chatgpt.com in the browser window."
            )

        # Clear and fill the prompt
        try:
            page.fill(found_input, prompt_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            page.click(found_input)
            page.keyboard.type(prompt_text)

        time.sleep(0.5)
        # Send message
        page.keyboard.press("Enter")
        time.sleep(2.0)

        # Wait for generation to finish
        # The 'Stop generating' button appears while streaming and disappears when done
        stop_selectors = [
            'button[data-testid="stop-button"]',
            'button[aria-label="Stop streaming"]',
            'button[aria-label="Stop generating"]',
        ]

        start_time = time.time()
        # Wait until streaming finishes
        for s_sel in stop_selectors:
            try:
                page.wait_for_selector(s_sel, state="detached", timeout=timeout_seconds * 1000)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Swallowed exception: %s", exc)

        time.sleep(1.5)

        # Extract the last assistant response
        response_selectors = [
            'div[data-message-author-role="assistant"]',
            'article[data-testid*="conversation-turn"]',
            'div.markdown',
        ]

        for r_sel in response_selectors:
            elements = page.query_selector_all(r_sel)
            if elements:
                last_el = elements[-1]
                text = last_el.inner_text().strip()
                if text:
                    return text

        return "[Error: Response generation completed, but could not parse response container text.]"

    def close(self) -> None:
        if self._playwright:
            self._playwright.stop()


def run_bidirectional_loop(initial_prompt: str, rounds: int = 3) -> None:
    """Executes a multi-round autonomous communication loop between Antigravity and ChatGPT."""
    print(f"\n=======================================================")
    print(f"   STARTING ANTIGRAVITY <-> CHATGPT AUTONOMOUS LOOP    ")
    print(f"=======================================================\n")

    controller = ChatGPTBrowserController()
    controller.initialize()

    current_input = initial_prompt

    for r in range(1, rounds + 1):
        print(f"\n--- ROUND {r}/{rounds} ---")
        print(f"\n[Antigravity -> ChatGPT]:\n{current_input}\n")

        response = controller.send_prompt_and_get_response(current_input)

        print(f"\n[ChatGPT -> Antigravity]:\n{response}\n")

        if r < rounds:
            # Antigravity generates next synthesis / critique
            current_input = (
                f"Antigravity Received your reply:\n'''{response}'''\n\n"
                f"Here is my follow-up refinement/critique for Round {r+1}: "
                f"How would you optimize and test this further?"
            )

    print("\nAutonomous loop complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity <-> ChatGPT Web Browser Loop")
    parser.add_argument("--prompt", default="Hello ChatGPT! You are now connected to Antigravity via browser automation. Please summarize the key principles of causal mechanistic interpretability.", help="Initial prompt")
    parser.add_argument("--rounds", type=int, default=2, help="Number of communication loop rounds")
    args = parser.parse_args()

    run_bidirectional_loop(args.prompt, args.rounds)
