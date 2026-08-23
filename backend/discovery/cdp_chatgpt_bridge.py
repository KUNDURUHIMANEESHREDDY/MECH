r"""Direct Chrome DevTools Protocol (CDP) WebSocket Controller for ChatGPT Web UI.

Uses pure asyncio + aiohttp WebSocket client (zero C-extension dependencies) to:
1. Connect to active Chrome session on http://localhost:9222.
2. Locate or navigate to ChatGPT (https://chatgpt.com).
3. Type prompts into ChatGPT and trigger the Send action.
4. Monitor streaming completion.
5. Extract the assistant's latest response.
6. Run an autonomous back-and-forth communication loop between Antigravity and ChatGPT.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.parse
import aiohttp


class CDPChatGPTController:
    """Controls ChatGPT web interface directly via Chrome DevTools Protocol WebSocket."""

    def __init__(self, host: str = "localhost", port: int = 9222) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.tab_id: Optional[str] = None
        self.ws_url: Optional[str] = None
        self._msg_id = 0
        self._pending_futures: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Finds or creates a ChatGPT tab and connects to its WebSocket debugger."""
        self.session = aiohttp.ClientSession()

        # List all pages
        async with self.session.get(f"{self.base_url}/json/list") as resp:
            tabs = await resp.json()

        chat_tab = None
        for t in tabs:
            if "chatgpt.com" in t.get("url", "") or "openai.com" in t.get("url", ""):
                chat_tab = t
                break

        if not chat_tab:
            print("[CDP] No existing ChatGPT tab found. Opening a new tab for https://chatgpt.com...")
            async with self.session.put(f"{self.base_url}/json/new?https://chatgpt.com") as resp:
                chat_tab = await resp.json()
                await asyncio.sleep(2.0)

        self.tab_id = chat_tab["id"]
        self.ws_url = chat_tab.get("webSocketDebuggerUrl")
        if not self.ws_url:
            self.ws_url = f"ws://{self.host}:{self.port}/devtools/page/{self.tab_id}"

        print(f"[CDP] Connecting to Chrome Tab [{self.tab_id}] at {self.ws_url}...")
        self.ws = await self.session.ws_connect(self.ws_url)
        self._reader_task = asyncio.create_task(self._read_ws_loop())

        # Enable runtime and page domains
        await self.send_command("Page.enable")
        await self.send_command("Runtime.enable")
        await self.send_command("Page.bringToFront")

        print("[CDP] Successfully connected and focused ChatGPT tab!")
        return True

    async def _read_ws_loop(self) -> None:
        """Reads incoming WebSocket messages from Chrome CDP."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_id = data.get("id")
                    if msg_id in self._pending_futures:
                        fut = self._pending_futures.pop(msg_id)
                        if not fut.done():
                            fut.set_result(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[CDP WS Error] {e}", file=sys.stderr)

    async def send_command(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sends a CDP command and awaits the response."""
        self._msg_id += 1
        msg_id = self._msg_id
        payload = {
            "id": msg_id,
            "method": method,
            "params": params or {},
        }
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_futures[msg_id] = fut

        await self.ws.send_str(json.dumps(payload))
        return await fut

    async def evaluate_js(self, js_code: str) -> Any:
        """Evaluates a JavaScript expression in the page context and returns the result."""
        res = await self.send_command("Runtime.evaluate", {
            "expression": js_code,
            "returnByValue": True,
            "awaitPromise": True,
        })
        result_obj = res.get("result", {}).get("result", {})
        if "value" in result_obj:
            return result_obj["value"]
        if "description" in result_obj:
            return result_obj["description"]
        return None

    async def send_prompt(self, prompt_text: str, timeout_seconds: int = 180) -> str:
        """Types prompt into ChatGPT input box, clicks send, and waits for complete response."""
        # 1. Bring page to front
        await self.send_command("Page.bringToFront")

        # 2. Check if logged in & locate input box
        find_input_js = """
        (() => {
            const sel = ['#prompt-textarea', 'div[contenteditable="true"]', 'textarea[placeholder*="Message"]', 'div[data-placeholder*="Message"]'];
            for (const s of sel) {
                const el = document.querySelector(s);
                if (el) return s;
            }
            return null;
        })()
        """
        input_selector = None
        for _ in range(15):
            input_selector = await self.evaluate_js(find_input_js)
            if input_selector:
                break
            await asyncio.sleep(1.0)

        if not input_selector:
            # Check page title/text to see if Cloudflare or login page is active
            page_text = await self.evaluate_js("document.title + ' | ' + (document.body ? document.body.innerText.slice(0, 200) : '')")
            return f"[Error: Could not locate ChatGPT prompt box. Page info: '{page_text}'. Please ensure you are logged into ChatGPT in Chrome.]"

        # 3. Enter text and click send
        escaped_prompt = json.dumps(prompt_text)
        type_and_send_js = f"""
        (() => {{
            const inputEl = document.querySelector('{input_selector}');
            if (!inputEl) return false;

            inputEl.focus();

            // Set content depending on element type
            if (inputEl.tagName === 'TEXTAREA') {{
                inputEl.value = {escaped_prompt};
                inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
                inputEl.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }} else {{
                inputEl.innerText = {escaped_prompt};
                inputEl.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}

            // Give React a moment to update state
            setTimeout(() => {{
                // Find send button
                const sendBtn = document.querySelector('button[data-testid="send-button"]') ||
                                document.querySelector('button[aria-label="Send prompt"]') ||
                                document.querySelector('button[aria-label="Send message"]');
                if (sendBtn && !sendBtn.disabled) {{
                    sendBtn.click();
                }} else {{
                    // Fallback to Enter keydown
                    const event = new KeyboardEvent('keydown', {{
                        key: 'Enter',
                        code: 'Enter',
                        keyCode: 13,
                        which: 13,
                        bubbles: true
                    }});
                    inputEl.dispatchEvent(event);
                }}
            }}, 300);

            return true;
        }})()
        """
        sent_ok = await self.evaluate_js(type_and_send_js)
        if not sent_ok:
            return "[Error: Failed to inject prompt into input element.]"

        print(f"[CDP] Prompt dispatched ({len(prompt_text)} chars). Waiting for response streaming to start...")
        await asyncio.sleep(2.5)

        # 4. Wait for generation to complete (detect when streaming/stop buttons disappear)
        is_streaming_js = """
        (() => {
            const stopBtn = document.querySelector('button[data-testid="stop-button"]') ||
                            document.querySelector('button[aria-label="Stop streaming"]') ||
                            document.querySelector('button[aria-label="Stop generating"]');
            return !!stopBtn;
        })()
        """
        start_time = time.time()
        # Poll until streaming completes
        while time.time() - start_time < timeout_seconds:
            is_streaming = await self.evaluate_js(is_streaming_js)
            if not is_streaming:
                # Wait an extra second to ensure DOM is fully rendered
                await asyncio.sleep(1.0)
                is_streaming_again = await self.evaluate_js(is_streaming_js)
                if not is_streaming_again:
                    break
            await asyncio.sleep(1.0)

        # 5. Extract the last assistant response
        extract_response_js = """
        (() => {
            const msgs = document.querySelectorAll('div[data-message-author-role="assistant"], article[data-testid*="conversation-turn"]');
            if (msgs.length > 0) {
                const last = msgs[msgs.length - 1];
                return last.innerText || last.textContent || '';
            }
            const markdownBlocks = document.querySelectorAll('div.markdown');
            if (markdownBlocks.length > 0) {
                const last = markdownBlocks[markdownBlocks.length - 1];
                return last.innerText || last.textContent || '';
            }
            return '';
        })()
        """
        response_text = await self.evaluate_js(extract_response_js)
        if response_text and len(response_text.strip()) > 0:
            return response_text.strip()
        else:
            return "[Error: Response generation finished, but no assistant message container was found.]"

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()


async def run_autonomous_loop(initial_prompt: str, rounds: int = 2) -> None:
    """Executes a multi-round bidirectional communication loop between Antigravity and ChatGPT."""
    print("=" * 65)
    print("   STARTING ANTIGRAVITY <-> CHATGPT AUTONOMOUS BROWSER LOOP   ")
    print("=" * 65)

    controller = CDPChatGPTController()
    connected = await controller.connect()
    if not connected:
        print("[Error] Failed to connect to Chrome CDP.")
        return

    current_prompt = initial_prompt

    for r in range(1, rounds + 1):
        print(f"\n" + "=" * 30 + f" ROUND {r}/{rounds} " + "=" * 30)
        print(f"\n[Antigravity -> ChatGPT]:\n{current_prompt}\n")

        response = await controller.send_prompt(current_prompt)
        print(f"\n[ChatGPT -> Antigravity]:\n{response}\n")

        if r < rounds:
            # Antigravity synthesizes follow-up critique based on ChatGPT's response
            current_prompt = (
                f"Antigravity received your reply:\n'''\n{response[:400]}...\n'''\n\n"
                f"Based on your response, what is the single most critical empirical experiment "
                f"to validate or falsify this hypothesis? Please answer concisely in 3 bullet points."
            )

    await controller.close()
    print("\n" + "=" * 65)
    print("   AUTONOMOUS LOOP COMPLETED SUCCESSFULLY!   ")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity <-> ChatGPT CDP Browser Loop")
    parser.add_argument(
        "--prompt",
        default="Hello ChatGPT! This is Google Antigravity communicating directly via browser automation. Can you briefly summarize the 3 main open problems in Mechanistic Interpretability?",
        help="Initial prompt for ChatGPT",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Number of loop rounds")
    args = parser.parse_args()

    asyncio.run(run_autonomous_loop(args.prompt, args.rounds))
