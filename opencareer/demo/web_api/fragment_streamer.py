"""
FragmentStreamer — token-level streaming with <split/> marker detection.

Wraps an LLM token stream and splits it into multiple short message
fragments. Detects <split/> markers (even across token boundaries),
sends clean tokens + fragment_break signals.

Designed for the "fragmented messaging" (碎片化发送) UX pattern:
AI responses appear as multiple sequential chat bubbles that mimic
WeChat-style natural conversation rhythm.
"""

import logging
from typing import Any, Callable

logger = logging.getLogger("web_api.fragment_streamer")

MARKER = "<split/>"
MAX_FRAGMENTS = 6  # Hard limit: max 6 fragments per reply (≤5 <split/> markers)


class FragmentStreamer:
    """State machine that buffers tokens, detects <split/> markers, and
    emits fragment_break signals between logical message fragments.

    Handles edge cases:
    - Marker split across token boundaries (prefix-matching buffer)
    - Consecutive markers
    - Marker at start/end of stream
    - Marker count exceeding MAX_FRAGMENTS (ignored as plain text)
    """

    def __init__(self, send_json: Callable[[dict], Any]):
        self._send_json = send_json
        self._buffer = ""
        self._fragment_count = 0

    async def feed(self, token: str) -> None:
        """Feed one token into the streamer. Automatically detects markers,
        sends clean tokens, and emits fragment_break signals."""
        self._buffer += token

        # Loop to handle consecutive markers in a single token
        while MARKER in self._buffer:
            if self._fragment_count >= MAX_FRAGMENTS:
                # Fragment limit reached — ignore remaining <split/> markers,
                # treat them as plain text in the current bubble
                safe_len = self._safe_prefix_len(self._buffer)
                if safe_len > 0:
                    safe = self._buffer[:safe_len]
                    await self._send_json({"type": "token", "content": safe})
                    self._buffer = self._buffer[safe_len:]
                return

            before, after = self._buffer.split(MARKER, 1)
            if before:
                await self._send_json({"type": "token", "content": before})
            await self._send_json({"type": "fragment_break"})
            self._fragment_count += 1
            self._buffer = after

        # No more markers — flush safe portion (excluding partial prefix)
        safe_len = self._safe_prefix_len(self._buffer)
        if safe_len > 0:
            safe = self._buffer[:safe_len]
            await self._send_json({"type": "token", "content": safe})
            self._buffer = self._buffer[safe_len:]

    async def flush(self) -> None:
        """Flush any remaining buffered tokens at stream end."""
        if self._buffer:
            await self._send_json({"type": "token", "content": self._buffer})
            self._buffer = ""

    @staticmethod
    def _safe_prefix_len(text: str) -> int:
        """Return the length of text that can be safely sent without risk
        of being part of a partial MARKER match.

        Checks if text ends with any prefix of MARKER ("<", "<s", "<sp",
        "<spl", "<spli", "<split", "<split/") and excludes that suffix.
        """
        for i in range(1, len(MARKER)):
            if text.endswith(MARKER[:i]):
                return len(text) - i
        return len(text)
