"""
Unit tests for FragmentStreamer — token-level streaming with <split/> marker detection.

Covers:
- Normal split
- Cross-token marker detection
- No marker / plain text passthrough
- Consecutive markers
- Marker at start / end of stream
- Empty input
- MAX_FRAGMENTS hard limit
- _safe_prefix_len edge cases
"""

import pytest
from unittest.mock import AsyncMock

from web_api.fragment_streamer import FragmentStreamer, MARKER, MAX_FRAGMENTS


# ------------------------------------------------------------------
# Helper: build a streamer with a mock send_json that records calls
# ------------------------------------------------------------------

def _make_streamer():
    """Return (FragmentStreamer, list) where list records send_json args."""
    calls = []
    send = AsyncMock()

    async def record(msg: dict):
        calls.append(msg)

    send.side_effect = record
    streamer = FragmentStreamer(send)
    return streamer, calls, send


# ------------------------------------------------------------------
# _safe_prefix_len static method tests (pure function, no async)
# ------------------------------------------------------------------

class TestSafePrefixLen:
    def test_no_partial_match(self):
        assert FragmentStreamer._safe_prefix_len("hello world") == 11

    def test_full_marker_is_not_a_prefix(self):
        # MARKER itself has no trailing partial prefix (ends with "/>", which
        # matches none of <, <s, <sp, <spl, <spli, <split, <split/), so the
        # full length is safe to emit. The while-loop in feed() handles the
        # actual split before safe_prefix_len is called on the full marker.
        assert FragmentStreamer._safe_prefix_len(MARKER) == len(MARKER)

    def test_prefix_lt(self):
        assert FragmentStreamer._safe_prefix_len("text<") == 4

    def test_prefix_lts(self):
        assert FragmentStreamer._safe_prefix_len("text<s") == 4

    def test_prefix_ltspl(self):
        assert FragmentStreamer._safe_prefix_len("text<sp") == 4

    def test_prefix_ltspli(self):
        assert FragmentStreamer._safe_prefix_len("text<spl") == 4

    def test_prefix_ltsplit(self):
        assert FragmentStreamer._safe_prefix_len("text<spli") == 4

    def test_prefix_ltsplit_slash(self):
        assert FragmentStreamer._safe_prefix_len("text<split") == 4

    def test_prefix_lt_slash(self):
        assert FragmentStreamer._safe_prefix_len("text<split/") == 4

    def test_empty_string(self):
        assert FragmentStreamer._safe_prefix_len("") == 0

    def test_only_partial_prefix(self):
        # String is entirely a prefix of MARKER
        assert FragmentStreamer._safe_prefix_len("<spl") == 0


# ------------------------------------------------------------------
# feed / flush tests (async)
# ------------------------------------------------------------------

class TestFeedFlush:
    @pytest.mark.asyncio
    async def test_normal_split(self):
        """#1: tokens split by <split/> produce clean token + fragment_break."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("今天不错")
        await streamer.feed("<split/>")
        await streamer.feed("继续努力")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "今天不错"},
            {"type": "fragment_break"},
            {"type": "token", "content": "继续努力"},
        ]

    @pytest.mark.asyncio
    async def test_cross_token_marker(self):
        """#2: <split/> split across token boundaries is detected."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("今天不错<spli")
        await streamer.feed("t/>继续努力")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "今天不错"},
            {"type": "fragment_break"},
            {"type": "token", "content": "继续努力"},
        ]

    @pytest.mark.asyncio
    async def test_no_marker(self):
        """#3: plain text with no marker passes through as single token."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("今天不错，继续努力")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "今天不错，继续努力"},
        ]

    @pytest.mark.asyncio
    async def test_consecutive_markers(self):
        """#4: consecutive <split/> markers produce consecutive fragment_breaks."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("a<split/><split/>b")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "a"},
            {"type": "fragment_break"},
            {"type": "fragment_break"},
            {"type": "token", "content": "b"},
        ]

    @pytest.mark.asyncio
    async def test_marker_at_start(self):
        """#5: marker at the very beginning of a stream."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("<split/>你好")
        await streamer.flush()

        assert calls == [
            {"type": "fragment_break"},
            {"type": "token", "content": "你好"},
        ]

    @pytest.mark.asyncio
    async def test_marker_at_end(self):
        """#6: marker at the very end of a stream."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("你好<split/>")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "你好"},
            {"type": "fragment_break"},
        ]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """#7: no tokens — flush produces nothing."""
        streamer, calls, _ = _make_streamer()
        await streamer.flush()

        assert calls == []

    @pytest.mark.asyncio
    async def test_exceeds_max_fragments(self):
        """#8: after MAX_FRAGMENTS breaks, subsequent <split/> is plain text."""
        streamer, calls, _ = _make_streamer()
        # Input: 7 <split/> markers — only first 5 should break (starting 6 fragments)
        # "a" then 7 pairs of "<split/>x" → this gives 7 markers and text after each
        await streamer.feed(
            "a<split/>b<split/>c<split/>d<split/>e<split/>f<split/>g<split/>h"
        )
        await streamer.flush()

        # After 5 <split/> markers, fragment_count = 5
        # The 6th marker triggers the MAX_FRAGMENTS check (5 >= 6? no, 5 < 6)
        # Wait — let me trace through:
        #
        # feed "a<split/>b<split/>c<split/>d<split/>e<split/>f<split/>g<split/>h"
        #
        # 1st iteration: buffer="a<split/>...", split → before="a", after="b<split/>c..."
        #   → token:"a", fragment_break, count=1
        # 2nd iteration: buffer="b<split/>c...", split → before="b", after="c<split/>d..."
        #   → token:"b", fragment_break, count=2
        # 3rd: token:"c", fragment_break, count=3
        # 4th: token:"d", fragment_break, count=4
        # 5th: token:"e", fragment_break, count=5
        # 6th: buffer="f<split/>g<split/>h" — count=5, 5 >= MAX_FRAGMENTS(6)? No, 5 < 6
        #   → token:"f", fragment_break, count=6
        # 7th: buffer="g<split/>h" — count=6, 6 >= 6? YES
        #   → safe_prefix of "g<split/>h": no partial marker at end, so safe_len=10
        #   → token:"g<split/>h", buffer=""
        # flush: buffer empty, nothing

        expected = []
        for ch in ["a", "b", "c", "d", "e", "f"]:
            expected.append({"type": "token", "content": ch})
            expected.append({"type": "fragment_break"})
        # After the 6th break (count=6), remaining text is plain
        expected.append({"type": "token", "content": "g<split/>h"})

        assert calls == expected

    @pytest.mark.asyncio
    async def test_exactly_max_fragments(self):
        """Edge: exactly MAX_FRAGMENTS markers — all break, no leftover text."""
        streamer, calls, _ = _make_streamer()
        # 6 tokens separated by 5 <split/> → 6 fragments (MAX_FRAGMENTS)
        await streamer.feed("a<split/>b<split/>c<split/>d<split/>e<split/>f")
        await streamer.flush()

        expected = []
        for ch in ["a", "b", "c", "d", "e"]:
            expected.append({"type": "token", "content": ch})
            expected.append({"type": "fragment_break"})
        expected.append({"type": "token", "content": "f"})

        assert calls == expected

    @pytest.mark.asyncio
    async def test_marker_broken_across_multiple_tokens(self):
        """Cross-token: marker split into 3+ pieces."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("hello<")
        await streamer.feed("spli")
        await streamer.feed("t/>world")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "hello"},
            {"type": "fragment_break"},
            {"type": "token", "content": "world"},
        ]

    @pytest.mark.asyncio
    async def test_multiple_cross_token_markers(self):
        """Multiple cross-token markers in sequence."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("x<")
        await streamer.feed("split/>y<split")
        await streamer.feed("/>z")
        await streamer.flush()

        assert calls == [
            {"type": "token", "content": "x"},
            {"type": "fragment_break"},
            {"type": "token", "content": "y"},
            {"type": "fragment_break"},
            {"type": "token", "content": "z"},
        ]

    @pytest.mark.asyncio
    async def test_flush_without_feed(self):
        """flush on a fresh streamer does nothing."""
        streamer, calls, _ = _make_streamer()
        await streamer.flush()
        # Calling flush again should also be harmless
        await streamer.flush()
        assert calls == []

    @pytest.mark.asyncio
    async def test_partial_marker_held_at_end(self):
        """Text ending with a partial marker prefix is held until next feed."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("data<")
        # "data<" → safe_prefix_len returns 4 ("data" is safe), "<" is held
        assert calls == [{"type": "token", "content": "data"}]
        await streamer.feed("s")
        # buffer is now "<s" — still a prefix of MARKER, nothing emitted
        assert calls == [{"type": "token", "content": "data"}]
        await streamer.feed("pan>")  # buffer becomes "<span>", not MARKER
        await streamer.flush()
        assert calls == [
            {"type": "token", "content": "data"},
            {"type": "token", "content": "<span>"},
        ]

    @pytest.mark.asyncio
    async def test_send_json_called_correctly(self):
        """The send_json callback is invoked (not just recorded)."""
        streamer, _, send = _make_streamer()
        await streamer.feed("hello")
        await streamer.flush()
        send.assert_awaited_once_with({"type": "token", "content": "hello"})


# ------------------------------------------------------------------
# Real-world simulation tests
# ------------------------------------------------------------------

class TestRealWorld:
    @pytest.mark.asyncio
    async def test_multi_bubble_career_advice(self):
        """Simulate a realistic multi-fragment career advice response."""
        streamer, calls, _ = _make_streamer()

        # Simulate the LLM streaming tokens with <split/> markers
        tokens = [
            "根据你的背景，",
            "我建议从以下",
            "几个方面入手<split/>",
            "首先，优化简历",
            "中的项目描述，",
            "突出量化成果<split/>",
            "其次，准备系统",
            "设计面试，重点",
            "关注分布式架构<split/>",
            "最后，保持积极",
            "心态，祝你成功！",
        ]
        for t in tokens:
            await streamer.feed(t)
        await streamer.flush()

        # 11 token emissions + 3 fragment_breaks = 14 calls total
        assert len(calls) == 14
        fragment_breaks = [c for c in calls if c["type"] == "fragment_break"]
        assert len(fragment_breaks) == 3

    @pytest.mark.asyncio
    async def test_single_long_message_no_marker(self):
        """A long message without markers stays as one fragment."""
        streamer, calls, _ = _make_streamer()
        msg = "这是一个很长的回复，包含很多有用的建议和信息，但是没有使用任何分段标记。"
        await streamer.feed(msg)
        await streamer.flush()

        assert len(calls) == 1
        assert calls[0] == {"type": "token", "content": msg}

    @pytest.mark.asyncio
    async def test_marker_only_string(self):
        """Input is literally just '<split/>' — single fragment_break."""
        streamer, calls, _ = _make_streamer()
        await streamer.feed("<split/>")
        await streamer.flush()

        assert calls == [
            {"type": "fragment_break"},
        ]
