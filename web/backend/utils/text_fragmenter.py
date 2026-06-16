import random
import re
from dataclasses import dataclass
from typing import Literal


FragmentMode = Literal["chatty", "balanced", "structured"]


@dataclass
class Sentence:
    content: str
    emotion: str
    delay_ms: int
    kind: str = "message"
    mode: FragmentMode = "chatty"


class TextFragmenter:
    """Split assistant replies into WeChat-like message bubbles."""

    SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])\s+|(?<=[。！？!?])")
    SOFT_BOUNDARY = re.compile(r"(?<=[，,；;：:])\s*")
    MARKDOWN_HEADING = re.compile(r"(?m)^#{1,3}\s+.+$")
    NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)、]|[-*+])\s+")

    STRUCTURED_REQUEST_KEYWORDS = (
        "结构化", "专业", "详细", "报告", "Markdown", "markdown", "清单",
        "列表", "编号", "框架", "方案", "分析一下", "系统地", "小标题",
    )
    STRUCTURED_OUTPUT_MARKERS = ("## ", "### ", "\n1.", "\n1、", "\n- ", "\n* ", "|")

    EMOTION_KEYWORDS = {
        "empathy": ("理解", "明白", "懂", "别急", "正常", "没关系", "焦虑", "压力"),
        "excited": ("太好了", "不错", "很棒", "可以", "稳", "恭喜"),
        "serious": ("建议", "关键", "重要", "注意", "需要", "优先", "风险", "问题"),
    }

    PARTICLES = ("啊", "呢", "哈", "呀")
    PARTICLE_PROBABILITY = 0.015

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def fragment_with_particles(
        self,
        text: str,
        user_message: str = "",
        mode: FragmentMode | None = None,
    ) -> list[Sentence]:
        clean_text = self._clean_text(text)
        if not clean_text:
            return []

        resolved_mode = mode or self.detect_mode(user_message, clean_text)
        if resolved_mode == "structured":
            fragments = self._fragment_structured(clean_text)
        elif resolved_mode == "balanced":
            fragments = self._fragment_balanced(clean_text)
        else:
            fragments = self._fragment_chatty(clean_text)

        return [
            self._build_sentence(
                self._maybe_add_particle(item["content"], resolved_mode),
                kind=item.get("kind", "message"),
                mode=resolved_mode,
            )
            for item in fragments
            if item["content"].strip()
        ]

    def detect_mode(self, user_message: str, text: str) -> FragmentMode:
        if any(keyword in user_message for keyword in self.STRUCTURED_REQUEST_KEYWORDS):
            return "structured"
        if any(marker in text for marker in self.STRUCTURED_OUTPUT_MARKERS):
            return "balanced"
        if len(text) > 450:
            return "balanced"
        return "chatty"

    def _fragment_chatty(self, text: str) -> list[dict[str, str]]:
        pieces = self._split_plain_sentences(text)
        if len(pieces) == 1 and len(pieces[0]) > 42:
            pieces = self._split_long_piece(pieces[0])

        fragments: list[str] = []
        for piece in pieces:
            if len(piece) > 46:
                fragments.extend(self._split_long_piece(piece))
            else:
                fragments.append(piece)

        fragments = self._merge_tiny_fragments(fragments, min_len=6)
        fragments = self._cap_fragments(fragments, max_count=7, target_len=58)
        return [{"content": item, "kind": self._guess_kind(item)} for item in fragments]

    def _fragment_balanced(self, text: str) -> list[dict[str, str]]:
        if self._looks_structured(text):
            sections = self._split_markdown_sections(text)
            if sections:
                return [{"content": item, "kind": "section"} for item in self._cap_fragments(sections, 5, 360)]

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(paragraphs) <= 1:
            paragraphs = self._split_plain_sentences(text)
        chunks = self._cap_fragments(paragraphs, max_count=5, target_len=180)
        return [{"content": item, "kind": self._guess_kind(item)} for item in chunks]

    def _fragment_structured(self, text: str) -> list[dict[str, str]]:
        fragments = [
            {"content": "可以", "kind": "ack"},
            {"content": "我给你按专业版拆一下", "kind": "thinking"},
        ]

        sections = self._split_markdown_sections(text)
        if not sections:
            sections = self._split_structured_plain_text(text)

        if sections:
            first = sections[0]
            if not first.startswith("#") and len(first) <= 80:
                fragments.append({"content": f"先说结论：{first}", "kind": "summary"})
                sections = sections[1:]

        for section in sections:
            fragments.append({"content": section, "kind": "section"})

        return fragments

    def _split_markdown_sections(self, text: str) -> list[str]:
        lines = text.splitlines()
        sections: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            is_heading = bool(self.MARKDOWN_HEADING.match(line.strip()))
            if is_heading and current:
                sections.append(current)
                current = [line]
            else:
                current.append(line)

        if current:
            sections.append(current)

        cleaned = ["\n".join(part).strip() for part in sections if "\n".join(part).strip()]
        if len(cleaned) == 1 and not self.MARKDOWN_HEADING.search(cleaned[0]):
            return []
        return cleaned

    def _split_structured_plain_text(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if len(paragraphs) > 1:
            return self._cap_fragments(paragraphs, max_count=6, target_len=260)

        sentences = self._split_plain_sentences(text)
        return self._cap_fragments(sentences, max_count=5, target_len=220)

    def _split_plain_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s*\n+\s*", " ", text).strip()
        pieces = [p.strip() for p in self.SENTENCE_BOUNDARY.split(text) if p.strip()]
        return pieces or ([text] if text else [])

    def _split_long_piece(self, text: str) -> list[str]:
        parts = [p.strip() for p in self.SOFT_BOUNDARY.split(text) if p.strip()]
        if len(parts) <= 1:
            return [text]
        return self._merge_tiny_fragments(parts, min_len=8)

    def _merge_tiny_fragments(self, fragments: list[str], min_len: int) -> list[str]:
        merged: list[str] = []
        pending = ""
        for fragment in fragments:
            if pending:
                fragment = pending + fragment
                pending = ""
            if len(fragment) < min_len:
                pending = fragment
                continue
            merged.append(fragment)
        if pending:
            if merged:
                merged[-1] += pending
            else:
                merged.append(pending)
        return merged

    def _cap_fragments(self, fragments: list[str], max_count: int, target_len: int) -> list[str]:
        if len(fragments) <= max_count:
            return fragments

        capped: list[str] = []
        current = ""
        for fragment in fragments:
            separator = "\n\n" if self._looks_structured(fragment) or self._looks_structured(current) else ""
            candidate = f"{current}{separator}{fragment}" if current else fragment
            if current and len(candidate) > target_len and len(capped) < max_count - 1:
                capped.append(current)
                current = fragment
            else:
                current = candidate
        if current:
            capped.append(current)
        return capped

    def _looks_structured(self, text: str) -> bool:
        if not text:
            return False
        return bool(self.MARKDOWN_HEADING.search(text)) or any(
            self.NUMBERED_LINE.match(line) for line in text.splitlines()
        )

    def _guess_kind(self, text: str) -> str:
        if self._looks_structured(text):
            return "section"
        if any(word in text for word in ("理解", "懂", "别急", "没关系")):
            return "empathy"
        if any(word in text for word in ("建议", "可以", "先", "下一步")):
            return "suggestion"
        return "message"

    def _build_sentence(self, content: str, kind: str, mode: FragmentMode) -> Sentence:
        emotion = self.detect_emotion(content)
        return Sentence(
            content=content,
            emotion=emotion,
            delay_ms=self.calculate_delay(content, emotion, kind, mode),
            kind=kind,
            mode=mode,
        )

    def detect_emotion(self, sentence: str) -> str:
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(keyword in sentence for keyword in keywords):
                return emotion
        return "neutral"

    def calculate_delay(self, sentence: str, emotion: str, kind: str = "message", mode: FragmentMode = "chatty") -> int:
        if kind == "ack":
            return self._jitter(320, 620)
        if mode == "structured":
            return self._jitter(700, 1400)
        if kind == "section":
            return self._jitter(650, 1300)

        base = min(max(len(sentence) * 55, 420), 1800)
        if emotion == "empathy":
            base *= 1.25
        elif emotion == "excited":
            base *= 0.75
        elif emotion == "serious":
            base *= 1.1
        return int(min(max(base * self.random.uniform(0.85, 1.2), 350), 2400))

    def _jitter(self, low: int, high: int) -> int:
        return self.random.randint(low, high)

    def _maybe_add_particle(self, sentence: str, mode: FragmentMode) -> str:
        if mode != "chatty" or not sentence:
            return sentence
        if len(sentence) > 28 or sentence[-1] in "。！？!?~":
            return sentence
        if self.random.random() < self.PARTICLE_PROBABILITY:
            return sentence + self.random.choice(self.PARTICLES)
        return sentence

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    # Backward-compatible API used by older services/tests.
    def split(self, text: str) -> list[Sentence]:
        return self.fragment_with_particles(text, mode="chatty")

    def simplify_text(self, text: str) -> str:
        return self._clean_text(text)
