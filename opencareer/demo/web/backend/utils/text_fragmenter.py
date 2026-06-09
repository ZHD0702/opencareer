import re
from dataclasses import dataclass
from typing import List
import random

@dataclass
class Sentence:
    content: str
    emotion: str
    delay_ms: int

class TextFragmenter:
    # 主要断句标点
    MAIN_BOUNDARY_PATTERN = re.compile(r'[。？！\.！？]+')
    # 次要断句标点
    SECONDARY_BOUNDARY_PATTERN = re.compile(r'[，；,;]+')
    # 最大句子长度
    MAX_SENTENCE_LENGTH = 20
    
    EMOTION_KEYWORDS = {
        'empathy': ['理解', '明白', '感同身受', '心疼', '感觉得到', '很正常', '我懂', '没关系'],
        'excited': ['太棒了', '太好了', '恭喜', '厉害', '加油', '太棒', '真不错', '好啊', '棒'],
        'serious': ['注意', '建议', '必须', '一定', '重要', '关键', '记住', '要', '需要'],
    }
    
    def split(self, text: str) -> List[Sentence]:
        sentences = []
        remaining = text.strip()
        
        if not remaining:
            return []
        
        # 第一阶段：按主要标点拆分
        main_parts = self.MAIN_BOUNDARY_PATTERN.split(remaining)
        main_boundaries = [m.end() for m in self.MAIN_BOUNDARY_PATTERN.finditer(remaining)]
        
        current_pos = 0
        for i, part in enumerate(main_parts):
            part = part.strip()
            if not part:
                continue
            
            # 添加标点
            punctuation = ""
            if i < len(main_boundaries):
                b_pos = main_boundaries[i]
                if b_pos <= len(remaining):
                    punctuation = remaining[b_pos-1]
            
            # 第二阶段：如果句子太长，按次要标点继续拆分
            if len(part) > self.MAX_SENTENCE_LENGTH:
                sub_sentences = self._split_by_secondary(part)
                # 给最后一个子句子加上原本的标点
                if sub_sentences and punctuation:
                    sub_sentences[-1] = sub_sentences[-1] + punctuation
                
                for sub in sub_sentences:
                    if sub.strip():
                        emotion = self.detect_emotion(sub)
                        delay = self.calculate_delay(sub, emotion)
                        sentences.append(Sentence(content=sub, emotion=emotion, delay_ms=delay))
            else:
                # 正常长度的句子
                full_part = part + punctuation if punctuation else part
                emotion = self.detect_emotion(full_part)
                delay = self.calculate_delay(full_part, emotion)
                sentences.append(Sentence(content=full_part, emotion=emotion, delay_ms=delay))
        
        # 如果没有拆分出任何句子，返回原文本
        if not sentences:
            emotion = self.detect_emotion(text)
            delay = self.calculate_delay(text, emotion)
            sentences = [Sentence(content=text, emotion=emotion, delay_ms=delay)]
        
        # 过滤掉空句子
        sentences = [s for s in sentences if s.content.strip()]
        
        return sentences
    
    def _split_by_secondary(self, text: str) -> List[str]:
        """按次要标点进一步拆分"""
        parts = []
        remaining = text
        
        while remaining:
            # 查找下一个次要标点
            match = self.SECONDARY_BOUNDARY_PATTERN.search(remaining)
            
            if match:
                # 取前面的部分（包含标点）
                end_pos = match.end()
                part = remaining[:end_pos].strip()
                if part:
                    parts.append(part)
                remaining = remaining[end_pos:].strip()
            else:
                # 没有标点了，检查长度
                if len(remaining) <= self.MAX_SENTENCE_LENGTH:
                    if remaining:
                        parts.append(remaining)
                    break
                else:
                    # 还是太长，强制截断
                    first_part = remaining[:self.MAX_SENTENCE_LENGTH].strip()
                    if first_part:
                        parts.append(first_part)
                    remaining = remaining[self.MAX_SENTENCE_LENGTH:].strip()
        
        return parts
    
    def detect_emotion(self, sentence: str) -> str:
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(kw in sentence for kw in keywords):
                return emotion
        return 'neutral'
    
    def calculate_delay(self, sentence: str, emotion: str) -> int:
        length = len(sentence)
        base_delay = length * 80  # 稍微延长基础延迟
        
        multipliers = {
            'empathy': 1.5,
            'excited': 0.7,
            'serious': 1.2,
            'neutral': 1.0,
        }
        
        delay = base_delay * multipliers.get(emotion, 1.0)
        jitter = 0.8 + random.random() * 0.4
        delay *= jitter
        
        return int(max(400, min(3000, delay)))
