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
    MAX_SENTENCE_LENGTH = 40
    # 最小句子长度
    MIN_SENTENCE_LENGTH = 8
    # 最大气泡数量
    MAX_BUBBLES = 4
    # 最小气泡数量
    MIN_BUBBLES = 2
    
    # 语气词列表
    PARTICLES = ["啦", "呀", "嘛", "哦", "呢", "哈", "呢~", "啊", "嘛~", "呀~"]
    
    # 语气词注入概率
    PARTICLE_PROBABILITY = 0.20
    
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
    
    def inject_particle(self, sentence: str) -> str:
        """
        随机给句子添加语气词
        仅在句尾添加，不会破坏句子完整性
        """
        # 不对已经以标点结尾的句子添加语气词
        if sentence and sentence[-1] in '。？！.!?~':
            # 替换末尾标点为语气词
            sentence = sentence[:-1] + random.choice(self.PARTICLES)
        elif random.random() < self.PARTICLE_PROBABILITY:
            # 在末尾添加语气词
            particle = random.choice(self.PARTICLES)
            sentence = sentence + particle
        
        return sentence
    
    def merge_sentences_to_limit(self, sentences: List[Sentence]) -> List[Sentence]:
        """
        合并句子，限制气泡数量在 MIN_BUBBLES 到 MAX_BUBBLES 之间
        """
        if len(sentences) <= self.MAX_BUBBLES:
            return sentences
        
        # 计算需要合并成多少个气泡
        target_count = self.MAX_BUBBLES
        
        # 计算每个气泡的目标字符数
        total_chars = sum(len(s.content) for s in sentences)
        chars_per_bubble = total_chars / target_count
        
        result = []
        current_content = []
        current_length = 0
        current_emotion = 'neutral'
        
        for sentence in sentences:
            if current_content and (current_length + len(sentence.content) > chars_per_bubble * 1.2):
                # 当前气泡已满，合并并加入结果
                merged_content = ''.join(current_content)
                emotion = current_emotion if current_emotion != 'neutral' else sentence.emotion
                result.append(Sentence(
                    content=merged_content,
                    emotion=emotion,
                    delay_ms=self.calculate_delay(merged_content, emotion)
                ))
                current_content = [sentence.content]
                current_length = len(sentence.content)
                current_emotion = sentence.emotion
            else:
                # 继续合并
                current_content.append(sentence.content)
                current_length += len(sentence.content)
                if current_emotion == 'neutral':
                    current_emotion = sentence.emotion
        
        # 处理剩余的内容
        if current_content:
            merged_content = ''.join(current_content)
            result.append(Sentence(
                content=merged_content,
                emotion=current_emotion,
                delay_ms=self.calculate_delay(merged_content, current_emotion)
            ))
        
        return result
    
    def simplify_text(self, text: str) -> str:
        """
        简化文本，删除冗余信息
        """
        # 1. 移除重复的句子
        lines = text.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
        
        text = '\n'.join(unique_lines)
        
        # 2. 移除冗余的表达
        redundant_phrases = [
            '我认为', '我觉得', '在我看来', '我想', '可以说', '也就是说',
            '总的来说', '简单来说', '综上所述', '简单概括',
            '首先', '其次', '最后', '第一', '第二', '第三',
            '的是', '非常', '很', '特别', '极其', '真的',
        ]
        
        for phrase in redundant_phrases:
            text = text.replace(phrase, '')
        
        # 3. 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def fragment_with_particles(self, text: str) -> List[Sentence]:
        """
        碎片化文本并注入语气词
        返回碎片列表（包含语气词），限制气泡数量
        """
        # 先简化文本
        simplified_text = self.simplify_text(text)
        
        sentences = self.split(simplified_text)
        
        # 合并句子到目标数量
        sentences = self.merge_sentences_to_limit(sentences)
        
        # 对每个碎片随机注入语气词
        result = []
        for sentence in sentences:
            content = self.inject_particle(sentence.content)
            result.append(Sentence(
                content=content,
                emotion=sentence.emotion,
                delay_ms=sentence.delay_ms
            ))
        
        return result
