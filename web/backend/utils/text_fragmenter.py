import re
from dataclasses import dataclass
from typing import List
import random
import json

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
    # 序号模式（匹配数字）
    NUMBER_PATTERN = re.compile(r'(\d+)')
    
    MAX_SENTENCE_LENGTH = 40
    MIN_SENTENCE_LENGTH = 8
    MAX_BUBBLES = 4
    MIN_BUBBLES = 2
    
    PARTICLES = ["啦", "呀", "嘛", "哦", "呢", "哈", "呢~", "啊", "嘛~", "呀~"]
    PARTICLE_PROBABILITY = 0.03
    
    EMOTION_KEYWORDS = {
        'empathy': ['理解', '明白', '感同身受', '心疼', '感觉得到', '很正常', '我懂', '没关系'],
        'excited': ['太棒了', '太好了', '恭喜', '厉害', '加油', '太棒', '真不错', '好啊', '棒'],
        'serious': ['注意', '建议', '必须', '一定', '重要', '关键', '记住', '要', '需要'],
    }
    
    def __init__(self, use_llm_segmentation: bool = True):
        self.use_llm_segmentation = use_llm_segmentation
        self._llm_adapter = None
    
    @property
    def llm_adapter(self):
        if self._llm_adapter is None and self.use_llm_segmentation:
            try:
                from llm.registry import get_llm_adapter
                self._llm_adapter = get_llm_adapter()
            except Exception as e:
                print(f"Failed to load LLM adapter: {e}")
                self.use_llm_segmentation = False
        return self._llm_adapter
    
    def split(self, text: str) -> List[Sentence]:
        if self.use_llm_segmentation and self.llm_adapter:
            try:
                sentences = self._split_with_llm(text)
                if sentences:
                    return sentences
            except Exception as e:
                print(f"LLM segmentation failed: {e}")
        
        return self._split_with_rules(text)
    
    def _split_with_llm(self, text: str) -> List[Sentence]:
        prompt = f"""请将以下文本按照自然语义边界进行断句，返回一个JSON数组，每个元素是一个完整语义的句子。

要求：
1. 每个句子必须是完整的语义单元，保持原意不变
2. 保留所有标点符号
3. 序号（如"1."、"2."等）必须放在对应句子的开头
4. 句子长度适中，不要过长或过短

文本：
{text}

请只返回JSON数组："""
        
        try:
            response = self.llm_adapter.invoke(prompt)
            result = self._parse_llm_response(response)
            
            if result:
                sentences = []
                for content in result:
                    content = content.strip()
                    if content:
                        emotion = self.detect_emotion(content)
                        delay = self.calculate_delay(content, emotion)
                        sentences.append(Sentence(content=content, emotion=emotion, delay_ms=delay))
                return sentences
        except Exception as e:
            print(f"LLM error: {e}")
        
        return []
    
    def _parse_llm_response(self, response: str) -> List[str]:
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                start = response.find('[')
                end = response.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = response[start:end+1]
                else:
                    return []
            
            result = json.loads(json_str)
            if isinstance(result, list):
                return [str(s).strip() for s in result if str(s).strip()]
        except Exception as e:
            print(f"Parse error: {e}")
        
        return []
    
    def _split_with_rules(self, text: str) -> List[Sentence]:
        sentences = []
        
        # 第一步：按序号边界拆分
        numbered_sections = self._split_by_number_boundaries(text)
        
        for section in numbered_sections:
            section = section.strip()
            if not section:
                continue
            
            # 如果是序号项，直接作为句子
            if self._is_numbered_item(section):
                emotion = self.detect_emotion(section)
                delay = self.calculate_delay(section, emotion)
                sentences.append(Sentence(content=section, emotion=emotion, delay_ms=delay))
            else:
                # 否则按主要标点拆分
                parts = self.MAIN_BOUNDARY_PATTERN.split(section)
                boundaries = [m.end() for m in self.MAIN_BOUNDARY_PATTERN.finditer(section)]
                
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    
                    punctuation = ""
                    if i < len(boundaries):
                        b_pos = boundaries[i]
                        if b_pos <= len(section):
                            punctuation = section[b_pos-1]
                    
                    full_part = part + punctuation if punctuation else part
                    emotion = self.detect_emotion(full_part)
                    delay = self.calculate_delay(full_part, emotion)
                    sentences.append(Sentence(content=full_part, emotion=emotion, delay_ms=delay))
        
        sentences = [s for s in sentences if s.content.strip()]
        
        return sentences
    
    def _split_by_number_boundaries(self, text: str) -> List[str]:
        """
        按序号边界拆分文本
        
        例如："生成简历：1 目标岗位？2.工作年限？3呀~" -> 
        ["生成简历：", "1 目标岗位？", "2.工作年限？", "3呀~"]
        """
        if not text:
            return []
        
        matches = list(self.NUMBER_PATTERN.finditer(text))
        
        if not matches:
            return [text]
        
        result = []
        last_pos = 0
        
        for i, match in enumerate(matches):
            # 获取序号前的文本
            before = text[last_pos:match.start()].strip()
            if before:
                result.append(before)
            
            # 获取序号和序号后的内容到下一个序号
            next_match = matches[i+1] if i+1 < len(matches) else None
            next_pos = next_match.start() if next_match else len(text)
            
            # 获取序号项（序号+后面的内容）
            number_item = text[match.start():next_pos].strip()
            if number_item:
                # 清理序号项：确保序号后面有适当的标点
                number_item = self._clean_numbered_item(number_item)
                result.append(number_item)
            
            last_pos = next_pos
        
        return result
    
    def _is_numbered_item(self, text: str) -> bool:
        """检查文本是否以序号开头"""
        if not text:
            return False
        return text[0].isdigit()
    
    def _clean_numbered_item(self, text: str) -> str:
        """清理序号项，确保格式正确"""
        if not text:
            return text
        
        # 找到第一个数字
        first_digit = None
        for i, char in enumerate(text):
            if char.isdigit():
                first_digit = i
                break
        
        if first_digit is None:
            return text
        
        # 获取序号部分
        number_end = first_digit + 1
        while number_end < len(text) and text[number_end].isdigit():
            number_end += 1
        
        number_str = text[first_digit:number_end]
        rest = text[number_end:]
        
        # 如果序号后面没有标点，添加句号
        if rest and rest[0] not in '.。、':
            rest = '. ' + rest
        
        return number_str + rest
    
    def detect_emotion(self, sentence: str) -> str:
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(kw in sentence for kw in keywords):
                return emotion
        return 'neutral'
    
    def calculate_delay(self, sentence: str, emotion: str) -> int:
        length = len(sentence)
        base_delay = length * 80
        
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
        if not sentence:
            return sentence
        
        ends_with_particle = any(sentence.endswith(p) for p in self.PARTICLES)
        if ends_with_particle:
            return sentence
        
        if sentence[-1] in '。？！.!?~':
            if random.random() < self.PARTICLE_PROBABILITY:
                sentence = sentence[:-1] + random.choice(self.PARTICLES)
        elif random.random() < self.PARTICLE_PROBABILITY:
            sentence = sentence + random.choice(self.PARTICLES)
        
        return sentence
    
    def merge_sentences_to_limit(self, sentences: List[Sentence]) -> List[Sentence]:
        if len(sentences) <= self.MAX_BUBBLES:
            return sentences
        
        # 检查哪些句子是序号项（以数字开头）
        numbered_indices = []
        for i, sentence in enumerate(sentences):
            content = sentence.content.strip()
            if content and (content[0].isdigit() or (len(content) > 1 and content[1].isdigit())):
                numbered_indices.append(i)
        
        result = []
        current_content = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            # 如果当前是序号项，先处理前面的内容，然后把序号项单独作为一个句子
            if i in numbered_indices:
                # 先处理前面非序号项的内容
                if current_content:
                    merged_content = ''.join(current_content)
                    result.append(Sentence(
                        content=merged_content,
                        emotion='neutral',
                        delay_ms=self.calculate_delay(merged_content, 'neutral')
                    ))
                    current_content = []
                    current_length = 0
                
                # 序号项单独处理
                result.append(sentence)
            else:
                # 非序号项，尝试合并
                current_content.append(sentence.content)
                current_length += len(sentence.content)
        
        # 处理剩余内容
        if current_content:
            merged_content = ''.join(current_content)
            result.append(Sentence(
                content=merged_content,
                emotion='neutral',
                delay_ms=self.calculate_delay(merged_content, 'neutral')
            ))
        
        # 如果还是超过最大数量，对非序号项进行合并
        if len(result) > self.MAX_BUBBLES:
            # 只对非序号项进行合并
            final_result = []
            non_numbered_content = []
            for sentence in result:
                is_numbered = False
                content = sentence.content.strip()
                if content and (content[0].isdigit() or (len(content) > 1 and content[1].isdigit())):
                    is_numbered = True
                
                if is_numbered:
                    # 先处理前面的非序号项
                    if non_numbered_content:
                        merged = ''.join(non_numbered_content)
                        final_result.append(Sentence(
                            content=merged,
                            emotion='neutral',
                            delay_ms=self.calculate_delay(merged, 'neutral')
                        ))
                        non_numbered_content = []
                    final_result.append(sentence)
                else:
                    non_numbered_content.append(sentence.content)
            
            # 处理最后的非序号项
            if non_numbered_content:
                merged = ''.join(non_numbered_content)
                final_result.append(Sentence(
                    content=merged,
                    emotion='neutral',
                    delay_ms=self.calculate_delay(merged, 'neutral')
                ))
            
            return final_result
        
        return result
    
    def simplify_text(self, text: str) -> str:
        lines = text.split('\n')
        seen_lines = set()
        unique_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                seen_lines.add(line)
                unique_lines.append(line)
        
        text = '\n'.join(unique_lines)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def fragment_with_particles(self, text: str) -> List[Sentence]:
        simplified_text = self.simplify_text(text)
        sentences = self.split(simplified_text)
        sentences = self.merge_sentences_to_limit(sentences)
        
        result = []
        for sentence in sentences:
            content = self.inject_particle(sentence.content)
            result.append(Sentence(
                content=content,
                emotion=sentence.emotion,
                delay_ms=sentence.delay_ms
            ))
        
        return result