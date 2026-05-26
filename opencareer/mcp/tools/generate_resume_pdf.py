#!/usr/bin/env python3
"""
中文简历 PDF 生成器
- 支持 chinese / bilingual 两种输出格式
- 优先注册可用中文字体，确保中文不乱码
- 生成文本型 PDF，便于 ATS 解析与复制
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def register_cjk_fonts():
    """优先使用系统中文字体，失败时回退到 STSong-Light。"""
    candidates = [
        ('CN-Regular', r'C:\Windows\Fonts\msyh.ttf'),
        ('CN-Regular', r'C:\Windows\Fonts\simsun.ttc'),
        ('CN-Bold', r'C:\Windows\Fonts\msyhbd.ttf'),
        ('CN-Bold', r'C:\Windows\Fonts\simhei.ttf'),
    ]

    regular = None
    bold = None

    for font_name, font_path in candidates:
        if not os.path.exists(font_path):
            continue
        try:
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            if font_name == 'CN-Regular' and regular is None:
                regular = font_name
            if font_name == 'CN-Bold' and bold is None:
                bold = font_name
        except Exception:
            continue

    if regular is None:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        regular = 'STSong-Light'

    if bold is None:
        bold = regular

    return regular, bold


class ResumePDFBuilder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.font_regular, self.font_bold = register_cjk_fonts()
        self.styles = self._build_styles()
        self.story = []

    def _build_styles(self):
        return {
            'name': ParagraphStyle(
                'name',
                fontName=self.font_bold,
                fontSize=18,
                leading=22,
                alignment=TA_CENTER,
                spaceAfter=6,
            ),
            'title': ParagraphStyle(
                'title',
                fontName=self.font_regular,
                fontSize=11,
                leading=14,
                alignment=TA_CENTER,
                spaceAfter=8,
            ),
            'contact': ParagraphStyle(
                'contact',
                fontName=self.font_regular,
                fontSize=10,
                leading=14,
                alignment=TA_CENTER,
                spaceAfter=10,
            ),
            'section': ParagraphStyle(
                'section',
                fontName=self.font_bold,
                fontSize=12,
                leading=16,
                alignment=TA_LEFT,
                spaceBefore=8,
                spaceAfter=6,
            ),
            'heading': ParagraphStyle(
                'heading',
                fontName=self.font_bold,
                fontSize=10.5,
                leading=14,
                alignment=TA_LEFT,
                spaceBefore=4,
                spaceAfter=3,
            ),
            'body': ParagraphStyle(
                'body',
                fontName=self.font_regular,
                fontSize=10,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=3,
            ),
            'bullet': ParagraphStyle(
                'bullet',
                fontName=self.font_regular,
                fontSize=10,
                leading=14,
                leftIndent=10,
                firstLineIndent=-8,
                spaceAfter=2,
            ),
        }

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ''
        return str(value)

    def _p(self, text: Any, style: str):
        safe_text = self._as_text(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.story.append(Paragraph(safe_text, self.styles[style]))

    def _sp(self, h=4):
        self.story.append(Spacer(1, h))

    def add_contact_block(self, contact: dict):
        self._p(contact.get('name', ''), 'name')
        if contact.get('title'):
            self._p(contact['title'], 'title')

        parts = [
            contact.get('location', ''),
            contact.get('phone', ''),
            contact.get('email', ''),
        ]
        links = contact.get('links', [])
        if isinstance(links, list):
            parts.extend(self._as_text(item) for item in links)
        line = ' | '.join([self._as_text(p).strip() for p in parts if self._as_text(p).strip()])
        if line:
            self._p(line, 'contact')

    def add_summary(self, text: str, title='职业摘要'):
        if not text:
            return
        self._p(title, 'section')
        self._p(text, 'body')

    def add_sections(self, sections: list):
        for section in sections or []:
            if not isinstance(section, dict):
                continue

            section_title = self._as_text(section.get('title', '')).strip()
            if section_title:
                self._p(section_title, 'section')

            for item in section.get('items', []):
                if not isinstance(item, dict):
                    continue

                heading = self._as_text(item.get('heading', '')).strip()
                if heading:
                    self._p(heading, 'heading')

                for bullet in item.get('bullets', []):
                    bullet_text = self._as_text(bullet).strip()
                    if bullet_text:
                        self._p('• ' + bullet_text, 'bullet')

                text = self._as_text(item.get('text', '')).strip()
                if text:
                    self._p(text, 'body')

    def add_bilingual(self, data: dict):
        zh = data.get('zh', {})
        en = data.get('en', {})

        self._p('中文简历', 'section')
        self.add_summary(zh.get('summary', ''), title='职业摘要')
        self.add_sections(zh.get('sections', []))

        self._sp(10)
        self._p('English Resume', 'section')
        self.add_summary(en.get('summary', ''), title='Professional Summary')
        self.add_sections(en.get('sections', []))

    def build(self, data: dict):
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title='Resume',
            author=data.get('contact', {}).get('name', 'Candidate'),
        )

        self.add_contact_block(data.get('contact', {}))

        fmt = (data.get('format') or 'chinese').lower()
        if fmt == 'bilingual':
            self.add_bilingual(data)
        else:
            self.add_summary(data.get('summary', ''), title='职业摘要')
            self.add_sections(data.get('sections', []))

        doc.build(self.story)


def validate_payload(data: dict):
    if not isinstance(data, dict):
        raise ValueError('输入 JSON 顶层必须是对象。')
    if 'contact' not in data:
        raise ValueError('缺少 contact 字段。')
    if not isinstance(data.get('contact'), dict):
        raise ValueError('contact 必须是对象。')
    if 'sections' in data and not isinstance(data.get('sections'), list):
        raise ValueError('sections 必须是列表。')


def main():
    parser = argparse.ArgumentParser(description='生成中文 ATS 友好简历 PDF')
    parser.add_argument('--input', '-i', required=True, help='输入 JSON 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出 PDF 文件路径')
    parser.add_argument('--format', '-f', choices=['chinese', 'bilingual'], default=None, help='输出格式')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)

    if args.format:
        data['format'] = args.format

    validate_payload(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder = ResumePDFBuilder(str(output_path))
    builder.build(data)
    print(f'OK: PDF generated at {output_path}')


if __name__ == '__main__':
    main()
