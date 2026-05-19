# opencareer/mcp/skills/__init__.py
class SkillRegistry:
    def _load_skills(self):
        """扫描 skills/ 目录下的 YAML 和 SKILL.md"""
        skills = {}
        
        # 加载旧的 skills.yaml
        skills.update(self._load_yaml_skills())
        
        # 扫描每个文件夹的 SKILL.md
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                md_file = skill_dir / "SKILL.md"
                if md_file.exists():
                    skills.update(self._load_markdown_skill(md_file))
        
        return skills