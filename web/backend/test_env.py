import os
from dotenv import load_dotenv
from pathlib import Path

# 检查 CareerAgent 会从哪里加载 .env
env_path = Path('F:/opencareer/opencareer/agents/../../.env')
print(f'CareerAgent .env 路径: {env_path}')
print(f'文件存在: {env_path.exists()}')

# 检查 web/backend/.env
web_env = Path('F:/opencareer/web/backend/.env')
print(f'backend/.env 路径: {web_env}')
print(f'文件存在: {web_env.exists()}')

# 检查 web/.env
web_env2 = Path('F:/opencareer/web/.env')
print(f'web/.env 路径: {web_env2}')
print(f'文件存在: {web_env2.exists()}')

key = os.getenv('DEEPSEEK_API_KEY', '')
print(f'\n当前环境变量 DEEPSEEK_API_KEY: {key[:30]}... (共 {len(key)} 字符)')
