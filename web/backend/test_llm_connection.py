import sys
sys.path.insert(0, '.')
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载和后端一样的 env
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
print(f'项目根目录 .env: {_env_path} (存在: {_env_path.exists()})')
if _env_path.exists():
    load_dotenv(_env_path)
load_dotenv()

key = os.getenv('DEEPSEEK_API_KEY', '')
print(f'\nDEEPSEEK_API_KEY: {key[:30]}... (长度: {len(key)})')

# 测试网络连接
print('\n--- 测试 DeepSeek API 连接 ---')
try:
    import requests
    r = requests.get('https://api.deepseek.com/v1/models',
                     headers={'Authorization': f'Bearer {key}'},
                     timeout=15)
    print(f'状态码: {r.status_code}')
    if r.status_code == 200:
        print('✅ 连接成功!')
        data = r.json()
        print(f'可用模型数: {len(data.get("data", []))}')
    else:
        print(f'❌ 请求失败: {r.text[:200]}')
except Exception as e:
    print(f'❌ 网络错误: {type(e).__name__}: {e}')
