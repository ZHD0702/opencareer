# OpenCareer 生产环境部署指南

## 方式一：Docker 部署 (推荐)

### 前置要求
- Docker 20.10+
- Docker Compose 2.0+
- Git

### 部署步骤

#### 1. 拉取代码
```bash
git clone <your-repo-url> opencareer
cd opencareer/web
```

#### 2. 配置环境变量
```bash
# 复制环境变量配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
nano .env
```

**必须配置的值：**
```env
DEEPSEEK_API_KEY=your_actual_deepseek_api_key
CAREER_USE_MCP=true
LLM_PROVIDER=deepseek
```

#### 3. 启动服务
```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

#### 4. 验证部署
```bash
# 检查后端健康状态
curl http://localhost:8000/health

# 检查 MCP 服务
curl http://localhost:8001/mcp
```

#### 5. 访问应用
- 前端界面：http://your-server-ip:5173
- 后端 API：http://your-server-ip:8000
- API 文档：http://your-server-ip:8000/docs

### 服务管理命令

```bash
# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 更新代码后重新部署
git pull
docker-compose up -d --build

# 查看实时日志
docker-compose logs -f backend
docker-compose logs -f mcp-server
```

---

## 方式二：直接部署到 Linux 服务器

### 前置要求
- Ubuntu 20.04+ / Debian 11+
- Python 3.10+
- Nginx
- Git

### 部署步骤

#### 1. 安装依赖
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 和依赖
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y nginx certbot python3-certbot-nginx

# 安装 Node.js (用于前端)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

#### 2. 部署后端
```bash
# 创建项目目录
sudo mkdir -p /var/www/opencareer
cd /var/www/opencareer

# 克隆代码
sudo git clone <your-repo-url> .

# 创建虚拟环境
cd web/backend
python3.11 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp ../../.env.example .env
nano .env  # 填入你的 API 密钥
```

#### 3. 配置后端服务 (systemd)
```bash
sudo nano /etc/systemd/system/opencareer-backend.service
```

写入以下内容：
```ini
[Unit]
Description=OpenCareer Backend API
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/opencareer/web/backend
Environment="PATH=/var/www/opencareer/web/backend/venv/bin"
ExecStart=/var/www/opencareer/web/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

启动后端服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start opencareer-backend
sudo systemctl enable opencareer-backend
```

#### 4. 部署前端
```bash
cd /var/www/opencareer/web/front

# 安装依赖并构建
npm install
npm run build

# 创建前端服务
sudo nano /etc/systemd/system/opencareer-frontend.service
```

写入以下内容：
```ini
[Unit]
Description=OpenCareer Frontend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/opencareer/web/front
ExecStart=/usr/bin/npx serve -s dist -l 5173
Restart=always

[Install]
WantedBy=multi-user.target
```

启动前端服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start opencareer-frontend
sudo systemctl enable opencareer-frontend
```

#### 5. 配置 Nginx 反向代理
```bash
sudo nano /etc/nginx/sites-available/opencareer
```

写入以下内容：
```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名

    # 前端
    location / {
        proxy_pass http://127.0.0.1:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # 后端 API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

启用站点：
```bash
sudo ln -s /etc/nginx/sites-available/opencareer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. 配置 HTTPS (Let's Encrypt)
```bash
sudo certbot --nginx -d your-domain.com
```

#### 7. 配置防火墙
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## 方式三：Docker + Nginx + HTTPS (生产推荐)

### 架构
```
用户 -> Nginx (443) -> 前端 (5173)
                      -> 后端 (8000)
                      -> MCP (8001)
```

### Nginx 配置示例
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # 前端
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # 后端 API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 常见问题

### 1. MCP 服务启动失败
```bash
# 检查 MCP 容器日志
docker-compose logs mcp-server

# 常见原因：
# - 端口 8001 被占用
# - API 密钥未设置
# - Python 包安装失败
```

### 2. 后端无法连接 MCP
```bash
# 检查网络连接
docker-compose exec backend ping mcp-server

# 检查环境变量
docker-compose exec backend env | grep MCP
```

### 3. 前端无法连接后端
```bash
# 检查 CORS 配置
# 确保后端的 CORS 设置允许前端域名

# 检查 API 地址配置
# 确保 VITE_API_BASE_URL 正确
```

### 4. 数据库权限问题
```bash
# 修复权限
sudo chown -R www-data:www-data /var/www/opencareer/web/backend/data
```

---

## 性能优化建议

### 1. 使用 PM2 管理进程
```bash
npm install -g pm2
pm2 start backend/start.sh --name opencareer-backend
pm2 save
pm2 startup
```

### 2. 配置 Redis 缓存 (可选)
```yaml
# docker-compose.yml 添加
redis:
  image: redis:alpine
  ports:
    - "6379:6379"
```

### 3. 配置 CDN 加速静态文件
- 将前端构建产物上传到 CDN
- 减少服务器带宽压力

---

## 监控和日志

### 日志管理
```bash
# 配置日志轮转
sudo nano /etc/logrotate.d/opencareer
```

写入：
```
/var/log/opencareer/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
}
```

### 监控脚本
```bash
#!/bin/bash
# check_health.sh
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "OK"
    exit 0
else
    echo "FAILED"
    exit 1
fi
```

---

## 备份策略

### 数据库备份
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/opt/backups/opencareer

# 备份 SQLite 数据库
cp /var/www/opencareer/web/backend/data/app.db $BACKUP_DIR/db_$DATE.db

# 保留最近 30 天的备份
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
```

添加到 crontab：
```bash
crontab -e
# 每天凌晨 3 点备份
0 3 * * * /opt/scripts/backup.sh
```

---

## 安全建议

1. **使用环境变量存储密钥**，不要硬编码
2. **配置 HTTPS**，始终使用加密连接
3. **限制 API 访问频率**，防止滥用
4. **定期更新依赖**，修复安全漏洞
5. **配置防火墙**，只开放必要端口
6. **使用非 root 用户运行服务**
