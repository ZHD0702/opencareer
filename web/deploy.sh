#!/bin/bash
# OpenCareer 一键部署脚本 (Linux)
# 使用方法: chmod +x deploy.sh && ./deploy.sh

set -e

echo "================================================"
echo "  OpenCareer 一键部署脚本"
echo "================================================"
echo ""

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 1. 安装 Docker
echo "[1/5] 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "安装 Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    echo "安装 Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

echo "✅ Docker 已就绪"

# 2. 创建项目目录
echo ""
echo "[2/5] 创建项目目录..."
PROJECT_DIR="/var/www/opencareer"
if [ ! -d "$PROJECT_DIR" ]; then
    sudo mkdir -p $PROJECT_DIR
    echo "⚠️  请将项目文件复制到 $PROJECT_DIR"
    echo "   然后重新运行此脚本"
    exit 1
fi

cd $PROJECT_DIR

# 3. 配置环境变量
echo ""
echo "[3/5] 配置环境变量..."
if [ ! -f ".env" ]; then
    sudo cp web/.env.example .env
    echo "⚠️  请编辑 .env 文件填入你的 API 密钥"
    echo "   nano $PROJECT_DIR/.env"
    exit 1
fi

# 4. 启动服务
echo ""
echo "[4/5] 启动服务..."
cd $PROJECT_DIR/web

# 构建并启动
docker-compose down 2>/dev/null || true
docker-compose up -d --build

# 5. 验证部署
echo ""
echo "[5/5] 验证部署..."
sleep 5

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 后端启动成功"
else
    echo "❌ 后端启动失败，请检查日志"
    docker-compose logs backend
    exit 1
fi

if curl -f http://localhost:8001 > /dev/null 2>&1; then
    echo "✅ MCP 服务器启动成功"
else
    echo "❌ MCP 服务器启动失败，请检查日志"
    docker-compose logs mcp-server
    exit 1
fi

echo ""
echo "================================================"
echo "  部署完成！"
echo "================================================"
echo ""
echo "📍 访问地址："
echo "   前端界面: http://$(hostname -I | awk '{print $1}'):5173"
echo "   后端 API:  http://$(hostname -I | awk '{print $1}'):8000"
echo "   API 文档:  http://$(hostname -I | awk '{print $1}'):8000/docs"
echo ""
echo "📝 常用命令："
echo "   查看日志:  docker-compose logs -f"
echo "   停止服务:  docker-compose down"
echo "   重启服务:  docker-compose restart"
echo ""
echo "⚠️  首次部署请确保 .env 文件中的 DEEPSEEK_API_KEY 已正确配置"
echo "================================================"
