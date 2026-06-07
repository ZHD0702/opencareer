"""MCP Server Service - 启动和管理 MCP 服务器"""

import asyncio
import logging
import subprocess
import sys
import socket
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPService:
    """MCP 服务器服务管理"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self.process = None
    
    def is_port_available(self) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((self.host, self.port)) != 0
        except:
            return True
    
    async def start_server(self):
        """启动 MCP 服务器 - 使用外面的 opencareer/mcp/server.py"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            mcp_script = project_root / "opencareer" / "mcp" / "server.py"
            
            if not mcp_script.exists():
                logger.warning(f"MCP server script not found: {mcp_script}")
                return False
            
            if not self.is_port_available():
                logger.info(f"MCP port {self.host}:{self.port} already in use, assuming already running")
                return True
            
            logger.info(f"Starting MCP server on {self.host}:{self.port}")
            
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "opencareer.mcp.server",
                ],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # 等待 MCP 服务器启动
            for i in range(10):
                await asyncio.sleep(0.5)
                if self.process.poll() is None:
                    if not self.is_port_available():
                        logger.info(f"MCP server started successfully on {self.host}:{self.port}")
                        return True
                else:
                    stderr = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else ''
                    logger.error(f"MCP server failed to start: {stderr}")
                    return False
            
            logger.warning("MCP server may not have started properly")
            return self.process.poll() is None
                
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}", exc_info=True)
            return False
    
    async def stop_server(self):
        """停止 MCP 服务器"""
        if self.process:
            try:
                if sys.platform == 'win32':
                    self.process.kill()
                else:
                    self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                logger.info("MCP server stopped")
            except Exception as e:
                logger.error(f"Failed to stop MCP server: {e}")
    
    def is_running(self) -> bool:
        """检查 MCP 服务器是否运行中"""
        # 先检查端口是否被占用
        if not self.is_port_available():
            return True
        # 再检查我们自己的进程引用
        return self.process is not None and self.process.poll() is None


_mcp_service = None


async def get_mcp_service():
    """获取 MCP 服务实例"""
    global _mcp_service
    if _mcp_service is None:
        from careers_config import config
        _mcp_service = MCPService(
            host=config.MCP_HOST,
            port=config.MCP_PORT
        )
    return _mcp_service


async def start_mcp_server():
    """启动 MCP 服务器"""
    service = await get_mcp_service()
    return await service.start_server()


async def stop_mcp_server():
    """停止 MCP 服务器"""
    service = await get_mcp_service()
    await service.stop_server()
