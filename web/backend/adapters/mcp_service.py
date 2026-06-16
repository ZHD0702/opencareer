"""MCP Server Service - 启动和管理 MCP 服务器"""

import asyncio
import logging
import subprocess
import sys
import socket
import os
from pathlib import Path

import httpx

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

    async def is_healthy(self) -> bool:
        """Verify that the listener speaks MCP instead of only checking the port."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "opencareer-health", "version": "1.0"},
            },
        }
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
                response = await client.post(
                    f"http://{self.host}:{self.port}/mcp",
                    json=payload,
                    headers=headers,
                )
            return response.status_code == 200 and "jsonrpc" in response.text
        except (httpx.HTTPError, asyncio.TimeoutError):
            return False
    
    async def start_server(self):
        """启动 MCP 服务器 - 使用外面的 opencareer/mcp/server.py"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            mcp_script = project_root / "opencareer" / "mcp" / "server.py"
            
            if not mcp_script.exists():
                logger.warning(f"MCP server script not found: {mcp_script}")
                return False
            
            if not self.is_port_available():
                if await self.is_healthy():
                    logger.info(f"MCP server on {self.host}:{self.port} passed protocol health check")
                    return True
                logger.warning(f"MCP port {self.host}:{self.port} is occupied but not responding")
                if self.process and self.process.poll() is None:
                    await self.stop_server()
                else:
                    return False
            
            logger.info(f"Starting MCP server on {self.host}:{self.port}")
            
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "opencareer.mcp.server",
                ],
                cwd=str(project_root),
                env={
                    **os.environ,
                    "CAREER_MCP_HOST": self.host,
                    "CAREER_MCP_PORT": str(self.port),
                },
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # 等待 MCP 服务器启动
            for i in range(10):
                await asyncio.sleep(0.5)
                if self.process.poll() is None:
                    if await self.is_healthy():
                        logger.info(f"MCP server started successfully on {self.host}:{self.port}")
                        return True
                else:
                    logger.error(f"MCP server failed to start with exit code {self.process.returncode}")
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
