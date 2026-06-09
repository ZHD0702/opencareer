"""
MCP (Model Context Protocol) Server for OpenCareer.

This module implements an MCP server that dynamically loads SKILLs
and makes them available as tools for LangChain agents.
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..skills.skill_registry import get_global_registry, SkillRegistry
from ..skills.base_skill import BaseSkill, SkillMetadata
from .skill_loader import SkillLoader
from .tools import SkillTool, create_langchain_tool_from_skill


# Pydantic models for API
class SkillExecuteRequest(BaseModel):
    """Request model for SKILL execution."""
    skill_name: str = Field(..., description="Name of the SKILL to execute")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the SKILL")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context information")


class SkillRegisterRequest(BaseModel):
    """Request model for SKILL registration."""
    skill_metadata: Dict[str, Any] = Field(..., description="SKILL metadata")
    skill_module: Optional[str] = Field(None, description="Python module containing the SKILL")
    skill_class: Optional[str] = Field(None, description="SKILL class name")


class SkillInfoResponse(BaseModel):
    """Response model for SKILL information."""
    name: str
    version: str
    description: str
    category: str
    tags: List[str]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    initialized: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str
    skills_loaded: int
    skills_healthy: int


class MCPServer:
    """MCP Server for OpenCareer."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        skills_dir: Optional[Path] = None,
        auto_load_skills: bool = True
    ):
        """Initialize MCP Server.

        Args:
            host: Host to bind to
            port: Port to listen on
            skills_dir: Directory containing SKILLs
            auto_load_skills: Whether to automatically load SKILLs on startup
        """
        self.host = host
        self.port = port
        self.skills_dir = skills_dir or Path(__file__).parent.parent / "skills"
        self.auto_load_skills = auto_load_skills
        self.logger = logging.getLogger("mcp.server")

        # Initialize components
        self.registry = get_global_registry()
        self.skill_loader = SkillLoader(self.skills_dir)
        self.langchain_tools: List[SkillTool] = []

        # FastAPI app
        self.app = FastAPI(
            title="OpenCareer MCP Server",
            description="Model Context Protocol Server for OpenCareer SKILLs",
            version="0.1.0"
        )

        # Configure middleware
        self._configure_middleware()
        # Configure routes
        self._configure_routes()

    def _configure_middleware(self) -> None:
        """Configure FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            self.logger.info(f"{request.method} {request.url.path}")
            response = await call_next(request)
            return response

    def _configure_routes(self) -> None:
        """Configure FastAPI routes."""

        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "OpenCareer MCP Server",
                "version": "0.1.0",
                "endpoints": {
                    "health": "/health",
                    "skills": "/skills",
                    "execute": "/execute",
                    "tools": "/tools"
                }
            }

        @self.app.get("/health")
        async def health() -> HealthResponse:
            """Health check endpoint."""
            health_status = await self.registry.health_check()

            return HealthResponse(
                status="healthy" if health_status["healthy_skills"] > 0 else "degraded",
                version="0.1.0",
                timestamp=health_status.get("timestamp", ""),
                skills_loaded=health_status["total_skills"],
                skills_healthy=health_status["healthy_skills"]
            )

        @self.app.get("/skills")
        async def list_skills() -> List[SkillInfoResponse]:
            """List all available SKILLs."""
            skills_info = []

            for skill_info in self.registry.list_skills():
                skill = self.registry.get(skill_info["name"])
                if skill:
                    skills_info.append(SkillInfoResponse(
                        name=skill.metadata.name,
                        version=skill.metadata.version,
                        description=skill.metadata.description,
                        category=skill.metadata.category.value,
                        tags=skill.metadata.tags,
                        input_schema=skill.metadata.input_schema,
                        output_schema=skill.metadata.output_schema,
                        initialized=skill._initialized
                    ))

            return skills_info

        @self.app.get("/skills/{skill_name}")
        async def get_skill(skill_name: str) -> SkillInfoResponse:
            """Get information about a specific SKILL."""
            skill = self.registry.get(skill_name)
            if not skill:
                raise HTTPException(status_code=404, detail=f"SKILL not found: {skill_name}")

            return SkillInfoResponse(
                name=skill.metadata.name,
                version=skill.metadata.version,
                description=skill.metadata.description,
                category=skill.metadata.category.value,
                tags=skill.metadata.tags,
                input_schema=skill.metadata.input_schema,
                output_schema=skill.metadata.output_schema,
                initialized=skill._initialized
            )

        @self.app.post("/execute")
        async def execute_skill(request: SkillExecuteRequest) -> Dict[str, Any]:
            """Execute a SKILL."""
            try:
                result = await self.registry.execute(
                    skill_name=request.skill_name,
                    input_data=request.input_data,
                    context=request.context
                )
                return result
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except (ValueError, RuntimeError) as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Unexpected error executing SKILL: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/skills/register")
        async def register_skill(request: SkillRegisterRequest) -> Dict[str, Any]:
            """Register a new SKILL."""
            try:
                # Create metadata
                metadata = SkillMetadata.from_dict(request.skill_metadata)

                # Load SKILL from module if provided
                skill = None
                if request.skill_module:
                    skill = self.skill_loader.load_skill_from_module(
                        request.skill_module,
                        request.skill_class
                    )

                if not skill:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not load SKILL from module"
                    )

                # Register the SKILL
                self.registry.register(skill)

                # Create LangChain tool
                tool = create_langchain_tool_from_skill(skill)
                self.langchain_tools.append(tool)

                return {
                    "status": "success",
                    "skill": skill.metadata.name,
                    "version": skill.metadata.version,
                    "tool_created": True
                }

            except Exception as e:
                self.logger.error(f"Error registering SKILL: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/tools")
        async def list_tools() -> List[Dict[str, Any]]:
            """List all available LangChain tools."""
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": str(tool.args_schema) if hasattr(tool, "args_schema") else None,
                    "skill_source": getattr(tool, "skill_name", "unknown")
                }
                for tool in self.langchain_tools
            ]

        @self.app.post("/reload")
        async def reload_skills() -> Dict[str, Any]:
            """Reload all SKILLs from disk."""
            try:
                # Clean up existing SKILLs
                await self.registry.cleanup_all()

                # Clear registry
                for skill_name in list(self.registry.skills.keys()):
                    self.registry.unregister(skill_name)

                # Clear tools
                self.langchain_tools.clear()

                # Reload SKILLs
                await self.load_skills()

                return {
                    "status": "success",
                    "skills_loaded": len(self.registry.skills),
                    "tools_created": len(self.langchain_tools)
                }

            except Exception as e:
                self.logger.error(f"Error reloading SKILLs: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def load_skills(self) -> None:
        """Load SKILLs from the skills directory."""
        self.logger.info(f"Loading SKILLs from: {self.skills_dir}")

        if not self.skills_dir.exists():
            self.logger.warning(f"Skills directory does not exist: {self.skills_dir}")
            return

        # Discover and load SKILLs
        skill_modules = self.skill_loader.discover_skills()

        for module_info in skill_modules:
            try:
                skill = self.skill_loader.load_skill(module_info["module_path"])
                if skill:
                    self.registry.register(skill)
                    self.logger.info(f"Loaded SKILL: {skill.metadata.name}")

                    # Create LangChain tool
                    tool = create_langchain_tool_from_skill(skill)
                    self.langchain_tools.append(tool)
                    self.logger.debug(f"Created tool for SKILL: {skill.metadata.name}")

            except Exception as e:
                self.logger.error(f"Failed to load SKILL {module_info['name']}: {e}")

        # Initialize all loaded SKILLs
        await self.registry.initialize_all()

        self.logger.info(f"Loaded {len(self.registry.skills)} SKILLs, "
                        f"created {len(self.langchain_tools)} tools")

    def get_langchain_tools(self) -> List[SkillTool]:
        """Get all LangChain tools.

        Returns:
            List of LangChain tools
        """
        return self.langchain_tools

    async def execute_skill(self, skill_name: str, input_data: Dict[str, Any] = None,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a SKILL. Convenience method for testing.

        Args:
            skill_name: Name of the SKILL to execute
            input_data: Input data for the SKILL
            context: Additional context information

        Returns:
            SKILL execution result
        """
        if input_data is None:
            input_data = {}
        if context is None:
            context = {}

        try:
            return await self.registry.execute(skill_name, input_data, context)
        except Exception as e:
            self.logger.error(f"Error executing SKILL {skill_name}: {e}")
            raise

    def get_available_skills(self) -> List[Dict[str, Any]]:
        """Get all available SKILLs. Convenience method for testing.

        Returns:
            List of SKILL information dictionaries
        """
        return self.registry.list_skills()

    async def start(self) -> None:
        """Start the MCP server."""
        # Load SKILLs if enabled
        if self.auto_load_skills:
            await self.load_skills()

        # Start the server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)

        self.logger.info(f"Starting MCP server on {self.host}:{self.port}")
        await server.serve()

    async def stop(self) -> None:
        """Stop the MCP server."""
        # Clean up SKILLs
        await self.registry.cleanup_all()
        self.logger.info("MCP server stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    logger = logging.getLogger("mcp.server")
    logger.info("Starting MCP server...")

    # Create and load server instance
    server = MCPServer()
    await server.load_skills()

    # Store server instance in app state
    app.state.server = server

    yield

    # Shutdown
    logger.info("Shutting down MCP server...")
    await server.stop()


def create_app() -> FastAPI:
    """Create FastAPI application with lifespan.

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="OpenCareer MCP Server",
        description="Model Context Protocol Server for OpenCareer SKILLs",
        version="0.1.0",
        lifespan=lifespan
    )

    # Simple routes for compatibility
    @app.get("/")
    async def root():
        return {"message": "OpenCareer MCP Server"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


async def main():
    """Main entry point for MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and start server
    server = MCPServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())