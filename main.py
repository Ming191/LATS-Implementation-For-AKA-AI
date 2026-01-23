"""
LATS Test Generation Server - Main FastAPI Application

A unified test generation server using Language Agent Tree Search (LATS)
with Monte Carlo Tree Search for C++ MC/DC coverage.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import lats_router
from core.config import settings

# Create FastAPI application
app = FastAPI(
    title="LATS Test Generation API",
    description="Language Agent Tree Search for automated C++ test generation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS (allow Java backend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(lats_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "LATS Test Generation API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/lats/health",
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("LATS Test Generation Server Starting")
    print("=" * 60)
    print(f"Host: {settings.host}:{settings.port}")
    print(f"Docs: http://{settings.host}:{settings.port}/docs")
    print(f"LLM: {settings.deepseek_model}")
    print(f"Java Backend: {settings.java_backend_url}")
    print(f"Max Iterations: {settings.mcts_max_iterations}")
    print(f"Coverage Target: {settings.mcts_coverage_target * 100}%")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\nLATS Test Generation Server Shutting Down")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}", "type": type(exc).__name__},
    )


def main():
    """Run the server"""
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Enable auto-reload during development
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
