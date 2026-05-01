from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uvicorn
from config import settings
from database import get_db, engine
from models import Base
from core.auth import hash_password
from models.user import User
from ml import load_all_models
from api import auth_router, user_router, admin_router, query_router

app = FastAPI(title="Philosophical Engine API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(user_router, prefix="/api", tags=["user"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(query_router, prefix="/api", tags=["query"])

# Static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.on_event("startup")
async def startup_event():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Seed admin
    async for db in get_db():
        result = await db.execute(select(User).where(User.email == settings.admin_email))
        if not result.scalar_one_or_none():
            hashed = hash_password(settings.admin_password)
            admin = User(email=settings.admin_email, hashed_password=hashed, role="admin")
            db.add(admin)
            await db.commit()
        break
    
    # Load models
    load_all_models()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)