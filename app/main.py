from fastapi import FastAPI
from app.routes import match

app = FastAPI(title="Cofounder Matcher API")
app.include_router(match.router, prefix="/api")