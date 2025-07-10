from fastapi import FastAPI
from app.routes import match

app = FastAPI(title="Cofounder Matcher API")
app.include_router(match.router, prefix="/api")
@app.get("/")
def root():
    return {"status": "Cofounder Matcher API is running"}
