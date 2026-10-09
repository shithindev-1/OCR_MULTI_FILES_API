"""Every route the service exposes, collected into one router.

`/health` stays at the root and unauthenticated, so load balancers and the service monitor can
probe it; everything else lives under `/api/v1` behind the X-API-Key check.
"""

from fastapi import APIRouter, Depends

from app.api.routes import health, leasing, leasing_detect, ocr
from app.core.exceptions import API_V1_PREFIX
from app.core.security import require_api_key

v1_router = APIRouter(prefix=API_V1_PREFIX, dependencies=[Depends(require_api_key)])
v1_router.include_router(ocr.router)
v1_router.include_router(leasing.router)
v1_router.include_router(leasing_detect.router)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(v1_router)
