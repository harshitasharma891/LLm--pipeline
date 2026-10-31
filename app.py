##app.py#
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
# from dotenv import load_dotenv
import os, shutil
from datetime import datetime
from urllib.parse import urlparse

from src.license_execute import (
    check_license,
    LicenseFileError,
    LicenseVerificationError,
    LicenseExpiredError,
    LicenseError,
    NTPError,
)
from api.v1 import api_router  # Ensure this router exists

# --------------------------- Load environment variables ---------------------------
# load_dotenv()

# --------------------------- Global License Status Flag ---------------------------
license_valid = False
license_dir = "./License"


def verify_license_dependency():
    global license_valid
    if not license_valid:
        raise HTTPException(
            status_code=403,
            detail="License not verified. Please upload and validate license first.",
        )


# --------------------------- FastAPI App Initialization ---------------------------
app = FastAPI(
    title="RAG API",
    description="API for RAG Pipeline Evaluation",
    version="1.0.0",
)

# --------------------------- CORS Configuration ---------------------------
cors_allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
if cors_allowed_origins_str:
    cors_allowed_origins = [origin.strip() for origin in cors_allowed_origins_str.split(",") if origin.strip()]
else:
    # Strict whitelist based on security hardening plan
    cors_allowed_origins = [
        "http://localhost",
        "http://localhost:5001",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:5001",
        "http://127.0.0.1:3000"
    ]

allow_credentials = True
if "*" in cors_allowed_origins:
    if len(cors_allowed_origins) == 1:
        allow_credentials = False
    else:
        cors_allowed_origins = [o for o in cors_allowed_origins if o != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Security Headers & Origin Validation ---------------------------
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    method = request.method.upper()
    origin = request.headers.get("origin")

    # State-changing methods require CSRF origin validation
    state_changing = method in {"POST", "PUT", "DELETE", "PATCH"}

    # Allow WebSocket upgrade requests (they use their own origin check via CORS)
    is_ws_upgrade = (
        request.headers.get("upgrade", "").lower() == "websocket"
    )

    if state_changing and not is_ws_upgrade:
        if not origin:
            # No Origin header on a state-changing request → block
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Missing Origin header"}
            )

        parsed_origin = urlparse(origin)
        allowed_hosts = {"localhost", "127.0.0.1", "34.93.237.26"}

        if parsed_origin.hostname not in allowed_hosts and origin not in cors_allowed_origins:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Untrusted Origin"}
            )
    elif origin:
        # For non-state-changing requests, still validate Origin when present
        parsed_origin = urlparse(origin)
        allowed_hosts = {"localhost", "127.0.0.1", "34.93.237.26"}

        if parsed_origin.hostname not in allowed_hosts and origin not in cors_allowed_origins:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden: Untrusted Origin"}
            )

    response = await call_next(request)

    # Security Headers
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response



# --------------------------- Custom Exception Handler ---------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )


# --------------------------- Protected Router with License Check ---------------------------
protected_router = APIRouter(dependencies=[Depends(verify_license_dependency)])
protected_router.include_router(api_router, prefix="/v2")
app.include_router(protected_router)


# --------------------------- License Status Endpoint ---------------------------
@app.get("/license-status")
async def license_status():
    status_code = 200 if license_valid else 201
    return JSONResponse(
        status_code=status_code,
        content={
            "status_code": status_code,
            "license_valid": license_valid,
            "message": "License is valid." if license_valid else "License is invalid or not verified.",
            "timestamp": datetime.now().isoformat(),
        },
    )


# --------------------------- License Upload Endpoint ---------------------------
@app.post("/upload-license-components")
async def upload_license_components(
    license_file: UploadFile = File(...),
    key_file: UploadFile = File(...),
    certificate_file: UploadFile = File(...),
):
    global license_valid
    try:
        os.makedirs(license_dir, exist_ok=True)

        # --- Strict filename validation ---
        if license_file.filename != "license.key":
            raise HTTPException(
                status_code=400,
                detail="Invalid license filename. Must be 'license.key'.",
            )
        if key_file.filename != "key.pem":
            raise HTTPException(
                status_code=400,
                detail="Invalid key filename. Must be 'key.pem'.",
            )
        if certificate_file.filename != "certificate.pem":
            raise HTTPException(
                status_code=400,
                detail="Invalid certificate filename. Must be 'certificate.pem'.",
            )

        # --- Ensure non-empty uploads ---
        contents = {
            "license.key": await license_file.read(),
            "key.pem": await key_file.read(),
            "certificate.pem": await certificate_file.read(),
        }
        for fname, content in contents.items():
            if not content.strip():
                raise HTTPException(status_code=400, detail=f"{fname} file is empty.")

        # --- Save uploaded files (write from validated contents, not tainted file objects) ---
        for fname, content in contents.items():
            with open(os.path.join(license_dir, fname), "wb") as buffer:
                buffer.write(content)

        # --- Verify license ---
        if check_license():
            license_valid = True
            return JSONResponse(
                status_code=200,
                content={
                    "status_code": 200,
                    "message": "Files uploaded and license verified successfully.",
                },
            )
        else:
            license_valid = False
            raise HTTPException(
                status_code=400,
                detail="Files uploaded, but license verification failed.",
            )

    except (
        LicenseFileError,
        LicenseVerificationError,
        LicenseExpiredError,
        NTPError,
    ) as e:
        license_valid = False
        raise HTTPException(status_code=400, detail=f"License validation failed: {e}")
    except Exception as e:
        license_valid = False
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --------------------------- Root & Health Check ---------------------------
@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "status_code": 200,
            "message": "Welcome to the RAG API. Visit /v2 for versioned routes or /health to check server status.",
        },
    )


@app.get("/health")
async def health_check():
    status_code = 200 if license_valid else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status_code": status_code,
            "status": "healthy" if license_valid else "unhealthy",
            "license_valid": license_valid,
            "timestamp": datetime.now().isoformat(),
        },
    )


# --------------------------- Auto-check license on startup ---------------------------
@app.on_event("startup")
def startup_check_license():
    global license_valid
    try:
        files = ["license.key", "key.pem", "certificate.pem"]
        if all(os.path.exists(os.path.join(license_dir, f)) for f in files):
            if check_license():
                license_valid = True
                print(" License verified at startup.")
            else:
                license_valid = False
                print(" License files exist but verification failed.")
        else:
            license_valid = False
            print(" License files missing. Please upload.")
    except Exception as e:
        license_valid = False
        print(f" Error verifying license at startup: {e}")