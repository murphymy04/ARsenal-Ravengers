import json
from typing import Optional

from fastapi import HTTPException, Header
from config import FIREBASE_CREDENTIALS, FIREBASE_PROJECT_ID

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials
    from firebase_admin.exceptions import FirebaseError
except ImportError:
    firebase_admin = None


def _init_firebase():
    """Initialize Firebase Admin SDK once."""
    if not firebase_admin:
        raise RuntimeError("firebase_admin package is required for Firebase token validation")
    if firebase_admin._apps:
        return

    if not FIREBASE_CREDENTIALS:
        raise RuntimeError("FIREBASE_CREDENTIALS env var is required for Firebase auth")

    try:
        if FIREBASE_CREDENTIALS.strip().startswith("{"):
            data = FIREBASE_CREDENTIALS
            cred = credentials.Certificate(json.loads(data))
        else:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Firebase credentials: {exc}. "
                          f"Make sure FIREBASE_CREDENTIALS points to a valid service account key file "
                          f"(not firebase.json config file). Download from Firebase Console > "
                          f"Project Settings > Service Accounts > Generate new private key.")

    try:
        if FIREBASE_CREDENTIALS.strip().startswith("{"):
            parsed = json.loads(FIREBASE_CREDENTIALS)
        else:
            with open(FIREBASE_CREDENTIALS, 'r') as f:
                parsed = json.load(f)

        if parsed.get('type') != 'service_account':
            raise RuntimeError("Firebase credentials file is not a service account key. "
                               "It appears to be a client config file. Please download the "
                               "service account key from Firebase Console > Project Settings > "
                               "Service Accounts > Generate new private key.")
    except Exception as exc:
        if "not a service account key" in str(exc):
            raise exc
        raise RuntimeError(f"Failed to validate Firebase credentials format: {exc}")

    kwargs = {}
    if FIREBASE_PROJECT_ID:
        kwargs["projectId"] = FIREBASE_PROJECT_ID

    firebase_admin.initialize_app(cred, kwargs)


def _get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Require Bearer token in Authorization header and validate against Firebase."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    try:
        _init_firebase()
        decoded = firebase_auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decoded.get("uid")
