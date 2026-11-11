
import jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.auth import decode_access_token
from backend.database.connection import get_db
from backend.database.models import Farmer


bearer_scheme = HTTPBearer()


def get_current_farmer(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(credentials.credentials)
        farmer_id = payload.get("sub")

        if farmer_id is None:
            raise credentials_exception

        farmer_id = int(farmer_id)

    except (jwt.InvalidTokenError, ValueError, TypeError):
        raise credentials_exception

    farmer = db.query(Farmer).filter(
        Farmer.id == farmer_id
    ).first()

    if farmer is None:
        raise credentials_exception

    return farmer
