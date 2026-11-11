
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import Farmer
from backend.schemas import (
    FarmerSignup,
    FarmerLogin,
    FarmerResponse,
    TokenResponse
)
from backend.auth import (
    hash_password,
    verify_password,
    create_access_token
)
from backend.dependencies import get_current_farmer


router = APIRouter(tags=["Farmer Authentication"])


@router.post(
    "/signup",
    response_model=FarmerResponse,
    status_code=status.HTTP_201_CREATED
)
def signup(
    farmer_data: FarmerSignup,
    db: Session = Depends(get_db)
):
    existing_farmer = db.query(Farmer).filter(
        Farmer.email == farmer_data.email
    ).first()

    if existing_farmer:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_farmer = Farmer(
        full_name=farmer_data.full_name,
        email=farmer_data.email,
        hashed_password=hash_password(farmer_data.password)
    )

    db.add(new_farmer)
    db.commit()
    db.refresh(new_farmer)

    return new_farmer


@router.post("/login", response_model=TokenResponse)
def login(
    login_data: FarmerLogin,
    db: Session = Depends(get_db)
):
    farmer = db.query(Farmer).filter(
        Farmer.email == login_data.email
    ).first()

    if not farmer or not verify_password(
        login_data.password,
        farmer.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(
        data={"sub": str(farmer.id)}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=FarmerResponse)
def get_my_profile(
    current_farmer: Farmer = Depends(get_current_farmer)
):
    return current_farmer
