
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class FarmerSignup(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class FarmerLogin(BaseModel):
    email: EmailStr
    password: str


class FarmerResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
