from datetime import datetime, timedelta
import hmac
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.api.config import settings


ALGORITHM = "HS256"
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="JWT Bearer de usuario/demo. Tokens gerados por /auth/demo-token expiram.",
)
service_token_scheme = APIKeyHeader(
    name="X-ML-Service-Token",
    auto_error=False,
    description="Service token fixo para comunicacao server-to-server Java BFF -> FastAPI ML.",
)
ROLE_LEVELS = {
    "viewer": 1,
    "analyst": 2,
    "admin": 3,
    "service": 3,
}


def _role_value(role: Any) -> str:
    return role.value if hasattr(role, "value") else str(role)


def create_access_token(subject: str, role: Any) -> str:
    expires_at = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "role": _role_value(role),
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "exp": expires_at,
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def _service_token() -> str | None:
    return settings.ML_SERVICE_TOKEN or settings.FORD_ML_SERVICE_TOKEN


def _is_valid_service_token(token: str | None) -> bool:
    expected = _service_token()
    return bool(token and expected and hmac.compare_digest(token, expected))


def _current_role(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_ml_service_token: str | None = Depends(service_token_scheme),
):
    if x_ml_service_token is not None:
        if _is_valid_service_token(x_ml_service_token):
            return "service"
        raise HTTPException(status_code=401, detail="Service token invalido")

    if credentials is None:
        raise HTTPException(status_code=401, detail="Credenciais ausentes")

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={
                "require_exp": True,
                "require_iss": True,
                "require_aud": True,
                "require_sub": True,
            },
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Token JWT invalido")

    if not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Token JWT invalido")

    role = payload.get("role")
    if role not in ROLE_LEVELS:
        raise HTTPException(status_code=403, detail="Role invalida")
    return role


def require_role(required_role: Any):
    def dependency(role: str = Depends(_current_role)):
        if ROLE_LEVELS[role] < ROLE_LEVELS[_role_value(required_role)]:
            raise HTTPException(status_code=403, detail="Permissao insuficiente")
        return role

    return dependency
