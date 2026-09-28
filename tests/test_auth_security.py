import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from src.api.security import auth


def _credentials(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _encode(payload):
    return jwt.encode(payload, auth.settings.SECRET_KEY, algorithm=auth.ALGORITHM)


def test_current_role_rejeita_sub_vazio():
    token = _encode({
        "sub": "",
        "role": "analyst",
        "iss": auth.settings.JWT_ISSUER,
        "aud": auth.settings.JWT_AUDIENCE,
        "exp": 4102444800,
    })

    with pytest.raises(HTTPException) as exc:
        auth._current_role(credentials=_credentials(token), x_ml_service_token=None)

    assert exc.value.status_code == 401


def test_current_role_rejeita_role_invalida():
    token = _encode({
        "sub": "user",
        "role": "hacker",
        "iss": auth.settings.JWT_ISSUER,
        "aud": auth.settings.JWT_AUDIENCE,
        "exp": 4102444800,
    })

    with pytest.raises(HTTPException) as exc:
        auth._current_role(credentials=_credentials(token), x_ml_service_token=None)

    assert exc.value.status_code == 403
