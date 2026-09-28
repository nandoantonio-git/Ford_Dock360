import asyncio

import pytest
from fastapi import HTTPException

from src.api.config import validate_secret_key
from src.api.models.schemas import RoleEnum
from src.api.routers import auth as auth_router


def test_create_demo_token_rejeita_secret_nao_configurado(monkeypatch):
    monkeypatch.setattr(auth_router.settings, "DEMO_TOKEN_SECRET", None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth_router.create_demo_token.__wrapped__(request=None, x_demo_token_secret="qualquer"))

    assert exc.value.status_code == 503


def test_create_demo_token_rejeita_secret_invalido(monkeypatch):
    monkeypatch.setattr(auth_router.settings, "DEMO_TOKEN_SECRET", "segredo-correto")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(auth_router.create_demo_token.__wrapped__(request=None, x_demo_token_secret="segredo-errado"))

    assert exc.value.status_code == 403


def test_create_demo_token_retorna_token_para_analyst(monkeypatch):
    monkeypatch.setattr(auth_router.settings, "DEMO_TOKEN_SECRET", "segredo-correto")
    monkeypatch.setattr(auth_router.settings, "DEMO_TOKEN_SUBJECT", "demo")
    monkeypatch.setattr(auth_router.settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 15)
    monkeypatch.setattr(auth_router, "create_access_token", lambda subject, role: f"token-{subject}-{role.value}")

    response = asyncio.run(auth_router.create_demo_token.__wrapped__(request=None, x_demo_token_secret="segredo-correto"))

    assert response.access_token == "token-demo-analyst"
    assert response.role == RoleEnum.analyst
    assert response.expires_in_minutes == 15


def test_validate_secret_key_rejeita_chave_insegura():
    with pytest.raises(SystemExit, match="SECRET_KEY invalida"):
        validate_secret_key("changeme-local-only")
