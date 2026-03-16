"""Endpoint per il registro campi (built-in e custom)."""

from fastapi import APIRouter, HTTPException

from field_registry import registry

router = APIRouter(tags=["Fields"])


@router.get("/api/fields")
async def get_fields():
    return {
        "sections": registry.to_sections_json(),
        "all_keys": registry.get_keys(),
    }


@router.get("/api/fields/custom")
async def get_custom_fields():
    return [f.to_dict() for f in registry.get_custom_fields()]


@router.post("/api/fields/custom")
async def add_custom_field(payload: dict):
    try:
        fd = registry.add_custom_field(payload)
        return {"status": "ok", "field": fd.to_dict()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.put("/api/fields/custom/{key}")
async def update_custom_field(key: str, payload: dict):
    try:
        fd = registry.update_custom_field(key, payload)
        return {"status": "ok", "field": fd.to_dict()}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/fields/custom/{key}")
async def delete_custom_field(key: str):
    ok = registry.delete_custom_field(key)
    if not ok:
        raise HTTPException(404, f"Campo custom '{key}' non trovato")
    return {"status": "ok"}
