"""
UI Routes
UI Framework endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from api.auth import verify_api_key
from engines.ui import UIFramework, UIConfig

router = APIRouter()

# Cache for UI instances
_ui_cache = {}


async def get_ui(api_key: dict = Depends(verify_api_key)) -> UIFramework:
    """Get or create UI framework instance."""
    partner_id = api_key["partner_id"]

    if partner_id not in _ui_cache:
        config = UIConfig(theme=api_key.get("theme", "dark"))
        _ui_cache[partner_id] = UIFramework(config=config)

    return _ui_cache[partner_id]


@router.post("/dashboard")
async def render_dashboard(
    components: Optional[List[str]] = None,
    ui: UIFramework = Depends(get_ui),
    api_key: dict = Depends(verify_api_key),
):
    """Render NeuralBlitz dashboard."""
    result = await ui.render_dashboard(components=components)
    return {"success": True, **result}


@router.get("/components")
async def list_components(
    ui: UIFramework = Depends(get_ui), api_key: dict = Depends(verify_api_key)
):
    """List available UI components."""
    result = await ui.get_components()
    return result


@router.get("/capabilities")
async def get_ui_capabilities(
    ui: UIFramework = Depends(get_ui), api_key: dict = Depends(verify_api_key)
):
    """Get UI framework capabilities."""
    capabilities = await ui.get_capabilities()
    return capabilities
