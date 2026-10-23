import json

import pytest
from fastapi.testclient import TestClient

from ytindex.api import create_app
from ytindex.mcp_server import build_mcp


@pytest.mark.anyio
async def test_mcp_tools_direct(cfg, db):
    mcp = build_mcp(cfg, db)
    tools = {t.name for t in await mcp.list_tools()}
    assert {"search_videos", "get_transcript", "list_videos", "channel_info"} <= tools
    res = await mcp.call_tool("search_videos", {"query": "bitcoin", "limit": 5})
    payload = res.structured_content or json.loads(res.content[0].text)
    assert payload["total"] == 2 and "<mark>" in payload["results"][0]["match"]["highlighted"]
    tr = await mcp.call_tool("get_transcript", {"video_id": "vid00000002", "start_s": 4, "end_s": 6})
    tr = tr.structured_content or json.loads(tr.content[0].text)
    assert len(tr["segments"]) == 2


def test_mcp_streamable_http(cfg, db):
    with TestClient(create_app(cfg, db)) as client:
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        r = client.post("/mcp/", json=init, headers=headers)
        assert r.status_code == 200, r.text
        r = client.post("/mcp/", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=headers)
        names = {t["name"] for t in r.json()["result"]["tools"]}
        assert "search_videos" in names
        r = client.post("/mcp/", json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                       "params": {"name": "search_videos", "arguments": {"query": "inflace"}}}, headers=headers)
        out = r.json()["result"]
        data = out.get("structuredContent") or json.loads(out["content"][0]["text"])
        assert data["results"][0]["video_id"] == "vid00000002"
