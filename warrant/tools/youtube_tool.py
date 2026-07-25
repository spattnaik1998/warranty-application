"""YouTube discovery tool (exogenous injector: finds the newest video)."""

from __future__ import annotations

from warrant.config import get_settings
from warrant.exceptions import ToolError
from warrant.schemas.belief import EvidenceRef
from warrant.tools.fixtures import YOUTUBE_LATEST
from warrant.tools.registry import ToolResult, ToolRole, register


@register("youtube_latest", ToolRole.INJECTOR, "youtube",
          "Find the newest video from a channel and its referenced arXiv ids.")
def youtube_latest(channel: str) -> ToolResult:
    settings = get_settings()
    key = channel.strip().lower().replace(" ", "-")
    if settings.mock:
        video = YOUTUBE_LATEST.get(key) or YOUTUBE_LATEST.get("last-week-in-ai")
        if video is None:
            raise ToolError("youtube_latest", f"no fixture for channel {channel!r}")
    else:  # pragma: no cover - live network
        raise ToolError("youtube_latest",
                        "live YouTube discovery requires an API key; set WARRANT_MOCK=1")
    obs = (f"Newest video: {video['title']} ({video['url']}).\n"
           f"Description: {video['description']}")
    return ToolResult(
        observation=obs,
        evidence_refs=[EvidenceRef(source_type="youtube", source_id=video["video_id"],
                                   snippet=video["title"])],
        artifacts={"url": video["url"], "arxiv_ids": ",".join(video.get("arxiv_ids", []))},
    )
