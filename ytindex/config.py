"""Konfigurace projektu (ytindex.toml + proměnné prostředí)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_FILE = "ytindex.toml"


@dataclass
class ChannelConfig:
    input: str = ""          # co zadal uživatel (handle / id / url)
    id: str = ""             # UC...
    name: str = ""
    url: str = ""


@dataclass
class TranscriptConfig:
    # preferované jazyky titulků v pořadí priority
    languages: list[str] = field(default_factory=lambda: ["cs", "sk", "en"])
    # pořadí zdrojů transkriptu; první úspěšný vyhraje
    providers: list[str] = field(default_factory=lambda: ["captions", "ytdlp", "gemini"])
    proxy_url: str = ""      # http(s) proxy pro captions/ytdlp (nebo env YTINDEX_PROXY_URL)
    cookies_file: str = ""   # Netscape cookies.txt pro yt-dlp (nepovinné)
    request_delay: float = 1.0   # pauza mezi videi v sekundách (ochrana proti rate-limitu)


@dataclass
class GeminiConfig:
    model: str = "gemini-3.5-flash-lite"      # model pro transkripci videí bez titulků
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    transcribe_window_s: int = 600            # délka úseku videa pro jednu transkripční žádost (10 min ≈ 19k tokenů, ~13 s)
    transcribe_fps: float = 0.1               # málo snímků = levné; zajímá nás audio


@dataclass
class ChunkConfig:
    target_chars: int = 450    # cílová délka chunku (cca 30–60 s řeči)
    max_seconds: float = 75.0  # maximální délka chunku v sekundách
    overlap_segments: int = 1  # kolik segmentů se překrývá se sousedním chunkem


@dataclass
class Config:
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    transcripts: TranscriptConfig = field(default_factory=TranscriptConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    db_path: str = "data/index.sqlite"

    # --- klíče z prostředí (nikdy nejsou v toml) ---
    @property
    def gemini_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    @property
    def youtube_api_key(self) -> str:
        return os.environ.get("YOUTUBE_API_KEY", "")

    @property
    def proxy_url(self) -> str:
        return os.environ.get("YTINDEX_PROXY_URL", "") or self.transcripts.proxy_url

    def to_toml(self) -> str:
        return _dump_toml(asdict(self))


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _dump_toml(d: dict) -> str:
    """Minimální TOML writer pro náš plochý (max. 1 úroveň sekcí) tvar konfigurace."""
    lines: list[str] = []
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_toml_value(v)}")
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append("")
            lines.append(f"[{k}]")
            for kk, vv in v.items():
                lines.append(f"{kk} = {_toml_value(vv)}")
    return "\n".join(lines) + "\n"


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path or os.environ.get("YTINDEX_CONFIG", CONFIG_FILE))
    cfg = Config()
    if not p.exists():
        return cfg
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    cfg.db_path = raw.get("db_path", cfg.db_path)
    for section, cls in (("channel", ChannelConfig), ("transcripts", TranscriptConfig),
                         ("gemini", GeminiConfig), ("chunking", ChunkConfig)):
        if section in raw:
            known = {k: v for k, v in raw[section].items() if k in cls.__dataclass_fields__}
            setattr(cfg, section, cls(**known))
    return cfg


def save_config(cfg: Config, path: str | os.PathLike | None = None) -> Path:
    p = Path(path or os.environ.get("YTINDEX_CONFIG", CONFIG_FILE))
    p.write_text(cfg.to_toml(), encoding="utf-8")
    return p
