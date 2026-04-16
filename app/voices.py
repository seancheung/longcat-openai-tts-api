from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Voice:
    id: str
    wav_path: Path
    txt_path: Path
    prompt_text: str
    mtime: float


class VoiceCatalog:
    def __init__(self, root: Path):
        self.root = root

    def _read_text(self, path: Path) -> str:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        return raw.decode("utf-8").strip()

    def scan(self) -> Dict[str, Voice]:
        voices: Dict[str, Voice] = {}
        if not self.root.exists() or not self.root.is_dir():
            log.warning("voices directory %s missing or not a dir", self.root)
            return voices

        wavs = {p.stem: p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".wav"}
        txts = {p.stem: p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        for vid in sorted(set(wavs) & set(txts)):
            wav_path = wavs[vid]
            txt_path = txts[vid]
            try:
                text = self._read_text(txt_path)
            except UnicodeDecodeError:
                log.warning("voice %s: txt is not UTF-8, skipped", vid)
                continue
            if not text:
                log.warning("voice %s: txt empty, skipped", vid)
                continue
            mtime = max(wav_path.stat().st_mtime, txt_path.stat().st_mtime)
            voices[vid] = Voice(
                id=vid,
                wav_path=wav_path,
                txt_path=txt_path,
                prompt_text=text,
                mtime=mtime,
            )

        return voices

    def get(self, vid: str) -> Voice | None:
        return self.scan().get(vid)
