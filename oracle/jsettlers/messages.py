"""One recorded message of the harness's stream: its GameEventLog audience and its toString(),
parsed into fields. Resource sets (`clay=..|ore=..|sheep=..|wheat=..|wood=..|unknown=..`) become
dicts; every other value stays a string."""

from __future__ import annotations

from dataclasses import dataclass

from oracle.jsettlers.tables import js5_to_ours

SET_KEYS = ("clay", "ore", "sheep", "wheat", "wood", "unknown")


@dataclass(frozen=True)
class Entry:
    aud: str  # all, pN, !pN, !p[a, b], fN (a client's request), fo, ob, un
    name: str  # the message class, e.g. "SOCPutPiece"
    f: dict
    raw: dict | None = None  # the server-side position after it, when it changed

    @property
    def from_client(self) -> bool:
        return self.aud.startswith("f")

    @property
    def sender(self) -> int | None:
        return int(self.aud[1:]) if self.aud[:1] == "f" and self.aud[1:].isdigit() else None

    def num(self, key: str) -> int:
        v = self.f[key]
        return int(v, 16) if key == "coord" else int(v)


def parse(aud: str, text: str, raw: dict | None = None) -> Entry:
    name, _, body = text.partition(":")
    tokens = body.split("|") if body else []
    fields: dict = {}
    i = 0
    while i < len(tokens):
        key, _, value = tokens[i].partition("=")
        if value.startswith("clay="):
            parts = [value, *tokens[i + 1:i + 6]]
            rset = {}
            for part in parts:
                k, _, v = part.partition("=")
                rset[k] = int(v)
            if tuple(rset) != SET_KEYS:
                raise ValueError(f"a resource set without the six JSettlers keys in {text!r}")
            fields[key] = rset
            i += 6
        else:
            fields[key] = value
            i += 1
    return Entry(aud, name, fields, raw)


def ours5(rset: dict) -> list[int]:
    """A JSettlers resource set's known cards in our resource order."""
    return js5_to_ours([rset[k] for k in SET_KEYS[:5]])
