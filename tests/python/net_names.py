"""Malformed `net:` bot names, shared by the native and Catanatron arena tests. Each must be
rejected as an unknown bot before the nets directory is read: path characters, a bare or
over-long NAME, and every bad @N / +rD form."""

BAD_NET_NAMES = [
    "net:",
    "net:../x",
    "net:..",
    "net:.",
    "net:a/b",
    "net:/abs",
    "net:a\\b",
    "net:a.bin",
    "net:a b",
    "net:" + "a" * 65,
    "net:@5",
    "net:a@",
    "net:a@0",
    "net:a@abc",
    "net:a@1000001",
    "net:a@300+r0",
    "net:a@300+r201",
    "net:a@5@6",
    "net:a+r8",
]
