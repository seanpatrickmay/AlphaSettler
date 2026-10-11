"""Targets v2's training-only heads (spec 2026-10-11 Section 4): final ownership per node and edge,
VP parts, the award holders and turns left, on the trunk's node, edge and global embeddings and the
value head's hidden layer. They are never in the weights file: `fast_train` keeps them in the
`.train.pt` sidecar."""

from __future__ import annotations

import torch
from torch import nn

from alphasettler.netformat import EDGES, NODES

Q_MIX = 0.5
NODE_CLASSES, EDGE_CLASSES, HOLDER_CLASSES, PARTS = 9, 5, 5, 12
WEIGHTS = {"node": 0.25, "edge": 0.25, "parts": 0.25, "road": 0.1, "army": 0.1, "turns": 0.1}


class Extras(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.node = nn.Linear(2 * d, NODE_CLASSES)
        self.edge = nn.Linear(2 * d, EDGE_CLASSES)
        self.scalar = nn.Linear(4 * d, PARTS + 2 * HOLDER_CLASSES + 1)

    def forward(self, node, edge, glob, v) -> dict[str, torch.Tensor]:
        g = glob.unsqueeze(1)
        s = self.scalar(v)
        return {"node": self.node(torch.cat([node, g.expand(-1, NODES, -1)], 2)),
                "edge": self.edge(torch.cat([edge, g.expand(-1, EDGES, -1)], 2)),
                "parts": s[:, :PARTS], "road": s[:, PARTS:PARTS + HOLDER_CLASSES],
                "army": s[:, PARTS + HOLDER_CLASSES:PARTS + 2 * HOLDER_CLASSES],
                "turns": s[:, -1]}


def extras_losses(out: dict, t) -> dict[str, torch.Tensor]:
    ce = nn.functional.cross_entropy
    parts = {
        "extras_node": ce(out["node"].reshape(-1, NODE_CLASSES), t.own_node.reshape(-1)),
        "extras_edge": ce(out["edge"].reshape(-1, EDGE_CLASSES), t.own_edge.reshape(-1)),
        "extras_parts": nn.functional.mse_loss(out["parts"], t.parts),
        "extras_road": ce(out["road"], t.holders[:, 0]),
        "extras_army": ce(out["army"], t.holders[:, 1]),
        "extras_turns": nn.functional.mse_loss(out["turns"], t.turns_left),
    }
    parts["extras"] = sum(WEIGHTS[k] * parts[f"extras_{k}"] for k in WEIGHTS)
    return parts
