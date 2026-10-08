# Sub-project 3d-2: road-network and endgame-threat inputs (encoder v4)

Date: 2026-10-08. Builds on encoder v3 (`net/src/encode.rs`, `ENCODER_VERSION` 3, `PUBLIC_LEN`
1581), the v2 → v3 upgrade tool (`alphasettler/netupgrade.py`, c6b2646) and 3d-1b (the opening
model in self-play, `docs/superpowers/specs/2026-10-08-opening-selfplay-design.md`).

The user, after beating the bots by hand (2026-10-07): they "built their roads in a way that made it
impossible to get longest road later", did not build roads to the best locations, and missed
endgame threats. The graph net sees roads only as per-edge ownership and has two rounds of message
passing, so a road network's length, where it can still reach, and how close each seat is to an
award or to winning must be inferred from scratch. This sub-project states them as inputs. The user
asked (2026-10-08) for all of 3d to be done; decisions the user did not make are marked **Ruling**.
Per the user, nothing here encodes which resources a seat's buildings touch.

## 1. Done criteria

1. **Encoder v4** (Section 2): `ENCODER_VERSION` 4, the new hash, the golden position re-encoded by
   hand, seat rotation tests extended to the new blocks; the PyTorch twin and the parity test follow
   from the constants.
2. **Upgrade:** `alphasettler net-upgrade` turns a v3 net into a v4 net whose outputs are identical
   on every position (the new input columns are zero), checked by the Rust loader and a forward-pass
   comparison on real positions; `runs/3c/nets/gen-034.bin` upgraded is the start net.
3. **The test, at equal wall time:** arm D, a `fast-loop` from the upgraded gen-034 with the
   settings of the 3d-1b arm that won its test (arm C if 3d-1b was H1, else P2's arm A) and the same
   self-play seeds, 10,800 s of generations after an untimed set-up. Then `net:D@100+t[+o]` against
   three copies of that arm's best (`[+o]` exactly when arm D used `--opening`), `alphasettler sprt
   --trades` (p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 5,000,000,000). H1: v4 becomes
   the encoder of every new loop. H0: recorded.
4. **Results** in `docs/bot/results-3d2.md`: search speed (simulations per second, one core, batch 8,
   1,000 simulations, v3 against v4 on the same net upgraded), arm D's generation table beside its
   control's, the SPRT line.

## 2. The new inputs

All seat-relative (r0 = the viewer, then turn order), as every v3 block is; all appended at the end
of their block so the upgrade only pads columns (Section 3).

**Per node (+5, `NODE_WIDTH` 19 → 24):**

- r0..r3 road distance: the fewest roads seat r must build to touch the node with its network, where
  the network is every node its roads or buildings touch; paths may not use another seat's road or
  pass through a node another seat has built on; capped at 3 and divided by 3; unreachable within 3
  is 1.0. A node on the network is 0.
- open spot: 1 if the node is empty and no neighbour is built on (the distance rule).

**Per edge (+4, `EDGE_WIDTH` 4 → 8):**

- r0..r3 on the longest trail: 1 if the edge lies on one longest trail of seat r's roads (the trail
  `awards::longest_road` measures; ties between equally long trails go to the first found, a
  deterministic order).

**Globals: a threat block per seat (+24: 4 seats × 6, appended after the trade block):**

- longest road length / 15;
- roads to the longest-road award: 0 for the holder, else `min(5, max(5, holder length + 1) −
  own length)` (no holder: `max(0, 5 − own length)`), divided by 5;
- knights to the largest-army award: the same rule with 3 and played knights, divided by 3;
- points to win: `max(0, vp_to_win − points) / vp_to_win`, where points are the public VP for
  opponents and the total (VP cards included) for the viewer, the information each side has;
- settlements left in supply / 5, cities left in supply / 4.

`PUBLIC_LEN` becomes 1581 + 54·5 + 72·4 + 24 = 2163. `HIDDEN_LEN` is unchanged.

**Ruling:** the trail and the road length come from the engine's own longest-road walk (a function
returning the trail's edges beside its length, with `longest_road` delegating to it so the golden
trace cannot move), not a second implementation in the encoder. Cost if wrong: an engine edit;
`engine/tests/golden_trace.rs` guards it.

**Ruling:** distances capped at 3: a settlement more than three roads away is not a near-term
target, and the cap keeps the BFS to a few dozen nodes. Cost if wrong: the net cannot tell 4 from 6.

## 3. The upgrade

`netupgrade` gains v3 → v4: each encoder's first layer gains zero columns at the end of its input
(node 19 → 24, edge 4 → 8, global 115 → 139, the trade block included); every other byte is
copied; the header carries the v4 hash. The result is loaded by the Rust loader and checked to give
the same outputs as the v3 net on a set of real positions (within the forward pass's 1e-5 batch
tolerance, bit-equal at batch 1 if the zero columns contribute exactly zero). The v2 → v3 path
stays.

## 4. Out of scope

Resource-coverage inputs (the user ruled them out), a wider or deeper net, more message-passing
rounds, changes to the heads, retraining the opening model, 3d-3 and 3d-4.

## 5. Testing

- Rust: hand-checked road distances, trail edges and threat values on a built position (including a
  blocked path and a tie between trails); seat rotation permutes every new block; the golden
  position's new slots; `the_layout_adds_up` and the hash tests; `longest_road` unchanged on every
  existing test and the golden trace.
- Python: `net-upgrade` v3 → v4 gives identical outputs on real positions, refuses a v4 input and
  an existing output; the PyTorch parity test at the new widths; a tiny `fast-loop` from an upgraded
  net runs.
