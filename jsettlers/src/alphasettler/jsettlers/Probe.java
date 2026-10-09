// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import soc.game.SOCBoard;
import soc.game.SOCBoard4p;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCGameOptionSet;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.robot.SOCRobotDM;
import soc.util.Version;

/**
 * Prints, as one JSON object, the facts of the jar's classic 4-player board and codes that
 * oracle/jsettlers/tables.py is checked against: land hexes, their corner nodes in JSettlers'
 * clockwise-from-north order, every land edge and its end nodes, the hex-layout index of each
 * hex, the port edges, facings and hex numbers, and the resource, card, piece, hex, state and
 * strategy constants. Run by Task 1 into tests/python/data/jsettlers_board.json.
 */
public final class Probe {
    private Probe() {}

    public static void main(String[] args) throws ReflectiveOperationException {
        SOCBoard4p b = new SOCBoard4p(SOCGameOptionSet.getAllKnownOptions());
        Map<String, Object> out = Json.map("version", Version.version());
        List<Object> land = new ArrayList<>(), hexNodes = new ArrayList<>(), edges = new ArrayList<>(),
            hexNums = new ArrayList<>();
        TreeSet<Integer> seen = new TreeSet<>();
        for (int h : b.getLandHexCoords()) {
            land.add(h);
            int[] ns = b.getAdjacentNodesToHex_arr(h);
            List<Object> row = Json.list(h);
            for (int n : ns) row.add(n);
            hexNodes.add(row);
            for (int i = 0; i < 6; i++) {
                int e = b.getEdgeBetweenAdjacentNodes(ns[i], ns[(i + 1) % 6]);
                if (seen.add(e)) {
                    int[] ab = b.getAdjacentNodesToEdge_arr(e);
                    edges.add(Json.list(e, ab[0], ab[1]));
                }
            }
        }
        for (int h = 0; h < 0x100; h++) {
            int n = b.getHexNumFromCoord(h);
            if (n >= 0) hexNums.add(Json.list(h, n));
        }
        out.put("land_hexes", land);
        out.put("hex_nodes", hexNodes);
        out.put("edges", edges);
        out.put("hex_nums", hexNums);
        Field hexnum = SOCBoard4p.class.getDeclaredField("PORTS_HEXNUM_V1");
        hexnum.setAccessible(true);
        out.put("port_hexnums", Json.ints((int[]) hexnum.get(null)));
        out.put("port_edges", Json.ints(b.getPortsEdges()));
        out.put("port_facing", Json.ints(b.getPortsFacing()));
        out.put("resources", Json.map("clay", SOCResourceConstants.CLAY, "ore", SOCResourceConstants.ORE,
            "sheep", SOCResourceConstants.SHEEP, "wheat", SOCResourceConstants.WHEAT,
            "wood", SOCResourceConstants.WOOD, "unknown", SOCResourceConstants.UNKNOWN));
        out.put("dev_cards", Json.map("unknown", SOCDevCardConstants.UNKNOWN, "roads", SOCDevCardConstants.ROADS,
            "disc", SOCDevCardConstants.DISC, "mono", SOCDevCardConstants.MONO, "cap", SOCDevCardConstants.CAP,
            "market", SOCDevCardConstants.MARKET, "univ", SOCDevCardConstants.UNIV,
            "temple", SOCDevCardConstants.TEMPLE, "chapel", SOCDevCardConstants.CHAPEL,
            "knight", SOCDevCardConstants.KNIGHT));
        out.put("pieces", Json.map("road", SOCPlayingPiece.ROAD, "settlement", SOCPlayingPiece.SETTLEMENT,
            "city", SOCPlayingPiece.CITY));
        out.put("hex_types", Json.map("water", SOCBoard.WATER_HEX, "desert", SOCBoard.DESERT_HEX,
            "misc_port_hex", SOCBoard.MISC_PORT_HEX, "misc_port", SOCBoard.MISC_PORT));
        out.put("states", Json.map("START1A", SOCGame.START1A, "START1B", SOCGame.START1B,
            "START2A", SOCGame.START2A, "START2B", SOCGame.START2B, "ROLL_OR_CARD", SOCGame.ROLL_OR_CARD,
            "PLAY1", SOCGame.PLAY1, "PLACING_ROAD", SOCGame.PLACING_ROAD,
            "PLACING_SETTLEMENT", SOCGame.PLACING_SETTLEMENT, "PLACING_CITY", SOCGame.PLACING_CITY,
            "PLACING_ROBBER", SOCGame.PLACING_ROBBER, "PLACING_FREE_ROAD1", SOCGame.PLACING_FREE_ROAD1,
            "PLACING_FREE_ROAD2", SOCGame.PLACING_FREE_ROAD2, "WAITING_FOR_DISCARDS", SOCGame.WAITING_FOR_DISCARDS,
            "WAITING_FOR_ROB_CHOOSE_PLAYER", SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER,
            "WAITING_FOR_DISCOVERY", SOCGame.WAITING_FOR_DISCOVERY,
            "WAITING_FOR_MONOPOLY", SOCGame.WAITING_FOR_MONOPOLY, "OVER", SOCGame.OVER));
        out.put("smart_strategy", SOCRobotDM.SMART_STRATEGY);
        System.out.println(Json.write(out));
    }
}
