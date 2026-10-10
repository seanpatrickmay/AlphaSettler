// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import soc.game.SOCBoard;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;

/** The server-side game in JSettlers' own coordinates and codes; oracle/jsettlers/translate.py maps it. */
final class Snapshots {
    private static final Field DECK;

    static {
        try {
            DECK = SOCGame.class.getDeclaredField("devCardDeck");
            DECK.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private Snapshots() {}

    static Map<String, Object> raw(SOCGame ga) {
        List<Object> players = new ArrayList<>();
        for (int pn = 0; pn < ga.maxPlayers; pn++) {
            SOCPlayer pl = ga.getPlayer(pn);
            SOCResourceSet r = pl.getResources();
            int[] res = {r.getAmount(SOCResourceConstants.CLAY), r.getAmount(SOCResourceConstants.ORE),
                r.getAmount(SOCResourceConstants.SHEEP), r.getAmount(SOCResourceConstants.WHEAT),
                r.getAmount(SOCResourceConstants.WOOD)};
            int[] dev = new int[10];
            for (int c = 0; c < 10; c++) dev[c] = pl.getInventory().getAmount(c);
            int[] played = {pl.getNumKnights(), pl.numRBCards, pl.numDISCCards, pl.numMONOCards};
            players.add(Json.map("res", res, "dev", dev, "played", played,
                "settlements", coords(pl.getSettlements()), "cities", coords(pl.getCities()),
                "roads", coords(pl.getRoadsAndShips()),
                "discard", pl.getNeedToDiscard() ? pl.getCountToDiscard() : 0,
                "played_dev", pl.hasPlayedDevCard(), "vp", pl.getTotalVP()));
        }
        return Json.map("state", ga.getGameState(), "current", ga.getCurrentPlayerNumber(),
            "robber", ga.getBoard().getRobberHex(), "deck", deck(ga), "players", players,
            "longest", pn(ga.getPlayerWithLongestRoad()), "largest", pn(ga.getPlayerWithLargestArmy()),
            "winner", pn(ga.getPlayerWithWin()));
    }

    static Map<String, Object> board(SOCGame ga) {
        SOCBoard b = ga.getBoard();
        List<Object> hexes = new ArrayList<>();
        for (int h : b.getLandHexCoords()) hexes.add(Json.list(h, b.getHexTypeFromCoord(h), b.getNumberOnHexFromCoord(h)));
        List<Object> ports = new ArrayList<>();
        for (int e : b.getPortsEdges()) ports.add(Json.list(e, b.getPortTypeFromNodeCoord(b.getAdjacentNodesToEdge_arr(e)[0])));
        return Json.map("hexes", hexes, "ports", ports, "robber", b.getRobberHex());
    }

    /** The undrawn cards, the next draw first (SOCGame draws devCardDeck[numDevCards - 1]). */
    static List<Object> deck(SOCGame ga) {
        List<Object> out = new ArrayList<>();
        try {
            int[] d = (int[]) DECK.get(ga);
            if (d == null) return out;
            for (int i = ga.getNumDevCards() - 1; i >= 0; i--) out.add(d[i]);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        return out;
    }

    private static int pn(SOCPlayer p) {
        return p == null ? -1 : p.getPlayerNumber();
    }

    private static List<Object> coords(Collection<? extends SOCPlayingPiece> pieces) {
        List<Object> out = new ArrayList<>();
        pieces.stream().map(SOCPlayingPiece::getCoordinates).sorted().forEach(out::add);
        return out;
    }
}
