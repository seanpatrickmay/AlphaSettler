// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.Map;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.robot.OpeningBuildStrategy;

/**
 * Initial placement asked of Python; the stock strategy decides when Python falls back. Our first
 * settlement is kept in the stock field, so a stock second settlement plans around it.
 */
final class BridgeOpening extends OpeningBuildStrategy {
    private final BridgeBrain brain;

    BridgeOpening(SOCGame ga, SOCPlayer pl, BridgeBrain br) {
        super(ga, pl, br);
        brain = br;
    }

    @Override
    public int planInitialSettlements() {
        Map<String, Object> r = brain.ask("setup_settlement", null);
        if (r == null) return super.planInitialSettlements();
        firstSettlement = Json.i(r, "node");
        return firstSettlement;
    }

    @Override
    public int planSecondSettlement() {
        Map<String, Object> r = brain.ask("setup_settlement", null);
        return r != null ? Json.i(r, "node") : super.planSecondSettlement();
    }

    @Override
    public int planInitRoad() {
        Map<String, Object> r = brain.ask("setup_road", null);
        return r != null ? Json.i(r, "edge") : super.planInitRoad();
    }
}
