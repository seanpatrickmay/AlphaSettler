// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.Random;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.robot.RobberStrategy;

/** The victim our bot chose with its robber move, when JSettlers asks for one. */
final class BridgeRobber extends RobberStrategy {
    private final BridgeBrain brain;

    BridgeRobber(SOCGame ga, SOCPlayer pl, BridgeBrain br, Random rand) {
        super(ga, pl, br, rand);
        brain = br;
    }

    @Override
    public int chooseRobberVictim(boolean[] isVictim, boolean canChooseNone) {
        int v = brain.takeVictim();
        if (v >= 0 && v < isVictim.length && isVictim[v]) return v;
        if (v != BridgeBrain.STOCK_ROBBER) brain.fellBack("victim");
        return super.chooseRobberVictim(isVictim, canChooseNone);
    }
}
