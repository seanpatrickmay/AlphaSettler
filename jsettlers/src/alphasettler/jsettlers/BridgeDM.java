// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import soc.game.SOCResourceSet;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotDM;

/** The stock decision maker, plus our Year of Plenty picks (sent by the stock WAITING_FOR_DISCOVERY step). */
final class BridgeDM extends SOCRobotDM {
    BridgeDM(SOCRobotBrain br) {
        super(br);
    }

    void choose(int a, int b) {
        resourceChoices = new SOCResourceSet();
        resourceChoices.add(1, a);
        resourceChoices.add(1, b);
    }
}
