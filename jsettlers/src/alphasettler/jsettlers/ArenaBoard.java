// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import soc.game.SOCBoard4p;
import soc.game.SOCGameOptionSet;

/**
 * The classic 4-player board, laid out from the driver's layout instead of JSettlers' shuffle when
 * the game was created with one (ports are encoded in the hex layout, as the v1 encoding does).
 */
final class ArenaBoard extends SOCBoard4p {
    private final transient GameSpec spec;

    ArenaBoard(SOCGameOptionSet opts, GameSpec spec) {
        super(opts);
        this.spec = spec;
    }

    @Override
    public void makeNewBoard(SOCGameOptionSet opts) {
        if (spec == null || !spec.injected()) {
            super.makeNewBoard(opts);
            return;
        }
        setHexLayout(spec.hexes().clone());
        setNumberLayout(spec.numbers().clone());
        setRobberHex(spec.robber(), false);
    }
}
