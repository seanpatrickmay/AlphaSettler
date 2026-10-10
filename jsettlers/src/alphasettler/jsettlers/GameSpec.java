// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.Map;

/**
 * One game the driver asked for: its name, the injected board (or null for JSettlers' own), the
 * scripted first player, the seeds of the dice, deck and misc streams, the no-trading option and
 * whether JSettlers seat 0 is the bridge.
 */
record GameSpec(String name, int[] hexes, int[] numbers, int robber, int first, long dice, long deck, long misc,
                boolean noTrades, boolean bridge) {

    static GameSpec parse(Map<String, Object> m) {
        Map<String, Object> layout = Json.sub(m, "layout");
        int first = Json.i(m, "first");
        if (first < 0 || first > 3) throw new IllegalArgumentException("first player must be 0..3: " + m);
        int[] hexes = layout == null ? null : Json.intArray(layout, "hexes");
        int[] numbers = layout == null ? null : Json.intArray(layout, "numbers");
        if (hexes != null && (hexes.length != 37 || numbers.length != 37))
            throw new IllegalArgumentException("a layout has 37 hexes and 37 numbers: " + m);
        return new GameSpec(Json.str(m, "g"), hexes, numbers, layout == null ? -1 : Json.i(layout, "robber"), first,
            Json.num(m, "dice"), Json.num(m, "deck"), Json.num(m, "misc"), Json.bool(m, "nt"), Json.bool(m, "bridge"));
    }

    boolean injected() {
        return hexes != null;
    }
}
