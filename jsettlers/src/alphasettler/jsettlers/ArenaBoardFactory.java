// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import soc.game.SOCBoard;
import soc.game.SOCGameOptionSet;
import soc.server.SOCBoardAtServer;

/**
 * Installed as {@code SOCGame.boardFactory} before the server exists (it must extend
 * BoardFactoryAtServer, or SOCGameListAtServer replaces it). The arena sets {@link #pending} around
 * its own createGameAndBroadcast call, which builds the game's board on the same thread; every other
 * board (the robots' copies of the game) is JSettlers' usual one.
 */
final class ArenaBoardFactory extends SOCBoardAtServer.BoardFactoryAtServer {
    static volatile GameSpec pending;

    @Override
    public SOCBoard createBoard(SOCGameOptionSet opts, boolean largeBoard, int maxPlayers) {
        GameSpec spec = pending;
        if (!largeBoard && maxPlayers == 4 && spec != null) return new ArenaBoard(opts, spec);
        return super.createBoard(opts, largeBoard, maxPlayers);
    }
}
