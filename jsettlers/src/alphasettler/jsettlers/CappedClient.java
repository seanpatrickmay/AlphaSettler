// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * A robot named "robot N" (smart parameters) whose brain is {@link CappedBrain}. Unlike {@link BridgeClient}
 * it reports the default (built-in) brain class to the server, which uses that class for a stalled turn's
 * timeout (built-in: ROBOT_FORCE_ENDTURN_SECONDS, 60 s here; third-party: jsettlers.bots.timeout.turn,
 * 600 s, which our seat needs), the login version check (same jar) and its third-party join pool (unused
 * by the arena, which seats robots itself); so the aligned arm's stalled robot turns end after 60 s, as
 * the stock robots' do.
 */
final class CappedClient extends SOCRobotClient {
    CappedClient(ServerConnectInfo sci, String name) {
        super(sci, name, "pw");
    }

    @Override
    public SOCRobotBrain createBrain(SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        return new CappedBrain(this, params, ga, mq);
    }
}
