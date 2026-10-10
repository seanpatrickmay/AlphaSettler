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
 * Our seat: a robot connection (so games stay robots-only and run at the fast pace) named
 * "robot bridge" (so its fallback decisions use the smart parameters), with a third-party brain
 * class (so the long jsettlers.bots.timeout.turn applies to its turns).
 */
final class BridgeClient extends SOCRobotClient {
    private final Pipe pipe;

    BridgeClient(ServerConnectInfo sci, Pipe pipe) {
        super(sci, ArenaServer.BRIDGE, "pw");
        this.pipe = pipe;
        rbclass = BridgeClient.class.getName();
    }

    @Override
    public SOCRobotBrain createBrain(SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        return new BridgeBrain(this, params, ga, mq, pipe);
    }
}
