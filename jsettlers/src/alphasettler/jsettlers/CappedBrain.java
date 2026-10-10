// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import soc.game.SOCGame;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.robot.SOCBuildPlan;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * A stock smart robot whose offers fit our trading game (the aligned arm, spec Section 5): it sends
 * the stock negotiator's offer only when it is one card a side, sends it to every opponent, at most
 * two a turn, and never counter-offers. When the negotiator's offer does not fit, it stops offering
 * (doneTrading) until the stock brain clears that flag, at a new turn or after a declined request
 * mid-turn (resetFieldsAndBuildingPlan); it may then offer again, within the same caps. Everything else
 * is the stock brain.
 */
final class CappedBrain extends SOCRobotBrain {
    static final int MAX_OFFERS = 2;
    private int offersThisTurn;
    /**
     * The game turn {@link #offersThisTurn} counts. Kept by turn number rather than reset at the end of
     * the turn: resetFieldsAtEndTurn runs only when the robot ends its own turn, not when the server
     * force-ends it, and resetFieldsAndBuildingPlan also runs after a declined request mid-turn.
     */
    private int offersTurn = -1;

    CappedBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        super(rc, params, ga, mq);
    }

    @Override
    protected boolean makeOffer(SOCBuildPlan buildPlan) {
        if (offersTurn != game.getTurnCount()) {
            offersTurn = game.getTurnCount();
            offersThisTurn = 0;
        }
        SOCTradeOffer o = offersThisTurn < MAX_OFFERS ? negotiator.makeOffer(buildPlan) : null;
        negotiator.resetWantsAnotherOffer();
        if (o == null || o.getGiveSet().getTotal() != 1 || o.getGetSet().getTotal() != 1) {
            ourPlayerData.setCurrentOffer(null);
            doneTrading = true;
            waitingForTradeResponse = false;
            return false;
        }
        boolean[] to = new boolean[game.maxPlayers];
        for (int pn = 0; pn < game.maxPlayers; pn++) {
            to[pn] = pn != getOurPlayerNumber() && !game.isSeatVacant(pn);
            offerRejections[pn] = false;
        }
        SOCTradeOffer capped = new SOCTradeOffer(game.getName(), getOurPlayerNumber(), to, o.getGiveSet(), o.getGetSet());
        ourPlayerData.setCurrentOffer(capped);
        waitingForTradeResponse = true;
        tradeResponseTimeoutSec = TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
        counter = 0;
        offersThisTurn++;
        client.offerTrade(game, capped);
        return true;
    }

    @Override
    protected boolean makeCounterOffer(SOCTradeOffer offer) {
        return false;
    }
}
