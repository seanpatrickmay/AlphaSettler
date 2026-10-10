// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.robot.SOCBuildPlan;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotDM;
import soc.robot.SOCRobotNegotiator;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * The stock robot brain with its decision hooks asked of Python: initial placement, roll or knight
 * before the roll, the main-phase choice, free roads, the robber and its victim, discards, our
 * offers and our answers to offers. The stock turn machine decides when to ask; a null answer (Python
 * fell back, or no reply in time) lets the stock brain decide that hook. Each reply sets the flags
 * the stock code path it replaces sets, so the stock loop waits for the same server answers.
 */
final class BridgeBrain extends SOCRobotBrain {
    /** Below the server's jsettlers.bots.timeout.turn (600 s), so a slow reply is our counted java-timeout first. */
    static final long ASK_SECONDS = 540;
    static final int STOCK_ROBBER = -2;
    private final Pipe pipe;
    private volatile int victim = STOCK_ROBBER;
    /** Answers by offer (offerer, sets, offer time): the stock brain may consider one offer twice. */
    private final Map<String, Integer> answers = new ConcurrentHashMap<>();

    BridgeBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq, Pipe pipe) {
        super(rc, params, ga, mq);
        this.pipe = pipe;
    }

    /** Python's reply to a decision of {@code kind}, or null when the stock brain decides it. */
    Map<String, Object> ask(String kind, Map<String, Object> extra) {
        Map<String, Object> q = Json.map("t", "ask", "g", game.getName(), "kind", kind, "state", game.getGameState(),
            "extra", extra == null ? Json.map() : extra);
        Map<String, Object> r = pipe.ask(q, ASK_SECONDS);
        if (r == null) {
            fellBack("java-timeout");
            return null;
        }
        return r.containsKey("fallback") ? null : r;
    }

    void fellBack(String reason) {
        pipe.send(Json.map("t", "fallback", "g", game.getName(), "reason", reason));
    }

    int takeVictim() {
        int v = victim;
        victim = STOCK_ROBBER;
        return v;
    }

    static SOCResourceSet set(int[] js5) {
        return new SOCResourceSet(js5[0], js5[1], js5[2], js5[3], js5[4], 0);
    }

    static int[] js5(SOCResourceSet s) {
        return new int[] {s.getAmount(SOCResourceConstants.CLAY), s.getAmount(SOCResourceConstants.ORE),
            s.getAmount(SOCResourceConstants.SHEEP), s.getAmount(SOCResourceConstants.WHEAT),
            s.getAmount(SOCResourceConstants.WOOD)};
    }

    @Override
    protected void setStrategyFields() {
        super.setStrategyFields();
        openingBuildStrategy = new BridgeOpening(game, ourPlayerData, this);
        robberStrategy = new BridgeRobber(game, ourPlayerData, this, rand);
    }

    @Override
    protected SOCRobotDM createDM() {
        return new BridgeDM(this);
    }

    /** Replaces the stock knight-or-roll choice under the stock guard; flags as in playKnightCard / rollDice. */
    @Override
    protected void rollOrPlayKnightOrExpectDice() {
        if (ourTurn && !waitingForOurTurn && !expectPLAY1 && !expectDISCARD && !expectPLACING_ROBBER
            && !(expectDICERESULT && counter < 4000)) {
            Map<String, Object> r = ask("pre_roll", null);
            if (r != null) {
                expectROLL_OR_CARD = false;
                if ("knight".equals(r.get("do"))) {
                    expectPLACING_ROBBER = true;
                    waitingForGameState = true;
                    counter = 0;
                    client.playDevCard(game, SOCDevCardConstants.KNIGHT);
                } else {
                    expectDICERESULT = true;
                    counter = 0;
                    client.rollDice(game);
                }
                return;
            }
        }
        super.rollOrPlayKnightOrExpectDice();
    }

    @Override
    protected void planAndDoActionForPLAY1() {
        Map<String, Object> r = ask("main", null);
        if (r == null) {
            super.planAndDoActionForPLAY1();
            return;
        }
        main(r);
    }

    /** One main-phase reply, with the flags of the stock path it replaces. */
    void main(Map<String, Object> r) {
        switch (Json.str(r, "do")) {
            case "end" -> {  // as planAndDoActionForPLAY1's end of turn
                resetFieldsAtEndTurn();
                client.endTurn(game);
            }
            case "build" -> build(Json.i(r, "piece"), Json.i(r, "coord"));
            case "buy" -> {  // as buildRequestPlannedPiece's CARD
                waitingForDevCard = true;
                client.buyDevCard(game);
            }
            case "knight" -> play(SOCDevCardConstants.KNIGHT);
            case "rb" -> play(SOCDevCardConstants.ROADS);
            case "yop" -> {
                int[] res = Json.intArray(r, "res");
                ((BridgeDM) decisionMaker).choose(res[0], res[1]);
                play(SOCDevCardConstants.DISC);
            }
            case "mono" -> {
                monopolyStrategy.setMonopolyChoice(Json.i(r, "res"));
                play(SOCDevCardConstants.MONO);
            }
            case "bank" -> {  // as buildOrGetResourceByTradeOrCard's tradeWithBank
                waitingForTradeMsg = true;
                counter = 0;
                client.bankTrade(game, set(Json.intArray(r, "give")), set(Json.intArray(r, "get")));
            }
            case "offer" -> {  // as makeOffer: the stock loop waits for the answers, then plans (asks) again
                SOCTradeOffer offer = new SOCTradeOffer(game.getName(), getOurPlayerNumber(), Json.boolArray(r, "to"),
                    set(Json.intArray(r, "give")), set(Json.intArray(r, "get")));
                ourPlayerData.setCurrentOffer(offer);
                for (int pn = 0; pn < game.maxPlayers; pn++) offerRejections[pn] = false;
                waitingForTradeResponse = true;
                tradeResponseTimeoutSec = TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
                counter = 0;
                client.offerTrade(game, offer);
            }
            default -> pipe.fatal("unknown main reply " + r);
        }
    }

    /** As buildOrGetResourceByTradeOrCard's card plays (playKnightCard for a knight). */
    private void play(int card) {
        switch (card) {
            case SOCDevCardConstants.KNIGHT -> expectPLACING_ROBBER = true;
            case SOCDevCardConstants.ROADS -> expectPLACING_FREE_ROAD1 = true;
            case SOCDevCardConstants.DISC -> expectWAITING_FOR_DISCOVERY = true;
            case SOCDevCardConstants.MONO -> expectWAITING_FOR_MONOPOLY = true;
            default -> pipe.fatal("cannot play dev card " + card);
        }
        waitingForGameState = true;
        counter = 0;
        client.playDevCard(game, card);
    }

    /** As buildRequestPlannedPiece: placeIfExpectPlacing then puts whatWeWantToBuild. */
    private void build(int piece, int coord) {
        waitingForGameState = true;
        counter = 0;
        switch (piece) {
            case SOCPlayingPiece.ROAD -> {
                expectPLACING_ROAD = true;
                whatWeWantToBuild = new SOCRoad(ourPlayerData, coord, null);
            }
            case SOCPlayingPiece.SETTLEMENT -> {
                expectPLACING_SETTLEMENT = true;
                whatWeWantToBuild = new SOCSettlement(ourPlayerData, coord, null);
            }
            case SOCPlayingPiece.CITY -> {
                expectPLACING_CITY = true;
                whatWeWantToBuild = new SOCCity(ourPlayerData, coord, null);
            }
            default -> pipe.fatal("unknown piece " + piece);
        }
        client.buildRequest(game, piece);
    }

    /**
     * Free roads are asked of Python. The stock placement takes its roads from a building plan the
     * bridge never made, so a fallback places the first road JSettlers allows.
     */
    @Override
    protected void placeIfExpectPlacing() {
        int gs = game.getGameState();
        if (!waitingForGameState && ourTurn && !waitingForOurTurn
            && (gs == SOCGame.PLACING_FREE_ROAD1 || gs == SOCGame.PLACING_FREE_ROAD2)
            && (expectPLACING_FREE_ROAD1 || expectPLACING_FREE_ROAD2)) {
            Map<String, Object> r = ask("road_building", null);
            int edge = r != null ? Json.i(r, "edge") : anyPotentialRoad();
            expectPLACING_FREE_ROAD1 = false;
            expectPLACING_FREE_ROAD2 = gs == SOCGame.PLACING_FREE_ROAD1;
            if (gs == SOCGame.PLACING_FREE_ROAD2) expectPLAY1 = true;
            waitingForGameState = true;
            counter = 0;
            client.putPiece(game, new SOCRoad(ourPlayerData, edge, null));
            return;
        }
        super.placeIfExpectPlacing();
    }

    private int anyPotentialRoad() {
        for (int e = 0x22; e <= 0xCC; e++) if (ourPlayerData.isPotentialRoad(e)) return e;
        pipe.fatal("no road can be placed for Road Building in " + game.getName());
        return -1;
    }

    @Override
    protected void moveRobber() {
        Map<String, Object> r = ask("move_robber", null);
        if (r == null) {
            victim = STOCK_ROBBER;
            super.moveRobber();
            return;
        }
        victim = Json.i(r, "victim");
        client.moveRobber(game, ourPlayerData, Json.i(r, "hex"));
    }

    @Override
    protected void discard(int numDiscards) {
        Map<String, Object> r = ask("discard", Json.map("n", numDiscards));
        if (r == null) {
            super.discard(numDiscards);
            return;
        }
        client.discard(game, set(Json.intArray(r, "res")));
    }

    /**
     * An offer made to us is asked of Python (which rejects counter-offers and offers outside our trading
     * game unasked, counted); a fallback is the stock negotiator's answer.
     */
    @Override
    protected int considerOffer(SOCTradeOffer offer) {
        if (!offer.getTo()[getOurPlayerNumber()]) return SOCRobotNegotiator.IGNORE_OFFER;
        SOCPlayer from = game.getPlayer(offer.getFrom());
        String key = offer.getFrom() + "|" + offer.getGiveSet() + "|" + offer.getGetSet() + "|" + from.getCurrentOfferTime();
        Integer known = answers.get(key);
        if (known != null) return known;
        List<Object> toList = new ArrayList<>();
        for (boolean b : offer.getTo()) toList.add(b);
        Map<String, Object> r = ask("answer", Json.map("from", offer.getFrom(), "to", toList,
            "give", js5(offer.getGiveSet()), "get", js5(offer.getGetSet())));
        int answer = r == null ? super.considerOffer(offer)
            : (Json.bool(r, "accept") ? SOCRobotNegotiator.ACCEPT_OFFER : SOCRobotNegotiator.REJECT_OFFER);
        answers.put(key, answer);
        return answer;
    }

    /** Our seat offers only what our bot chose (the "offer" reply); the stock negotiator never offers for it. */
    @Override
    protected boolean makeOffer(SOCBuildPlan buildPlan) {
        doneTrading = true;
        return false;
    }

    /** Our rules have no counter-offers: the stock brain then rejects. */
    @Override
    protected boolean makeCounterOffer(SOCTradeOffer offer) {
        return false;
    }

    @Override
    protected void resetFieldsAtEndTurn() {
        super.resetFieldsAtEndTurn();
        answers.clear();
    }
}
