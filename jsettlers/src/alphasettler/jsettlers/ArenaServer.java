// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import soc.game.SOCGame;
import soc.game.SOCGameOption;
import soc.game.SOCGameOptionSet;
import soc.message.SOCMessage;
import soc.message.SOCMessageForGame;
import soc.server.SOCServer;
import soc.server.genericServer.Connection;

/**
 * JSettlers' own server on a string port, recording every game event (with its audience) to the
 * pipe, plus the server-side position whenever it changed. Games are created with the arena's
 * options, board, routed random source and seats.
 */
final class ArenaServer extends SOCServer {
    static final String PORT = "alphasettler-arena";
    static final String COOKIE = "arena";
    static final String BRIDGE = "robot bridge";
    private static final Field RAND;
    private static final Method ASK_ROBOTS;

    static {
        try {
            RAND = SOCGame.class.getDeclaredField("rand");
            RAND.setAccessible(true);
            ASK_ROBOTS = SOCServer.class.getDeclaredMethod("readyGameAskRobotsJoin",
                SOCGame.class, boolean[].class, Connection[].class, int.class);
            ASK_ROBOTS.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final Pipe pipe;
    private final Map<String, GameSpec> specs = new ConcurrentHashMap<>();
    private final Map<String, String> lastRaw = new ConcurrentHashMap<>();
    private final Set<String> started = ConcurrentHashMap.newKeySet();
    private final Set<String> finished = ConcurrentHashMap.newKeySet();
    private volatile GameSpec creating;
    /** Serializes recording and aborting (see {@link #record}). */
    private final Object recording = new Object();

    ArenaServer(Pipe pipe) {
        super(PORT, props());
        this.pipe = pipe;
    }

    static Properties props() {
        Properties p = new Properties();
        p.setProperty(SOCServer.PROP_JSETTLERS_STARTROBOTS, "0");
        p.setProperty(SOCServer.PROP_JSETTLERS_BOTS_COOKIE, COOKIE);
        p.setProperty(SOCServer.PROP_JSETTLERS_BOTS_TIMEOUT_TURN, "600");
        return p;
    }

    /** The robot connections' names, once every one of {@code names} has logged in. */
    boolean hasRobots(List<String> names) {
        List<String> have = new ArrayList<>();
        synchronized (robots) {
            for (Connection c : robots) have.add(c.getData());
        }
        return have.containsAll(names);
    }

    private Connection robot(String name) {
        synchronized (robots) {
            for (Connection c : robots) if (name.equals(c.getData())) return c;
        }
        throw new IllegalStateException("no robot named " + name + " is connected");
    }

    void create(GameSpec spec) throws ReflectiveOperationException {
        // A finished game's name stays taken: its late events are dropped by name, so a reused name would go silent.
        if (specs.containsKey(spec.name()) || finished.contains(spec.name()))
            throw new IllegalArgumentException("game name already used: " + spec.name());
        SOCGameOptionSet opts = new SOCGameOptionSet();
        for (SOCGameOption opt : knownOpts) {
            if (!((opt.key.charAt(0) == '_') || opt.hasFlag(SOCGameOption.FLAG_INACTIVE_HIDDEN)
                  || opt.hasFlag(SOCGameOption.FLAG_INTERNAL_GAME_PROPERTY)))
                opts.put(opt);
        }
        opts = new SOCGameOptionSet(opts, true);
        opts.get("PL").setIntValue(4);
        if (spec.injected()) opts.get("BC").setBoolValue(false);
        opts.get("NT").setBoolValue(spec.noTrades());
        specs.put(spec.name(), spec);
        creating = spec;
        ArenaBoardFactory.pending = spec;
        SOCGame ga;
        try {
            ga = createGameAndBroadcast(null, spec.name(), opts, true);
        } finally {
            ArenaBoardFactory.pending = null;
            creating = null;
        }
        if (ga == null) throw new IllegalStateException("the server did not create game " + spec.name());
        if (!(ga.getBoard() instanceof ArenaBoard)) throw new IllegalStateException("game " + spec.name() + " did not get the arena's board");
        if (!(RAND.get(ga) instanceof RoutedRandom)) throw new IllegalStateException("game " + spec.name() + " did not get the routed random source");
        String[] names = spec.bridge()
            ? new String[] {BRIDGE, "robot 1", "robot 2", "robot 3"}
            : new String[] {"robot 1", "robot 2", "robot 3", "robot 4"};
        Connection[] seats = new Connection[4];
        for (int i = 0; i < 4; i++) seats[i] = robot(names[i]);
        ga.setGameState(SOCGame.READY);
        if (!(Boolean) ASK_ROBOTS.invoke(this, ga, null, seats, 0))
            throw new IllegalStateException("no robots could join " + spec.name());
    }

    void abort(String g) {
        SOCGame ga = getGame(g);
        if (ga == null) return;
        synchronized (recording) {  // no event of g is queued after its "over"
            if (!finished.add(g)) return;
            pipe.send(over(ga, true));
        }
        destroyGameAndBroadcast(g, "arena abort");  // outside the lock: it takes JSettlers' own locks
        forget(g);
    }

    @Override
    public boolean isRecordGameEventsActive() {
        return true;
    }

    @Override
    public boolean isRecordGameEventsFromClientsActive() {
        return true;
    }

    @Override
    protected void startEmptyLog(SOCGame game, boolean isReset) {
        GameSpec spec = creating;
        if (spec == null || !spec.name().equals(game.getName()))
            throw new IllegalStateException("game " + game.getName() + " was not created by the arena");
        try {
            RAND.set(game, new RoutedRandom(spec));
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void recordGameEvent(String g, SOCMessage m) {
        record(g, "all", m);
    }

    @Override
    public void recordGameEventTo(String g, int pn, SOCMessage m) {
        record(g, audience(pn), m);
    }

    @Override
    public void recordGameEventNotTo(String g, int pn, SOCMessage m) {
        record(g, "!p" + pn, m);
    }

    @Override
    public void recordGameEventNotTo(String g, int[] pns, SOCMessage m) {
        record(g, "!p" + Arrays.toString(pns), m);
    }

    @Override
    public void recordClientMessage(String g, int pn, SOCMessageForGame m) {
        record(g, pn == PN_OBSERVER ? "fo" : "f" + pn, m);
    }

    private static String audience(int pn) {
        if (pn == PN_OBSERVER) return "ob";
        if (pn == PN_REPLY_TO_UNDETERMINED) return "un";
        return "p" + pn;
    }

    /**
     * Serialized by {@link #recording}: the server's treater thread and {@code SOCForceEndTurnThread} (which holds
     * the game's monitor) can both record events of one game, so the start block, the raw position, its comparison
     * with the last one and the queued lines are one atomic step per event, in the order recorded. The game's own
     * monitor is not taken ({@code takeMonitor} is not reentrant, and most callers already hold it); the game is
     * looked up outside the lock, and nothing under it waits on a JSettlers lock.
     */
    private void record(String g, String aud, Object m) {
        if (!specs.containsKey(g)) return;
        SOCGame ga = getGame(g);
        if (ga == null) return;
        synchronized (recording) {
            if (finished.contains(g)) return;
            int state = ga.getGameState();
            if (!started.contains(g) && (state < SOCGame.START1A || state >= SOCGame.OVER))
                return;  // joins and sit-downs before the game
            if (started.add(g)) {
                Map<String, Object> raw = Snapshots.raw(ga);
                lastRaw.put(g, Json.write(raw));
                List<Object> seats = new ArrayList<>(), smart = new ArrayList<>();
                for (int pn = 0; pn < 4; pn++) {
                    String name = ga.getPlayer(pn).getName();
                    seats.add(name);
                    smart.add(getRobotParameters(name).getStrategyType());
                }
                pipe.send(Json.map("t", "start", "g", g, "first", ga.getFirstPlayer(), "seats", seats, "smart", smart,
                    "board", Snapshots.board(ga), "s", raw));
            }
            Map<String, Object> ev = Json.map("t", "ev", "g", g, "a", aud, "m", String.valueOf(m));
            String raw = Json.write(Snapshots.raw(ga));
            if (!raw.equals(lastRaw.get(g))) {
                ev.put("s", new Json.Raw(raw));
                lastRaw.put(g, raw);
            }
            pipe.send(ev);
            if (state != SOCGame.OVER || !finished.add(g)) return;
            pipe.send(over(ga, false));
        }
        Thread t = new Thread(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            destroyGameAndBroadcast(g, "arena game over");
            forget(g);
        }, "arena-destroy");
        t.setDaemon(true);
        t.start();
    }

    private void forget(String g) {
        specs.remove(g);
        lastRaw.remove(g);
        started.remove(g);
    }

    private static Map<String, Object> over(SOCGame ga, boolean aborted) {
        int[] vp = new int[4];
        for (int pn = 0; pn < 4; pn++) vp[pn] = ga.getPlayer(pn).getTotalVP();
        int winner = aborted || ga.getPlayerWithWin() == null ? -1 : ga.getPlayerWithWin().getPlayerNumber();
        return Json.map("t", "over", "g", ga.getName(), "winner", winner, "vp", vp, "rounds", ga.getRoundCount(),
            "aborted", aborted);
    }
}
