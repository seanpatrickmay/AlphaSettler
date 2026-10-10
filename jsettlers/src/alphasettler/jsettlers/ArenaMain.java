// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.game.SOCGameOptionSet;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.server.SOCServer;
import soc.util.Version;

/**
 * The harness process: `java ... alphasettler.jsettlers.ArenaMain stock` (or `aligned`: the robots'
 * offers capped to our trading game, {@link CappedBrain}). Starts the server and its robots, says "ready", then reads commands from stdin until it closes.
 */
public final class ArenaMain {
    private ArenaMain() {}

    public static void main(String[] args) {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8);
        System.setOut(System.err);  // JSettlers prints progress to stdout; the pipe owns the real one
        Pipe pipe = new Pipe(out);
        Thread.setDefaultUncaughtExceptionHandler((t, e) -> pipe.fatal("uncaught in " + t.getName() + ": " + stack(e)));
        try {
            run(args.length > 0 ? args[0] : "stock", pipe);
        } catch (Throwable e) {
            pipe.fatal(stack(e));
        }
    }

    private static void run(String mode, Pipe pipe) throws Exception {
        if (!mode.equals("stock") && !mode.equals("aligned"))
            throw new IllegalArgumentException("unknown mode " + mode + " (stock, aligned)");
        SOCRobotBrain.BOTS_ONLY_FAST_PAUSE_FACTOR = 0.01f;
        SOCServer.ROBOT_FORCE_ENDTURN_SECONDS = 60;
        // A stubborn robot's turn (one force-ended before) as long, so our discards on it get the same window.
        SOCServer.ROBOT_FORCE_ENDTURN_STUBBORN_SECONDS = 60;
        SOCGame.boardFactory = new ArenaBoardFactory();
        ArenaServer srv = new ArenaServer(pipe);
        srv.setPriority(5);
        srv.start();
        ServerConnectInfo sci = new ServerConnectInfo(ArenaServer.PORT, ArenaServer.COOKIE);
        List<String> names = new ArrayList<>();
        for (int i = 1; i <= 4; i++) {
            String name = "robot " + i;
            names.add(name);
            start(mode.equals("aligned") ? new CappedClient(sci, name) : new SOCRobotClient(sci, name, "pw"), srv);
        }
        start(new BridgeClient(sci, pipe), srv);
        names.add(ArenaServer.BRIDGE);
        long deadline = System.currentTimeMillis() + 60_000;
        while (!srv.hasRobots(names)) {
            if (System.currentTimeMillis() > deadline) throw new IllegalStateException("robots did not connect: " + names);
            Thread.sleep(50);
        }
        pipe.send(Json.map("t", "ready", "version", Version.version(), "mode", mode, "robots", names));
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            Map<String, Object> m = Json.object(line);
            switch (Json.str(m, "t")) {
                case "game" -> srv.create(GameSpec.parse(m));
                case "reply" -> pipe.reply(m);
                case "abort" -> srv.abort(Json.str(m, "g"));
                case "quit" -> pipe.flushAndExit(0);
                default -> throw new IllegalArgumentException("unknown command: " + line);
            }
        }
        pipe.flushAndExit(0);  // stdin closed: the driver is gone, and so is this JVM
    }

    /** Starts a robot client on its own thread, as SOCLocalRobotClient (package-private) does. */
    static void start(SOCRobotClient client, SOCServer srv) {
        client.knownOpts = new SOCGameOptionSet(srv.knownOpts, true);
        client.printedInitialWelcome = true;
        Thread t = new Thread(client::init, "robot-" + client.getNickname());
        t.setDaemon(true);
        t.start();
    }

    static String stack(Throwable e) {
        StringWriter w = new StringWriter();
        e.printStackTrace(new PrintWriter(w));
        return w.toString();
    }
}
