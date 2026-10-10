// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * JSON lines to the Python driver on the real stdout (JSettlers' own prints go to stderr), written
 * in the order {@link #send} was called by one writer thread, and asks that block their caller
 * until the matching reply arrives on stdin.
 */
final class Pipe {
    private final PrintStream out;
    private final LinkedBlockingQueue<String> queue = new LinkedBlockingQueue<>();
    private final Map<Long, CompletableFuture<Map<String, Object>>> waiting = new ConcurrentHashMap<>();
    private final AtomicLong ids = new AtomicLong();

    Pipe(PrintStream out) {
        this.out = out;
        Thread w = new Thread(this::drain, "pipe-writer");
        w.setDaemon(true);
        w.start();
    }

    void send(Map<String, Object> m) {
        queue.add(Json.write(m));
    }

    private void drain() {
        try {
            while (true) {
                String line = queue.take();
                synchronized (out) {
                    out.println(line);
                    out.flush();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Sends {@code m} with a fresh "id" and waits for its reply; null if none came within {@code seconds}. */
    Map<String, Object> ask(Map<String, Object> m, long seconds) {
        long id = ids.incrementAndGet();
        CompletableFuture<Map<String, Object>> f = new CompletableFuture<>();
        waiting.put(id, f);
        m.put("id", id);
        send(m);
        try {
            return f.get(seconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (ExecutionException e) {
            throw new IllegalStateException(e);
        } finally {
            waiting.remove(id);
        }
    }

    /** A reply from Python; one whose ask already timed out is dropped (the asker has fallen back). */
    void reply(Map<String, Object> m) {
        CompletableFuture<Map<String, Object>> f = waiting.get(Json.num(m, "id"));
        if (f != null) f.complete(m);
    }

    /** Writes what was queued, then exits with {@code code}. */
    void flushAndExit(int code) {
        long deadline = System.currentTimeMillis() + 5000;
        while (!queue.isEmpty() && System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        synchronized (out) {
            out.flush();
        }
        Runtime.getRuntime().halt(code);
    }

    /** Reports a failure to Python and the log, then exits 1. */
    void fatal(String what) {
        System.err.println("arena harness failed: " + what);
        synchronized (out) {
            out.println(Json.write(Json.map("t", "error", "what", what)));
            out.flush();
        }
        Runtime.getRuntime().halt(1);
    }
}
