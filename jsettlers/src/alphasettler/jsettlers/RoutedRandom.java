// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.Random;

/**
 * Replaces {@code SOCGame.rand} (a private field, set by reflection before the game starts).
 * {@code nextInt()} is routed by its caller's method name: {@code startGame} (the first-player
 * draw) gets the scripted seat, {@code rollDice} draws each die from the seed's dice stream,
 * {@code startGame_setupDevCards} shuffles the deck from the deck stream, and every other caller
 * (the steals) uses the misc stream. So a seed's k-th roll and its deck order are the same in
 * every rotation and arm, whatever the robots do in between.
 */
final class RoutedRandom extends Random {
    private final int first;
    private final Random dice;
    private final Random deck;
    private final Random misc;

    RoutedRandom(GameSpec spec) {
        super(spec.misc());
        first = spec.first();
        dice = new Random(spec.dice());
        deck = new Random(spec.deck());
        misc = new Random(spec.misc());
    }

    @Override
    public int nextInt() {
        String caller = StackWalker.getInstance()
            .walk(frames -> frames.skip(1).findFirst().map(StackWalker.StackFrame::getMethodName).orElse(""));
        return switch (caller) {
            case "startGame" -> first;
            case "rollDice" -> dice.nextInt(6);
            case "startGame_setupDevCards" -> deck.nextInt(Integer.MAX_VALUE);
            default -> misc.nextInt(Integer.MAX_VALUE);
        };
    }
}
