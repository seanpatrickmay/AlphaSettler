// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.

package alphasettler.jsettlers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The minimal JSON the pipe needs: objects as insertion-ordered maps, arrays as lists,
 * integers as {@code Long}, other numbers as {@code Double}, strings, booleans and null.
 * {@link Raw} carries text that is already JSON.
 */
public final class Json {
    private Json() {}

    /** Already-serialized JSON, written as is. */
    public record Raw(String json) {}

    public static Map<String, Object> map(Object... kv) {
        if (kv.length % 2 != 0) throw new IllegalArgumentException("map needs key/value pairs");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    public static List<Object> list(Object... xs) {
        return new ArrayList<>(Arrays.asList(xs));
    }

    public static List<Object> ints(int[] xs) {
        List<Object> out = new ArrayList<>(xs.length);
        for (int x : xs) out.add(x);
        return out;
    }

    public static String write(Object v) {
        StringBuilder sb = new StringBuilder();
        write(sb, v);
        return sb.toString();
    }

    private static void write(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("null");
        } else if (v instanceof Raw r) {
            sb.append(r.json());
        } else if (v instanceof String s) {
            quote(sb, s);
        } else if (v instanceof Boolean || v instanceof Integer || v instanceof Long) {
            sb.append(v);
        } else if (v instanceof Double d) {
            if (d.isNaN() || d.isInfinite()) throw new IllegalArgumentException("non-finite number");
            sb.append(d);
        } else if (v instanceof int[] a) {
            write(sb, ints(a));
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                quote(sb, (String) e.getKey());
                sb.append(':');
                write(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) sb.append(',');
                write(sb, l.get(i));
            }
            sb.append(']');
        } else {
            throw new IllegalArgumentException("cannot write " + v.getClass().getName());
        }
    }

    private static void quote(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        sb.append('"');
    }

    public static Object parse(String s) {
        int[] at = {0};
        Object v = value(s, at);
        skip(s, at);
        if (at[0] != s.length()) throw new IllegalArgumentException("trailing text in JSON at " + at[0]);
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(String s) {
        Object v = parse(s);
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected a JSON object: " + s);
        return (Map<String, Object>) v;
    }

    private static void skip(String s, int[] at) {
        while (at[0] < s.length() && Character.isWhitespace(s.charAt(at[0]))) at[0]++;
    }

    private static Object value(String s, int[] at) {
        skip(s, at);
        if (at[0] >= s.length()) throw new IllegalArgumentException("unexpected end of JSON");
        char c = s.charAt(at[0]);
        if (c == '{') {
            at[0]++;
            Map<String, Object> m = new LinkedHashMap<>();
            skip(s, at);
            if (s.charAt(at[0]) == '}') { at[0]++; return m; }
            while (true) {
                skip(s, at);
                Object k = value(s, at);
                if (!(k instanceof String key)) throw new IllegalArgumentException("object key is not a string");
                skip(s, at);
                expect(s, at, ':');
                m.put(key, value(s, at));
                skip(s, at);
                if (s.charAt(at[0]) == ',') { at[0]++; continue; }
                expect(s, at, '}');
                return m;
            }
        }
        if (c == '[') {
            at[0]++;
            List<Object> l = new ArrayList<>();
            skip(s, at);
            if (s.charAt(at[0]) == ']') { at[0]++; return l; }
            while (true) {
                l.add(value(s, at));
                skip(s, at);
                if (s.charAt(at[0]) == ',') { at[0]++; continue; }
                expect(s, at, ']');
                return l;
            }
        }
        if (c == '"') return string(s, at);
        if (s.startsWith("true", at[0])) { at[0] += 4; return Boolean.TRUE; }
        if (s.startsWith("false", at[0])) { at[0] += 5; return Boolean.FALSE; }
        if (s.startsWith("null", at[0])) { at[0] += 4; return null; }
        int start = at[0];
        while (at[0] < s.length() && "+-0123456789.eE".indexOf(s.charAt(at[0])) >= 0) at[0]++;
        String num = s.substring(start, at[0]);
        if (num.isEmpty()) throw new IllegalArgumentException("bad JSON at " + start);
        if (num.contains(".") || num.contains("e") || num.contains("E")) return Double.parseDouble(num);
        return Long.parseLong(num);
    }

    private static void expect(String s, int[] at, char c) {
        if (at[0] >= s.length() || s.charAt(at[0]) != c)
            throw new IllegalArgumentException("expected '" + c + "' in JSON at " + at[0]);
        at[0]++;
    }

    private static String string(String s, int[] at) {
        expect(s, at, '"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = s.charAt(at[0]++);
            if (c == '"') return sb.toString();
            if (c != '\\') { sb.append(c); continue; }
            char e = s.charAt(at[0]++);
            switch (e) {
                case '"', '\\', '/' -> sb.append(e);
                case 'n' -> sb.append('\n');
                case 'r' -> sb.append('\r');
                case 't' -> sb.append('\t');
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'u' -> { sb.append((char) Integer.parseInt(s.substring(at[0], at[0] + 4), 16)); at[0] += 4; }
                default -> throw new IllegalArgumentException("bad escape \\" + e);
            }
        }
    }

    // Typed reads of a parsed object; each fails loudly on a missing or mistyped key.

    public static long num(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof Long l)) throw new IllegalArgumentException("expected an integer at " + k + ": " + m);
        return l;
    }

    public static int i(Map<String, Object> m, String k) {
        return Math.toIntExact(num(m, k));
    }

    public static boolean bool(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof Boolean b)) throw new IllegalArgumentException("expected a boolean at " + k + ": " + m);
        return b;
    }

    public static String str(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof String s)) throw new IllegalArgumentException("expected a string at " + k + ": " + m);
        return s;
    }

    public static int[] intArray(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof List<?> l)) throw new IllegalArgumentException("expected a list at " + k + ": " + m);
        int[] out = new int[l.size()];
        for (int j = 0; j < out.length; j++) out[j] = Math.toIntExact((Long) l.get(j));
        return out;
    }

    public static boolean[] boolArray(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof List<?> l)) throw new IllegalArgumentException("expected a list at " + k + ": " + m);
        boolean[] out = new boolean[l.size()];
        for (int j = 0; j < out.length; j++) out[j] = (Boolean) l.get(j);
        return out;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> sub(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) return null;
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected an object at " + k + ": " + m);
        return (Map<String, Object>) v;
    }
}
