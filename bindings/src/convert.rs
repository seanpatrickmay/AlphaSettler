//! Conversions between engine types and plain Python values.

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};
use settler_engine::topology::{NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::*;

pub fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// A Python int as `i128`. An int too large even for that is a ValueError; a non-int stays a
/// TypeError. A bool is a TypeError too, although Python treats it as an int.
fn py_int(v: &Bound<'_, PyAny>, what: &str) -> PyResult<i128> {
    if v.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{what} must be an int, not bool"
        )));
    }
    v.extract::<i128>().map_err(|e| {
        if e.is_instance_of::<PyOverflowError>(v.py()) {
            value_error(format!("{what} {v} out of range"))
        } else {
            e
        }
    })
}

/// A Python int converted to `T`; out-of-range values raise ValueError instead of
/// PyO3's OverflowError.
pub fn int_arg<T: TryFrom<i128>>(v: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    let wide = py_int(v, what)?;
    T::try_from(wide).map_err(|_| {
        value_error(format!(
            "{what} {wide} out of range for {}",
            std::any::type_name::<T>()
        ))
    })
}

/// A Python int in `0..end`, as `usize`.
fn index_arg(v: &Bound<'_, PyAny>, end: usize, what: &str) -> PyResult<usize> {
    let wide = py_int(v, what)?;
    if !(0..end as i128).contains(&wide) {
        return Err(value_error(format!("{what} {wide} out of range 0..{end}")));
    }
    Ok(wide as usize)
}

pub fn viewer_arg(v: &Bound<'_, PyAny>) -> PyResult<PlayerId> {
    Ok(index_arg(v, NUM_PLAYERS, "viewer")? as PlayerId)
}

/// A Python sequence (not a str) of exactly `n` items.
fn seq_arg<'py>(v: &Bound<'py, PyAny>, n: usize, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let items: Vec<Bound<'py, PyAny>> = v.extract()?;
    if items.len() != n {
        return Err(value_error(format!(
            "{what} must have {n} items, got {}",
            items.len()
        )));
    }
    Ok(items)
}

pub fn parse_config(config: Option<&Bound<'_, PyDict>>) -> PyResult<GameConfig> {
    let mut c = GameConfig::default();
    if let Some(d) = config {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "vp_to_win" => c.vp_to_win = int_arg(&v, "vp_to_win")?,
                "discard_limit" => c.discard_limit = int_arg(&v, "discard_limit")?,
                "max_offers_per_turn" => {
                    c.max_offers_per_turn = int_arg(&v, "max_offers_per_turn")?
                }
                "max_trade_cards" => c.max_trade_cards = int_arg(&v, "max_trade_cards")?,
                "max_turns" => c.max_turns = int_arg(&v, "max_turns")?,
                "catanatron_compat" => c.catanatron_compat = v.extract()?,
                other => return Err(value_error(format!("unknown config key {other:?}"))),
            }
        }
    }
    c.validate().map_err(value_error)?;
    Ok(c)
}

pub fn config_dict<'py>(py: Python<'py>, c: &GameConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("vp_to_win", c.vp_to_win)?;
    d.set_item("discard_limit", c.discard_limit)?;
    d.set_item("max_offers_per_turn", c.max_offers_per_turn)?;
    d.set_item("max_trade_cards", c.max_trade_cards)?;
    d.set_item("max_turns", c.max_turns)?;
    d.set_item("catanatron_compat", c.catanatron_compat)?;
    Ok(d)
}

pub fn action_from_id(v: &Bound<'_, PyAny>) -> PyResult<Action> {
    let id = index_arg(v, ACTION_SPACE_SIZE, "action id")?;
    Action::decode(id as u16)
        .ok_or_else(|| value_error(format!("action id {id} is not a valid action")))
}

pub fn counts(xs: &[u8]) -> Vec<u32> {
    xs.iter().map(|&x| x as u32).collect()
}

pub fn node_list(mask: u64) -> Vec<u32> {
    bits64(mask).map(|n| n as u32).collect()
}

pub fn edge_list(mask: u128) -> Vec<u32> {
    bits128(mask).map(|e| e as u32).collect()
}

pub fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::SetupSettlement => "setup_settlement",
        Phase::SetupRoad { .. } => "setup_road",
        Phase::PreRoll => "pre_roll",
        Phase::Discard => "discard",
        Phase::MoveRobber => "move_robber",
        Phase::Steal => "steal",
        Phase::Main => "main",
        Phase::RoadBuilding { .. } => "road_building",
        Phase::TradeResponse => "trade_response",
        Phase::TradeConfirm => "trade_confirm",
        Phase::GameOver { .. } => "game_over",
    }
}

const RESOURCE_NAMES: [&str; 5] = ["wood", "brick", "sheep", "wheat", "ore"];
const DEV_NAMES: [&str; 5] = [
    "knight",
    "victory_point",
    "road_building",
    "year_of_plenty",
    "monopoly",
];

pub fn resource_name(r: Resource) -> &'static str {
    RESOURCE_NAMES[r.index()]
}

pub fn dev_name(c: DevCard) -> &'static str {
    DEV_NAMES[c.index()]
}

fn parse_resource(name: &str) -> PyResult<Resource> {
    RESOURCE_NAMES
        .iter()
        .position(|&n| n == name)
        .map(Resource::from_index)
        .ok_or_else(|| value_error(format!("unknown resource {name:?}")))
}

fn parse_dev(name: &str) -> PyResult<DevCard> {
    DEV_NAMES
        .iter()
        .position(|&n| n == name)
        .map(DevCard::from_index)
        .ok_or_else(|| value_error(format!("unknown dev card {name:?}")))
}

pub fn parse_chance(d: &Bound<'_, PyDict>) -> PyResult<Chance> {
    for k in d.keys() {
        let key: String = k.extract()?;
        if !matches!(key.as_str(), "roll" | "discards" | "steal" | "dev") {
            return Err(value_error(format!("unknown chance key {key:?}")));
        }
    }
    let discards = d.get_item("discards")?;
    let keys = (
        d.get_item("roll")?,
        d.get_item("steal")?,
        d.get_item("dev")?,
    );
    if discards.is_some() && keys.0.is_none() {
        return Err(value_error("'discards' is only valid together with 'roll'"));
    }
    match keys {
        (Some(roll), None, None) => {
            let dice = seq_arg(&roll, 2, "roll")?;
            let dice = (int_arg(&dice[0], "die")?, int_arg(&dice[1], "die")?);
            let discards = match discards {
                None => None,
                Some(v) if v.is_none() => None,
                Some(v) => {
                    let mut hands = [[0u8; 5]; NUM_PLAYERS];
                    let rows = seq_arg(&v, NUM_PLAYERS, "discards")?;
                    for (hand, row) in hands.iter_mut().zip(rows) {
                        for (n, x) in hand.iter_mut().zip(seq_arg(&row, 5, "discard hand")?) {
                            *n = int_arg(&x, "discard count")?;
                        }
                    }
                    Some(hands)
                }
            };
            Ok(Chance::Roll { dice, discards })
        }
        (None, Some(r), None) => Ok(Chance::Steal(parse_resource(&r.extract::<String>()?)?)),
        (None, None, Some(c)) => Ok(Chance::Dev(parse_dev(&c.extract::<String>()?)?)),
        _ => Err(value_error(
            "chance must have exactly one of the keys 'roll', 'steal', 'dev'",
        )),
    }
}

fn trade_dict<'py>(py: Python<'py>, t: &PendingTrade) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("give", counts(&t.give))?;
    d.set_item("get", counts(&t.get))?;
    let responses: Vec<&str> = t
        .responses
        .iter()
        .map(|r| match r {
            Response::Pending => "pending",
            Response::Accepted => "accepted",
            Response::Rejected => "rejected",
        })
        .collect();
    d.set_item("responses", responses)?;
    d.set_item("next_responder", t.next_responder)?;
    Ok(d)
}

fn board_dict<'py>(py: Python<'py>, b: &Board) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let tiles: Vec<Option<&str>> = b
        .tile_resource
        .iter()
        .map(|r| r.map(resource_name))
        .collect();
    d.set_item("tile_resource", tiles)?;
    d.set_item("tile_number", counts(&b.tile_number))?;
    let ports = PyList::empty(py);
    for &(edge, kind) in &b.ports {
        let k = match kind {
            PortKind::Generic => "generic",
            PortKind::Specific(r) => resource_name(r),
        };
        ports.append((edge as u32, k))?;
    }
    d.set_item("ports", ports)?;
    Ok(d)
}

pub fn observation_dict<'py>(py: Python<'py>, o: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("viewer", o.viewer)?;
    d.set_item("config", config_dict(py, &o.config)?)?;
    d.set_item("phase", phase_name(o.phase))?;
    d.set_item("current", o.current)?;
    d.set_item("actor", o.actor)?;
    d.set_item("turn", o.turn)?;
    d.set_item("board", board_dict(py, &o.board)?)?;
    d.set_item("robber", o.robber)?;
    d.set_item("bank", counts(&o.bank))?;
    d.set_item("dev_deck_remaining", o.dev_deck_remaining)?;
    d.set_item("my_hand", counts(&o.my_hand))?;
    d.set_item("my_dev_cards", counts(&o.my_dev_cards))?;
    d.set_item("my_new_dev_cards", counts(&o.my_new_dev_cards))?;
    d.set_item("hand_counts", counts(&o.hand_counts))?;
    d.set_item("dev_card_counts", counts(&o.dev_card_counts))?;
    d.set_item("new_dev_card_counts", counts(&o.new_dev_card_counts))?;
    let played: Vec<Vec<u32>> = o.dev_cards_played.iter().map(|p| counts(p)).collect();
    d.set_item("dev_cards_played", played)?;
    let settlements: Vec<Vec<u32>> = o.settlements.iter().map(|&m| node_list(m)).collect();
    d.set_item("settlements", settlements)?;
    let cities: Vec<Vec<u32>> = o.cities.iter().map(|&m| node_list(m)).collect();
    d.set_item("cities", cities)?;
    let roads: Vec<Vec<u32>> = o.roads.iter().map(|&m| edge_list(m)).collect();
    d.set_item("roads", roads)?;
    d.set_item("knights_played", counts(&o.knights_played))?;
    d.set_item("public_vp", counts(&o.public_vp))?;
    d.set_item("longest_road_owner", o.longest_road_owner)?;
    d.set_item("largest_army_owner", o.largest_army_owner)?;
    match &o.trade {
        Some(t) => d.set_item("trade", trade_dict(py, t)?)?,
        None => d.set_item("trade", py.None())?,
    }
    d.set_item("dev_played_this_turn", o.dev_played_this_turn)?;
    d.set_item("offers_this_turn", o.offers_this_turn)?;
    d.set_item("discard_remaining", counts(&o.discard_remaining))?;
    d.set_item("return_phase", phase_name(o.return_phase))?;
    Ok(d)
}

pub fn event_dict<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match *e {
        Event::BuiltSettlement { player, node } => {
            d.set_item("type", "built_settlement")?;
            d.set_item("player", player)?;
            d.set_item("node", node)?;
        }
        Event::BuiltCity { player, node } => {
            d.set_item("type", "built_city")?;
            d.set_item("player", player)?;
            d.set_item("node", node)?;
        }
        Event::BuiltRoad { player, edge } => {
            d.set_item("type", "built_road")?;
            d.set_item("player", player)?;
            d.set_item("edge", edge)?;
        }
        Event::Rolled { player, dice } => {
            d.set_item("type", "rolled")?;
            d.set_item("player", player)?;
            d.set_item("dice", vec![dice.0 as u32, dice.1 as u32])?;
        }
        Event::Produced { player, resources } => {
            d.set_item("type", "produced")?;
            d.set_item("player", player)?;
            d.set_item("resources", counts(&resources))?;
        }
        Event::Discarded { player, resource } => {
            d.set_item("type", "discarded")?;
            d.set_item("player", player)?;
            d.set_item("resource", resource_name(resource))?;
        }
        Event::RobberMoved { player, tile } => {
            d.set_item("type", "robber_moved")?;
            d.set_item("player", player)?;
            d.set_item("tile", tile)?;
        }
        Event::Stole {
            thief,
            victim,
            resource,
        } => {
            d.set_item("type", "stole")?;
            d.set_item("thief", thief)?;
            d.set_item("victim", victim)?;
            d.set_item("resource", resource.map(resource_name))?;
        }
        Event::BoughtDev { player, card } => {
            d.set_item("type", "bought_dev")?;
            d.set_item("player", player)?;
            d.set_item("card", card.map(dev_name))?;
        }
        Event::PlayedDev { player, card } => {
            d.set_item("type", "played_dev")?;
            d.set_item("player", player)?;
            d.set_item("card", dev_name(card))?;
        }
        Event::MonopolyTaken {
            player,
            resource,
            from,
        } => {
            d.set_item("type", "monopoly_taken")?;
            d.set_item("player", player)?;
            d.set_item("resource", resource_name(resource))?;
            d.set_item("from", counts(&from))?;
        }
        Event::YearOfPlentyTaken { player, resources } => {
            d.set_item("type", "year_of_plenty_taken")?;
            d.set_item("player", player)?;
            d.set_item("resources", counts(&resources))?;
        }
        Event::MaritimeTraded { player, gave, got } => {
            d.set_item("type", "maritime_traded")?;
            d.set_item("player", player)?;
            d.set_item("gave", counts(&gave))?;
            d.set_item("got", counts(&got))?;
        }
        Event::TradeOffered { player, give, get } => {
            d.set_item("type", "trade_offered")?;
            d.set_item("player", player)?;
            d.set_item("give", counts(&give))?;
            d.set_item("get", counts(&get))?;
        }
        Event::TradeResponded { player, accepted } => {
            d.set_item("type", "trade_responded")?;
            d.set_item("player", player)?;
            d.set_item("accepted", accepted)?;
        }
        Event::TradeConfirmed {
            offerer,
            partner,
            offerer_gave,
            partner_gave,
        } => {
            d.set_item("type", "trade_confirmed")?;
            d.set_item("offerer", offerer)?;
            d.set_item("partner", partner)?;
            d.set_item("offerer_gave", counts(&offerer_gave))?;
            d.set_item("partner_gave", counts(&partner_gave))?;
        }
        Event::TradeCancelled { player } => {
            d.set_item("type", "trade_cancelled")?;
            d.set_item("player", player)?;
        }
        Event::TurnEnded { player } => {
            d.set_item("type", "turn_ended")?;
            d.set_item("player", player)?;
        }
        Event::GameOver { winner } => {
            d.set_item("type", "game_over")?;
            d.set_item("winner", winner)?;
        }
    }
    Ok(d)
}

const SNAPSHOT_KEYS: [&str; 17] = [
    "board",
    "robber",
    "bank",
    "dev_deck",
    "players",
    "current",
    "phase",
    "setup_road_node",
    "roads_left",
    "winner",
    "return_phase",
    "turn",
    "setup_step",
    "dev_played_this_turn",
    "offers_this_turn",
    "longest_road_owner",
    "largest_army_owner",
];
const PLAYER_KEYS: [&str; 9] = [
    "hand",
    "dev_hand",
    "dev_new",
    "dev_played",
    "knights_played",
    "settlements",
    "cities",
    "roads",
    "discard_remaining",
];
const BOARD_KEYS: [&str; 3] = ["tile_resource", "tile_number", "ports"];

/// Rejects the first key of `d` not in `allowed`.
fn check_keys(d: &Bound<'_, PyDict>, allowed: &[&str], what: &str) -> PyResult<()> {
    for k in d.keys() {
        let key: String = k.extract()?;
        if !allowed.contains(&key.as_str()) {
            return Err(value_error(format!("unknown {what} key {key:?}")));
        }
    }
    Ok(())
}

/// `d[key]`, or a ValueError naming the missing key.
fn item<'py>(d: &Bound<'py, PyDict>, key: &str, what: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| value_error(format!("{what} is missing {key:?}")))
}

/// `None` as `None`; anything else through `parse`.
fn opt<'py, T>(
    v: &Bound<'py, PyAny>,
    parse: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    if v.is_none() {
        Ok(None)
    } else {
        parse(v).map(Some)
    }
}

fn opt_player(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Option<PlayerId>> {
    opt(v, |v| Ok(index_arg(v, NUM_PLAYERS, what)? as PlayerId))
}

/// A list of node ids in `0..end` as a bitmask; each id at most once.
fn id_mask64(v: &Bound<'_, PyAny>, end: usize, what: &str) -> PyResult<u64> {
    let mut mask = 0u64;
    for x in v.extract::<Vec<Bound<'_, PyAny>>>()? {
        let n = index_arg(&x, end, what)?;
        if mask & (1u64 << n) != 0 {
            return Err(value_error(format!("{what} lists node {n} twice")));
        }
        mask |= 1u64 << n;
    }
    Ok(mask)
}

/// A list of edge ids in `0..end` as a bitmask; each id at most once.
fn id_mask128(v: &Bound<'_, PyAny>, end: usize, what: &str) -> PyResult<u128> {
    let mut mask = 0u128;
    for x in v.extract::<Vec<Bound<'_, PyAny>>>()? {
        let e = index_arg(&x, end, what)?;
        if mask & (1u128 << e) != 0 {
            return Err(value_error(format!("{what} lists edge {e} twice")));
        }
        mask |= 1u128 << e;
    }
    Ok(mask)
}

fn small_array(v: &Bound<'_, PyAny>, what: &str) -> PyResult<[u8; 5]> {
    let mut out = [0u8; 5];
    for (o, x) in out.iter_mut().zip(seq_arg(v, 5, what)?) {
        *o = int_arg::<u8>(&x, what)?;
    }
    Ok(out)
}

fn parse_board(v: &Bound<'_, PyAny>) -> PyResult<Board> {
    let d = v.cast::<PyDict>()?;
    check_keys(d, &BOARD_KEYS, "board")?;
    let res = seq_arg(
        &item(d, "tile_resource", "board")?,
        NUM_TILES,
        "tile_resource",
    )?;
    let mut tile_resource = [None; NUM_TILES];
    for (t, r) in res.iter().enumerate() {
        tile_resource[t] = if r.is_none() {
            None
        } else {
            Some(parse_resource(&r.extract::<String>()?)?)
        };
    }
    let nums = seq_arg(&item(d, "tile_number", "board")?, NUM_TILES, "tile_number")?;
    let mut tile_number = [0u8; NUM_TILES];
    for (t, n) in nums.iter().enumerate() {
        tile_number[t] = int_arg::<u8>(n, "tile_number")?;
    }
    let ports_in = seq_arg(&item(d, "ports", "board")?, 9, "ports")?;
    let mut ports = [(0u8, PortKind::Generic); 9];
    for (i, p) in ports_in.iter().enumerate() {
        let pair = seq_arg(p, 2, "port")?;
        let edge = index_arg(&pair[0], NUM_EDGES, "port edge")? as u8;
        let kind: String = pair[1].extract()?;
        ports[i] = (
            edge,
            if kind == "generic" {
                PortKind::Generic
            } else {
                PortKind::Specific(parse_resource(&kind)?)
            },
        );
    }
    Ok(Board::new(tile_resource, tile_number, ports))
}

/// The phase, from `phase` plus the three phase-specific keys. Each of those must be `None`
/// unless its phase uses it; `setup_road_node` and `roads_left` are required in their phase,
/// while `winner` may be `None` in `game_over` (no winner within `max_turns`).
fn parse_phase(d: &Bound<'_, PyDict>) -> PyResult<Phase> {
    const S: &str = "snapshot";
    let name: String = item(d, "phase", S)?.extract()?;
    let node = opt(&item(d, "setup_road_node", S)?, |v| {
        Ok(index_arg(v, NUM_NODES, "setup_road_node")? as u8)
    })?;
    let roads_left = opt(&item(d, "roads_left", S)?, |v| {
        int_arg::<u8>(v, "roads_left")
    })?;
    let winner = opt_player(&item(d, "winner", S)?, "winner")?;
    let stray = |key: &str, used_by: &str| {
        Err(value_error(format!(
            "{key} must be None unless phase is {used_by}"
        )))
    };
    if node.is_some() && name != "setup_road" {
        return stray("setup_road_node", "setup_road");
    }
    if roads_left.is_some() && name != "road_building" {
        return stray("roads_left", "road_building");
    }
    if winner.is_some() && name != "game_over" {
        return stray("winner", "game_over");
    }
    Ok(match name.as_str() {
        "setup_settlement" => Phase::SetupSettlement,
        "setup_road" => Phase::SetupRoad {
            node: node.ok_or_else(|| {
                value_error("setup_road_node is required when phase is setup_road")
            })?,
        },
        "pre_roll" => Phase::PreRoll,
        "discard" => Phase::Discard,
        "move_robber" => Phase::MoveRobber,
        "steal" => Phase::Steal,
        "main" => Phase::Main,
        "road_building" => Phase::RoadBuilding {
            roads_left: roads_left
                .ok_or_else(|| value_error("roads_left is required when phase is road_building"))?,
        },
        "game_over" => Phase::GameOver { winner },
        "trade_response" | "trade_confirm" => {
            return Err(value_error(
                "snapshots with a pending domestic trade are not supported",
            ))
        }
        other => return Err(value_error(format!("unknown phase {other:?}"))),
    })
}

fn parse_player(v: &Bound<'_, PyAny>, p: usize) -> PyResult<PlayerSnapshot> {
    let what = format!("player {p}");
    let d = v.cast::<PyDict>()?;
    check_keys(d, &PLAYER_KEYS, &what)?;
    let field = |key: &str| item(d, key, &what);
    Ok(PlayerSnapshot {
        hand: small_array(&field("hand")?, &format!("{what} hand"))?,
        dev_hand: small_array(&field("dev_hand")?, &format!("{what} dev_hand"))?,
        dev_new: small_array(&field("dev_new")?, &format!("{what} dev_new"))?,
        dev_played: small_array(&field("dev_played")?, &format!("{what} dev_played"))?,
        knights_played: int_arg::<u8>(
            &field("knights_played")?,
            &format!("{what} knights_played"),
        )?,
        settlements: id_mask64(
            &field("settlements")?,
            NUM_NODES,
            &format!("{what} settlements (node ids)"),
        )?,
        cities: id_mask64(
            &field("cities")?,
            NUM_NODES,
            &format!("{what} cities (node ids)"),
        )?,
        roads: id_mask128(
            &field("roads")?,
            NUM_EDGES,
            &format!("{what} roads (edge ids)"),
        )?,
        discard_remaining: int_arg::<u8>(
            &field("discard_remaining")?,
            &format!("{what} discard_remaining"),
        )?,
    })
}

/// A snapshot dict (the exact schema `snapshot_dict` produces) as an engine `Snapshot`.
pub fn parse_snapshot(d: &Bound<'_, PyDict>) -> PyResult<Snapshot> {
    const S: &str = "snapshot";
    check_keys(d, &SNAPSHOT_KEYS, S)?;
    let board = parse_board(&item(d, "board", S)?)?;
    let robber = index_arg(&item(d, "robber", S)?, NUM_TILES, "robber")? as u8;
    let bank = small_array(&item(d, "bank", S)?, "bank")?;
    let mut dev_deck = Vec::new();
    for c in item(d, "dev_deck", S)?.extract::<Vec<Bound<'_, PyAny>>>()? {
        dev_deck.push(parse_dev(&c.extract::<String>()?)?);
    }
    let rows = seq_arg(&item(d, "players", S)?, NUM_PLAYERS, "players")?;
    let mut players = [PlayerSnapshot::default(); NUM_PLAYERS];
    for (p, row) in rows.iter().enumerate() {
        players[p] = parse_player(row, p)?;
    }
    let current = index_arg(&item(d, "current", S)?, NUM_PLAYERS, "current")? as PlayerId;
    let phase = parse_phase(d)?;
    let return_phase = match item(d, "return_phase", S)?.extract::<String>()?.as_str() {
        "pre_roll" => Phase::PreRoll,
        "main" => Phase::Main,
        _ => return Err(value_error("return_phase must be 'pre_roll' or 'main'")),
    };
    Ok(Snapshot {
        board,
        robber,
        bank,
        dev_deck,
        players,
        current,
        phase,
        return_phase,
        turn: int_arg::<u32>(&item(d, "turn", S)?, "turn")?,
        setup_step: int_arg::<u8>(&item(d, "setup_step", S)?, "setup_step")?,
        dev_played_this_turn: item(d, "dev_played_this_turn", S)?.extract::<bool>()?,
        offers_this_turn: int_arg::<u8>(&item(d, "offers_this_turn", S)?, "offers_this_turn")?,
        longest_road_owner: opt_player(&item(d, "longest_road_owner", S)?, "longest_road_owner")?,
        largest_army_owner: opt_player(&item(d, "largest_army_owner", S)?, "largest_army_owner")?,
    })
}

pub fn snapshot_dict<'py>(py: Python<'py>, s: &Snapshot) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("board", board_dict(py, &s.board)?)?;
    d.set_item("robber", s.robber)?;
    d.set_item("bank", counts(&s.bank))?;
    let deck: Vec<&str> = s.dev_deck.iter().map(|&c| dev_name(c)).collect();
    d.set_item("dev_deck", deck)?;
    let players = PyList::empty(py);
    for p in &s.players {
        let pd = PyDict::new(py);
        pd.set_item("hand", counts(&p.hand))?;
        pd.set_item("dev_hand", counts(&p.dev_hand))?;
        pd.set_item("dev_new", counts(&p.dev_new))?;
        pd.set_item("dev_played", counts(&p.dev_played))?;
        pd.set_item("knights_played", p.knights_played)?;
        pd.set_item("settlements", node_list(p.settlements))?;
        pd.set_item("cities", node_list(p.cities))?;
        pd.set_item("roads", edge_list(p.roads))?;
        pd.set_item("discard_remaining", p.discard_remaining)?;
        players.append(pd)?;
    }
    d.set_item("players", players)?;
    d.set_item("current", s.current)?;
    d.set_item("phase", phase_name(s.phase))?;
    d.set_item(
        "setup_road_node",
        match s.phase {
            Phase::SetupRoad { node } => Some(node),
            _ => None,
        },
    )?;
    d.set_item(
        "roads_left",
        match s.phase {
            Phase::RoadBuilding { roads_left } => Some(roads_left),
            _ => None,
        },
    )?;
    d.set_item(
        "winner",
        match s.phase {
            Phase::GameOver { winner } => winner,
            _ => None,
        },
    )?;
    d.set_item("return_phase", phase_name(s.return_phase))?;
    d.set_item("turn", s.turn)?;
    d.set_item("setup_step", s.setup_step)?;
    d.set_item("dev_played_this_turn", s.dev_played_this_turn)?;
    d.set_item("offers_this_turn", s.offers_this_turn)?;
    d.set_item("longest_road_owner", s.longest_road_owner)?;
    d.set_item("largest_army_owner", s.largest_army_owner)?;
    Ok(d)
}

/// An event dict in the shape `event_dict` writes, e.g. from another engine's translated log.
pub fn parse_event(v: &Bound<'_, PyAny>) -> PyResult<Event> {
    const E: &str = "event";
    let d = v.cast::<PyDict>()?;
    let get = |k: &str| item(d, k, E);
    let player = |k: &str| -> PyResult<PlayerId> { Ok(index_arg(&get(k)?, NUM_PLAYERS, k)? as PlayerId) };
    let hand = |k: &str| -> PyResult<Hand> { small_array(&get(k)?, k) };
    let resource = |k: &str| -> PyResult<Resource> { parse_resource(&get(k)?.extract::<String>()?) };
    let kind: String = get("type")?.extract()?;
    Ok(match kind.as_str() {
        "built_settlement" => Event::BuiltSettlement { player: player("player")?, node: index_arg(&get("node")?, NUM_NODES, "node")? as u8 },
        "built_city" => Event::BuiltCity { player: player("player")?, node: index_arg(&get("node")?, NUM_NODES, "node")? as u8 },
        "built_road" => Event::BuiltRoad { player: player("player")?, edge: index_arg(&get("edge")?, NUM_EDGES, "edge")? as u8 },
        "rolled" => {
            let dice = small_dice(&get("dice")?)?;
            Event::Rolled { player: player("player")?, dice }
        }
        "produced" => Event::Produced { player: player("player")?, resources: hand("resources")? },
        "discarded" => Event::Discarded { player: player("player")?, resource: resource("resource")? },
        "robber_moved" => Event::RobberMoved { player: player("player")?, tile: index_arg(&get("tile")?, NUM_TILES, "tile")? as u8 },
        "stole" => Event::Stole {
            thief: player("thief")?,
            victim: player("victim")?,
            resource: opt(&get("resource")?, |r| parse_resource(&r.extract::<String>()?))?,
        },
        "bought_dev" => Event::BoughtDev {
            player: player("player")?,
            card: opt(&get("card")?, |c| parse_dev(&c.extract::<String>()?))?,
        },
        "played_dev" => Event::PlayedDev { player: player("player")?, card: parse_dev(&get("card")?.extract::<String>()?)? },
        "monopoly_taken" => {
            let xs = seq_arg(&get("from")?, NUM_PLAYERS, "from")?;
            let mut from = [0u8; NUM_PLAYERS];
            for (f, x) in from.iter_mut().zip(&xs) {
                *f = int_arg::<u8>(x, "from")?;
            }
            Event::MonopolyTaken { player: player("player")?, resource: resource("resource")?, from }
        }
        "year_of_plenty_taken" => Event::YearOfPlentyTaken { player: player("player")?, resources: hand("resources")? },
        "maritime_traded" => Event::MaritimeTraded { player: player("player")?, gave: hand("gave")?, got: hand("got")? },
        "trade_offered" => Event::TradeOffered { player: player("player")?, give: hand("give")?, get: hand("get")? },
        "trade_responded" => Event::TradeResponded { player: player("player")?, accepted: get("accepted")?.extract::<bool>()? },
        "trade_confirmed" => Event::TradeConfirmed {
            offerer: player("offerer")?,
            partner: player("partner")?,
            offerer_gave: hand("offerer_gave")?,
            partner_gave: hand("partner_gave")?,
        },
        "trade_cancelled" => Event::TradeCancelled { player: player("player")? },
        "turn_ended" => Event::TurnEnded { player: player("player")? },
        "game_over" => Event::GameOver { winner: opt_player(&get("winner")?, "winner")? },
        other => return Err(value_error(format!("unknown event type {other:?}"))),
    })
}

fn small_dice(v: &Bound<'_, PyAny>) -> PyResult<(u8, u8)> {
    let xs = seq_arg(v, 2, "dice")?;
    let die = |x: &Bound<'_, PyAny>| -> PyResult<u8> {
        let d = int_arg::<u8>(x, "dice")?;
        if (1..=6).contains(&d) { Ok(d) } else { Err(value_error(format!("a die shows 1-6, got {d}"))) }
    };
    Ok((die(&xs[0])?, die(&xs[1])?))
}
