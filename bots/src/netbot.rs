//! Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and
//! `net:NAME@N+rD`, each optionally with `+sK`, `+t` and `+g` in any order (`+g` is split off by
//! `ismcts::split_gumbel` before `parse_net_name` reads the rest), with the same limits on N,
//! D and K as `ismcts`: IsmctsBot searching with a `NetEvaluator`. NAME is `[A-Za-z0-9_-]{1,64}`
//! and names the file `<nets-dir>/NAME.bin`; the character set keeps every name a plain file name
//! inside the directory.

use crate::greedy;
use crate::ismcts::{self, IsmctsBot};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::Weights;
use settler_search::SearchConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DEFAULT_NETS_DIR: &str = "runs/3b/nets";
pub const MAX_NET_NAME: usize = 64;
/// Leaves per network call. Set from the batch sweep in `docs/perf/net-search.md`.
pub const NET_BATCH: usize = 8;

pub fn valid_net_name(net: &str) -> bool {
    (1..=MAX_NET_NAME).contains(&net.len())
        && net
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `net:NAME` -> (NAME, 1000, 0, 1, false); `net:NAME@N` -> (NAME, N, 0, 1, false);
/// `net:NAME@N+rD` -> (NAME, N, D, 1, false). `+sK` sets the setup factor to K and `+t` the last
/// field (the bot may offer); the suffixes come in any order (`ismcts::split_trades`).
pub fn parse_net_name(name: &str) -> Option<(&str, u32, u32, u32, bool)> {
    let (canon, offers) = ismcts::split_trades(name)?;
    let (canon, factor) = ismcts::split_setup(&canon)?;
    let rest = name.strip_prefix("net:")?;
    let end = rest.find(['@', '+']).unwrap_or(rest.len());
    let net = &rest[..end];
    if !valid_net_name(net) {
        return None;
    }
    let tail = &canon["net:".len() + net.len()..];
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{tail}"))?;
    Some((net, simulations, rollout, factor, offers))
}

pub fn net_path(nets_dir: &Path, net: &str) -> PathBuf {
    nets_dir.join(format!("{net}.bin"))
}

/// Load and validate `<nets_dir>/<net>.bin`. `net` must be a valid NAME (checked first, so a bad
/// name never reaches the filesystem).
pub fn load_net(nets_dir: &Path, net: &str) -> Result<Arc<Weights>, String> {
    if !valid_net_name(net) {
        return Err(format!(
            "bad net name {net:?}: use 1 to {MAX_NET_NAME} of A-Z a-z 0-9 _ -"
        ));
    }
    Weights::load(&net_path(nets_dir, net))
        .map(Arc::new)
        .map_err(|e| format!("net {net:?}: {e}"))
}

/// The bot a `net:` name builds: `net_bot_with` at `NET_BATCH` leaves per network call.
pub fn net_bot(
    seed: u64,
    weights: Arc<Weights>,
    simulations: u32,
    rollout: u32,
) -> IsmctsBot<NetEvaluator> {
    net_bot_with(seed, weights, simulations, rollout, NET_BATCH)
}

/// A net bot (name `"net"`) searching `batch` (positive) leaves per network call; `rollout` > 0
/// adds that many greedy moves per leaf before the value is read.
pub fn net_bot_with(
    seed: u64,
    weights: Arc<Weights>,
    simulations: u32,
    rollout: u32,
    batch: usize,
) -> IsmctsBot<NetEvaluator> {
    let eval = if rollout == 0 {
        NetEvaluator::new(weights)
    } else {
        NetEvaluator::with_rollout(weights, rollout, greedy::choose)
    };
    let cfg = SearchConfig {
        simulations,
        batch,
        ..SearchConfig::default()
    };
    IsmctsBot::with_evaluator("net", seed, cfg, eval)
}
