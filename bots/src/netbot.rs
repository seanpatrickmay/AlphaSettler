//! Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and
//! `net:NAME@N+rD`, each optionally with `+sK`, `+t`, `+g`, `+c` and `+o` (an opening model picks
//! the setup settlements, spec 3d-1 Section 4.5) in any order (`+g` and `+c` are split off by
//! `ismcts::split_search_flags` before `parse_net_name` reads the rest), with the same limits on
//! N, D and K as `ismcts`: IsmctsBot searching with a `NetEvaluator`. NAME is `[A-Za-z0-9_-]{1,64}`
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

/// A parsed `net:` bot name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetName<'a> {
    pub net: &'a str,
    pub simulations: u32,
    pub rollout: u32,
    /// `+sK`: setup settlements search K times the simulations (1 without it).
    pub setup_factor: u32,
    /// `+t`: the bot may offer.
    pub offers: bool,
    /// `+o`: `<nets_dir>/opening.bin` picks the setup settlements.
    pub opening: bool,
}

/// `net:NAME` -> 1,000 simulations, no rollout; `net:NAME@N` -> N simulations; `net:NAME@N+rD`
/// -> D rollout moves. `+sK`, `+t` and `+o` set the other fields; the suffixes come in any order
/// (`ismcts::split_suffixes`). A name with `+g` or `+c` gives `None`: callers split them off first
/// with `ismcts::split_search_flags`, as `Roster` does.
pub fn parse_net_name(name: &str) -> Option<NetName<'_>> {
    let (canon, flags) = ismcts::split_suffixes(name)?;
    let (canon, setup_factor) = ismcts::split_setup(&canon)?;
    let rest = name.strip_prefix("net:")?;
    let end = rest.find(['@', '+']).unwrap_or(rest.len());
    let net = &rest[..end];
    if !valid_net_name(net) {
        return None;
    }
    let tail = &canon["net:".len() + net.len()..];
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{tail}"))?;
    Some(NetName {
        net,
        simulations,
        rollout,
        setup_factor,
        offers: flags.offers,
        opening: flags.opening,
    })
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
