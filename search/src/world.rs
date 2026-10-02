//! Sampled worlds: complete states consistent with everything one player has seen.
//!
//! Built once per decision: a template `State` (validated by `State::from_snapshot`) carries
//! everything public plus the viewer's own cards. Each sample copies it and overwrites only what
//! the viewer cannot see: opponents' hands (drawn from the belief), their dev cards and the deck
//! order (a uniform deal of the unseen cards, conditioned on no opponent holding enough VP cards
//! to have won already), and the random seed.

use crate::belief::Belief;
use settler_engine::rng::Rng;
use settler_engine::types::*;
use settler_engine::{Observation, Phase, PlayerSnapshot, Snapshot, State};

pub struct WorldSampler {
    template: State,
    viewer: PlayerId,
    /// Dev cards the viewer cannot place (in opponents' hands or the deck), in kind order.
    unseen: [DevCard; DEV_DECK_SIZE],
    unseen_len: usize,
    /// Per player: dev cards held (0 for the viewer, whose cards are known) and how many are new.
    held: [u8; NUM_PLAYERS],
    new: [u8; NUM_PLAYERS],
    /// When some opponent must be kept below the win: every joint count of VP cards among the
    /// players' hidden dev cards (indexed by player, 0 for the viewer) that the deal allows, with
    /// its cumulative weight. `None` deals uniformly.
    vp_cells: Option<Vec<([u8; NUM_PLAYERS], f64)>>,
}

struct Deal {
    dev_hand: [[u8; 5]; NUM_PLAYERS],
    dev_new: [[u8; 5]; NUM_PLAYERS],
    deck: [DevCard; DEV_DECK_SIZE],
    deck_len: usize,
}

/// Setup placements done so far: each setup step ends with a road, so during setup it is the
/// number of roads on the board; afterwards it is 8.
fn setup_step(obs: &Observation) -> u8 {
    match obs.phase {
        Phase::SetupSettlement | Phase::SetupRoad { .. } => {
            obs.roads.iter().map(|r| r.count_ones()).sum::<u32>() as u8
        }
        _ => 8,
    }
}

/// Binomial coefficient as a float (0 when `r > n`).
fn choose(n: usize, r: usize) -> f64 {
    if r > n {
        return 0.0;
    }
    (0..r).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The joint counts of VP cards in each player's `held` hidden dev cards under a uniform deal of
/// `vp` VP cards among those cards and `deck` deck places (multivariate hypergeometric: weight
/// `prod_q C(held_q, k_q) * C(deck, vp - sum k)`), keeping only counts within `caps`. Returns the
/// cells of positive weight with cumulative weights; empty when none has any.
fn vp_cells(
    held: &[u8; NUM_PLAYERS],
    vp: usize,
    deck: usize,
    caps: &[usize; NUM_PLAYERS],
) -> Vec<([u8; NUM_PLAYERS], f64)> {
    let top: [usize; NUM_PLAYERS] =
        std::array::from_fn(|q| (held[q] as usize).min(vp).min(caps[q]));
    let mut cells = Vec::new();
    let mut total = 0.0;
    let mut k = [0usize; NUM_PLAYERS];
    loop {
        let sum: usize = k.iter().sum();
        if sum <= vp {
            let w = (0..NUM_PLAYERS)
                .map(|q| choose(held[q] as usize, k[q]))
                .product::<f64>()
                * choose(deck, vp - sum);
            if w > 0.0 {
                total += w;
                cells.push((std::array::from_fn(|q| k[q] as u8), total));
            }
        }
        // Odometer: bump the lowest count still below its top and reset the ones before it.
        let Some(q) = (0..NUM_PLAYERS).find(|&q| k[q] < top[q]) else {
            break;
        };
        k[q] += 1;
        k[..q].fill(0);
    }
    cells
}

impl WorldSampler {
    pub fn new(obs: &Observation, belief: &Belief) -> Result<WorldSampler, String> {
        belief.check(obs)?;
        let hands = belief
            .states()
            .first()
            .ok_or("the belief has no states")?
            .0
            .hands;
        if matches!(
            obs.phase,
            Phase::TradeResponse | Phase::TradeConfirm | Phase::GameOver { .. }
        ) {
            return Err(format!("no worlds to sample in phase {:?}", obs.phase));
        }
        let me = obs.viewer as usize;
        let mut unseen_counts = DEV_DECK_COUNTS;
        for k in 0..5 {
            let played: u8 = (0..NUM_PLAYERS).map(|p| obs.dev_cards_played[p][k]).sum();
            unseen_counts[k] = unseen_counts[k]
                .checked_sub(played + obs.my_dev_cards[k])
                .ok_or_else(|| {
                    format!(
                        "more {:?} cards seen than the deck has",
                        DevCard::from_index(k)
                    )
                })?;
        }
        let held: [u8; NUM_PLAYERS] =
            std::array::from_fn(|p| if p == me { 0 } else { obs.dev_card_counts[p] });
        let new: [u8; NUM_PLAYERS] = std::array::from_fn(|p| {
            if p == me {
                0
            } else {
                obs.new_dev_card_counts[p]
            }
        });
        let wanted = held.iter().map(|&c| c as u32).sum::<u32>() + obs.dev_deck_remaining as u32;
        let have: u32 = unseen_counts.iter().map(|&c| c as u32).sum();
        if wanted != have {
            return Err(format!(
                "{have} unseen dev cards for {wanted} hidden places"
            ));
        }
        let mut unseen = [DevCard::Knight; DEV_DECK_SIZE];
        let mut unseen_len = 0;
        for (k, &n) in unseen_counts.iter().enumerate() {
            for _ in 0..n {
                unseen[unseen_len] = DevCard::from_index(k);
                unseen_len += 1;
            }
        }
        if obs.current != obs.viewer {
            let public = obs.public_vp[obs.current as usize];
            if public >= obs.config.vp_to_win {
                return Err(format!(
                    "player {} has {public} public VP with the game still on",
                    obs.current
                ));
            }
        }
        // Nobody but the player on turn can win off their own turn, and hidden VP cards change
        // only on their owner's turn, so no opponent holds enough VP cards to reach the target:
        // each is capped one short of it. An opponent whose public VP already reaches the target
        // (a longest road gained off turn) is left uncapped. The one case this gets wrong is an
        // off-turn longest-road transfer that lifts public VP plus VP cards to the target; such a
        // player is never dealt their winning holding.
        let vp = unseen_counts[DevCard::VictoryPoint.index()] as usize;
        let deck = obs.dev_deck_remaining as usize;
        let caps: [usize; NUM_PLAYERS] = std::array::from_fn(|q| {
            let public = obs.public_vp[q];
            if q == me || public >= obs.config.vp_to_win {
                usize::MAX
            } else {
                (obs.config.vp_to_win - 1 - public) as usize
            }
        });
        let on_turn_caps: [usize; NUM_PLAYERS] = std::array::from_fn(|q| {
            if q == obs.current as usize {
                caps[q]
            } else {
                usize::MAX
            }
        });
        let binds = |caps: &[usize; NUM_PLAYERS]| {
            (0..NUM_PLAYERS).any(|q| (held[q] as usize).min(vp) > caps[q])
        };
        // When no deal keeps every opponent below the target (only possible after such a
        // transfer), drop the off-turn caps and keep the on-turn one, which the engine requires;
        // if even that has no deal, deal uniformly. So this never fails.
        let vp_cells = [caps, on_turn_caps]
            .iter()
            .filter(|c| binds(c))
            .map(|c| vp_cells(&held, vp, deck, c))
            .find(|cells| !cells.is_empty());
        let mut sampler = WorldSampler {
            template: State::new(0, obs.config),
            viewer: obs.viewer,
            unseen,
            unseen_len,
            held,
            new,
            vp_cells,
        };
        // The template's hidden parts come from one sample; `sample` overwrites them each time.
        let mut rng = Rng::new(0x7E4D);
        let deal = sampler.deal(&mut rng);
        let snap = Snapshot {
            board: obs.board,
            robber: obs.robber,
            bank: obs.bank,
            dev_deck: deal.deck[..deal.deck_len].to_vec(),
            players: std::array::from_fn(|p| PlayerSnapshot {
                hand: hands[p],
                dev_hand: if p == me {
                    obs.my_dev_cards
                } else {
                    deal.dev_hand[p]
                },
                dev_new: if p == me {
                    obs.my_new_dev_cards
                } else {
                    deal.dev_new[p]
                },
                dev_played: obs.dev_cards_played[p],
                knights_played: obs.knights_played[p],
                settlements: obs.settlements[p],
                cities: obs.cities[p],
                roads: obs.roads[p],
                discard_remaining: obs.discard_remaining[p],
            }),
            current: obs.current,
            phase: obs.phase,
            return_phase: obs.return_phase,
            turn: obs.turn,
            setup_step: setup_step(obs),
            dev_played_this_turn: obs.dev_played_this_turn,
            offers_this_turn: obs.offers_this_turn,
            longest_road_owner: obs.longest_road_owner,
            largest_army_owner: obs.largest_army_owner,
        };
        sampler.template = State::from_snapshot(0, obs.config, &snap)?;
        Ok(sampler)
    }

    /// A uniform deal of the unseen dev cards to opponents (new cards first) and the deck,
    /// conditioned on every capped opponent holding at most their cap of VP cards (see `new`).
    /// Drawn exactly: first the joint VP-card counts (truncated multivariate hypergeometric), then
    /// each opponent's VP and other cards from those parts of the pool, then the rest as the deck.
    fn deal(&self, rng: &mut Rng) -> Deal {
        let mut pool = self.unseen;
        let pool = &mut pool[..self.unseen_len];
        rng.shuffle(pool);
        let mut d = Deal {
            dev_hand: [[0; 5]; NUM_PLAYERS],
            dev_new: [[0; 5]; NUM_PLAYERS],
            deck: [DevCard::Knight; DEV_DECK_SIZE],
            deck_len: 0,
        };
        let Some(cells) = &self.vp_cells else {
            let mut next = 0;
            for q in 0..NUM_PLAYERS {
                for i in 0..self.held[q] {
                    let c = pool[next].index();
                    next += 1;
                    d.dev_hand[q][c] += 1;
                    if i < self.new[q] {
                        d.dev_new[q][c] += 1;
                    }
                }
            }
            d.deck_len = pool.len() - next;
            d.deck[..d.deck_len].copy_from_slice(&pool[next..]);
            return d;
        };
        let total = cells.last().expect("`new` keeps only non-empty cells").1;
        let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
        let i = cells
            .partition_point(|&(_, cum)| cum <= u)
            .min(cells.len() - 1);
        let k = cells[i].0;
        // The pool is in random order, so taking VP and other cards in pool order, each player in
        // turn up to their counts, is a uniform choice of each player's cards of each part.
        let mut vp_left: [u8; NUM_PLAYERS] = k;
        let mut other_left: [u8; NUM_PLAYERS] = std::array::from_fn(|q| self.held[q] - k[q]);
        let mut hands = [[DevCard::Knight; DEV_DECK_SIZE]; NUM_PLAYERS];
        let mut taken = [0usize; NUM_PLAYERS];
        let mut kept = 0;
        for &c in pool.iter() {
            let left = if c == DevCard::VictoryPoint {
                &mut vp_left
            } else {
                &mut other_left
            };
            match left.iter().position(|&n| n > 0) {
                Some(q) => {
                    left[q] -= 1;
                    hands[q][taken[q]] = c;
                    taken[q] += 1;
                }
                None => {
                    d.deck[kept] = c;
                    kept += 1;
                }
            }
        }
        for q in 0..NUM_PLAYERS {
            let mine = &mut hands[q][..taken[q]];
            rng.shuffle(mine);
            for (i, c) in mine.iter().enumerate() {
                d.dev_hand[q][c.index()] += 1;
                if i < self.new[q] as usize {
                    d.dev_new[q][c.index()] += 1;
                }
            }
        }
        // Taking the first cards of each kind skews the order of what is left (it tends to start
        // with whichever kind lost fewer cards), so shuffle it afresh.
        rng.shuffle(&mut d.deck[..kept]);
        d.deck_len = kept;
        d
    }

    /// One world: hands drawn from the belief by probability, a fresh deal of the unseen dev cards, and
    /// a fresh random seed.
    pub fn sample(&self, belief: &Belief, rng: &mut Rng) -> State {
        let mut w = self.template;
        let hands = belief.sample_hands(rng);
        let deal = self.deal(rng);
        for q in 0..NUM_PLAYERS {
            if q != self.viewer as usize {
                w.players[q].hand = hands[q];
                w.players[q].dev_hand = deal.dev_hand[q];
                w.players[q].dev_new = deal.dev_new[q];
            }
        }
        w.set_remaining_deck(&deal.deck[..deal.deck_len]);
        w.reseed(rng.next_u64());
        w
    }
}
