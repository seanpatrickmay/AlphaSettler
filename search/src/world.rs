//! Sampled worlds: complete states consistent with everything one player has seen.
//!
//! Built once per decision: a template `State` (validated by `State::from_snapshot`) carries
//! everything public plus the viewer's own cards. Each sample copies it and overwrites only what
//! the viewer cannot see: opponents' hands (a belief particle), their dev cards and the deck
//! order (a uniform deal of the unseen cards), and the random seed.

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
    /// Weights of how many VP cards the opponent on turn holds (index = count), when that opponent
    /// is not the viewer: the hypergeometric weights cut off at the VP that would win the game.
    on_turn_vp: Option<(usize, [f64; 6])>,
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
        let on_turn_vp = if obs.current == obs.viewer {
            None
        } else {
            let public = obs.public_vp[obs.current as usize];
            if public >= obs.config.vp_to_win {
                return Err(format!(
                    "player {} has {public} public VP with the game still on",
                    obs.current
                ));
            }
            let (n, v, h) = (
                unseen_len,
                unseen_counts[DevCard::VictoryPoint.index()] as usize,
                held[obs.current as usize] as usize,
            );
            let most = ((obs.config.vp_to_win - public - 1) as usize).min(v).min(h);
            let mut w = [0.0; 6];
            for (k, wk) in w.iter_mut().enumerate().take(most + 1) {
                *wk = choose(v, k) * choose(n - v, h - k);
            }
            if w.iter().all(|&x| x == 0.0) {
                return Err(format!(
                    "no dev-card deal keeps player {} below {} VP",
                    obs.current, obs.config.vp_to_win
                ));
            }
            Some((obs.current as usize, w))
        };
        let mut sampler = WorldSampler {
            template: State::new(0, obs.config),
            viewer: obs.viewer,
            unseen,
            unseen_len,
            held,
            new,
            on_turn_vp,
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

    /// A uniform deal of the unseen dev cards to opponents (new cards first) and the deck, except that
    /// an opponent on turn never holds enough VP cards to have already won. That is the uniform deal
    /// conditioned on it, drawn exactly: first how many VP cards that opponent holds (truncated
    /// hypergeometric), then their cards from the VP and other parts of the pool, then everything
    /// left uniformly to the others and the deck.
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
        let mut dealt = [false; NUM_PLAYERS];
        let mut rest: &[DevCard] = pool;
        let mut leftover = [DevCard::Knight; DEV_DECK_SIZE];
        if let Some((cur, weights)) = self.on_turn_vp {
            let total: f64 = weights.iter().sum();
            let mut u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
            let mut k = weights
                .iter()
                .rposition(|&w| w > 0.0)
                .expect("`new` checked a deal exists");
            for (i, &w) in weights.iter().enumerate() {
                if u < w {
                    k = i;
                    break;
                }
                u -= w;
            }
            // The pool is in random order, so its first `k` VP cards and first `h - k` other cards are
            // a uniform choice of each.
            let (mut vp_left, mut other_left) = (k, self.held[cur] as usize - k);
            let mut mine = [DevCard::Knight; DEV_DECK_SIZE];
            let (mut taken, mut kept) = (0, 0);
            for &c in pool.iter() {
                let want = if c == DevCard::VictoryPoint {
                    &mut vp_left
                } else {
                    &mut other_left
                };
                if *want > 0 {
                    *want -= 1;
                    mine[taken] = c;
                    taken += 1;
                } else {
                    leftover[kept] = c;
                    kept += 1;
                }
            }
            rng.shuffle(&mut mine[..taken]);
            for (i, c) in mine[..taken].iter().enumerate() {
                d.dev_hand[cur][c.index()] += 1;
                if i < self.new[cur] as usize {
                    d.dev_new[cur][c.index()] += 1;
                }
            }
            dealt[cur] = true;
            // Taking the first cards of each kind skews the order of what is left (it tends to start
            // with whichever kind lost fewer cards), so shuffle it afresh.
            rng.shuffle(&mut leftover[..kept]);
            rest = &leftover[..kept];
        }
        let mut next = 0;
        for q in (0..NUM_PLAYERS).filter(|&q| !dealt[q]) {
            for i in 0..self.held[q] {
                let c = rest[next].index();
                next += 1;
                d.dev_hand[q][c] += 1;
                if i < self.new[q] {
                    d.dev_new[q][c] += 1;
                }
            }
        }
        d.deck_len = rest.len() - next;
        d.deck[..d.deck_len].copy_from_slice(&rest[next..]);
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
