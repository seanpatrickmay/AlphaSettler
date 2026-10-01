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

/// Deals rejected before giving up because every deal hands the player on turn a win.
const MAX_DEALS: u32 = 10_000;

pub struct WorldSampler {
    template: State,
    viewer: PlayerId,
    /// Dev cards the viewer cannot place (in opponents' hands or the deck), in kind order.
    unseen: [DevCard; DEV_DECK_SIZE],
    unseen_len: usize,
    /// Per player: dev cards held (0 for the viewer, whose cards are known) and how many are new.
    held: [u8; NUM_PLAYERS],
    new: [u8; NUM_PLAYERS],
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

impl WorldSampler {
    pub fn new(obs: &Observation, belief: &Belief) -> Result<WorldSampler, String> {
        belief.check(obs)?;
        if matches!(obs.phase, Phase::TradeResponse | Phase::TradeConfirm | Phase::GameOver { .. }) {
            return Err(format!("no worlds to sample in phase {:?}", obs.phase));
        }
        let me = obs.viewer as usize;
        let mut unseen_counts = DEV_DECK_COUNTS;
        for k in 0..5 {
            let played: u8 = (0..NUM_PLAYERS).map(|p| obs.dev_cards_played[p][k]).sum();
            unseen_counts[k] = unseen_counts[k]
                .checked_sub(played + obs.my_dev_cards[k])
                .ok_or_else(|| format!("more {:?} cards seen than the deck has", DevCard::from_index(k)))?;
        }
        let held: [u8; NUM_PLAYERS] = std::array::from_fn(|p| if p == me { 0 } else { obs.dev_card_counts[p] });
        let new: [u8; NUM_PLAYERS] = std::array::from_fn(|p| if p == me { 0 } else { obs.new_dev_card_counts[p] });
        let wanted = held.iter().map(|&c| c as u32).sum::<u32>() + obs.dev_deck_remaining as u32;
        let have: u32 = unseen_counts.iter().map(|&c| c as u32).sum();
        if wanted != have {
            return Err(format!("{have} unseen dev cards for {wanted} hidden places"));
        }
        let mut unseen = [DevCard::Knight; DEV_DECK_SIZE];
        let mut unseen_len = 0;
        for (k, &n) in unseen_counts.iter().enumerate() {
            for _ in 0..n {
                unseen[unseen_len] = DevCard::from_index(k);
                unseen_len += 1;
            }
        }
        let mut sampler = WorldSampler { template: State::new(0, obs.config), viewer: obs.viewer, unseen, unseen_len, held, new };
        // The template's hidden parts come from one sample; `sample` overwrites them each time.
        let mut rng = Rng::new(0x7E4D);
        let deal = sampler.deal(obs.current, obs.public_vp[obs.current as usize], obs.config.vp_to_win, &mut rng)?;
        let hands = belief.states()[0].0.hands;
        let snap = Snapshot {
            board: obs.board,
            robber: obs.robber,
            bank: obs.bank,
            dev_deck: deal.deck[..deal.deck_len].to_vec(),
            players: std::array::from_fn(|p| PlayerSnapshot {
                hand: hands[p],
                dev_hand: if p == me { obs.my_dev_cards } else { deal.dev_hand[p] },
                dev_new: if p == me { obs.my_new_dev_cards } else { deal.dev_new[p] },
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
    /// rejecting deals that would give `current` (an opponent on turn) enough VP to have won.
    fn deal(&self, current: PlayerId, current_public_vp: u8, vp_to_win: u8, rng: &mut Rng) -> Result<Deal, String> {
        for _ in 0..MAX_DEALS {
            let mut cards = self.unseen;
            rng.shuffle(&mut cards[..self.unseen_len]);
            let mut d = Deal {
                dev_hand: [[0; 5]; NUM_PLAYERS],
                dev_new: [[0; 5]; NUM_PLAYERS],
                deck: [DevCard::Knight; DEV_DECK_SIZE],
                deck_len: 0,
            };
            let mut next = 0;
            for q in 0..NUM_PLAYERS {
                for i in 0..self.held[q] {
                    let c = cards[next].index();
                    next += 1;
                    d.dev_hand[q][c] += 1;
                    if i < self.new[q] {
                        d.dev_new[q][c] += 1;
                    }
                }
            }
            d.deck_len = self.unseen_len - next;
            d.deck[..d.deck_len].copy_from_slice(&cards[next..self.unseen_len]);
            let hidden_vp = d.dev_hand[current as usize][DevCard::VictoryPoint.index()];
            if current == self.viewer || current_public_vp + hidden_vp < vp_to_win {
                return Ok(d);
            }
        }
        Err(format!("no dev-card deal keeps player {current} below {vp_to_win} VP"))
    }

    /// One world: hands drawn from the belief by probability, a fresh deal of the unseen dev cards, and
    /// a fresh random seed.
    pub fn sample(&self, belief: &Belief, rng: &mut Rng) -> State {
        let mut w = self.template;
        let hands = belief.sample_hands(rng);
        let current = w.current;
        let deal = self
            .deal(current, w.public_vp(current as usize), w.config.vp_to_win, rng)
            .expect("the template's own deal passed this check");
        for q in 0..NUM_PLAYERS {
            if q != self.viewer as usize {
                w.players[q].hand = hands[q];
                w.players[q].dev_hand = deal.dev_hand[q];
                w.players[q].dev_new = deal.dev_new[q];
            }
        }
        let rest = &deal.deck[..deal.deck_len];
        let pos = DEV_DECK_SIZE - rest.len();
        let mut drawn = DEV_DECK_COUNTS;
        for c in rest {
            drawn[c.index()] -= 1;
        }
        let mut i = 0;
        for (k, &n) in drawn.iter().enumerate() {
            for _ in 0..n {
                w.dev_deck[i] = DevCard::from_index(k);
                i += 1;
            }
        }
        w.dev_deck[pos..].copy_from_slice(rest);
        w.dev_deck_pos = pos as u8;
        w.reseed(rng.next_u64());
        w
    }
}
