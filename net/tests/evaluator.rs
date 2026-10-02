mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::evaluator::{absolute_value, masked_prior, NetEvaluator};
use settler_net::weights::{Arch, Weights};
use settler_search::{search_actions, Belief, Evaluator, Leaf, WorldSampler};
use std::sync::Arc;

fn net() -> Arc<Weights> {
    Arc::new(Weights::random(Arch::DEFAULT, 11))
}

fn searchable(s: &State) -> Vec<Action> {
    let mut legal = Vec::new();
    search_actions(&s.legal_actions(), &mut legal);
    legal
}

#[test]
fn the_prior_depends_only_on_what_the_actor_sees() {
    let mut eval = NetEvaluator::new(net());
    let mut rng = Rng::new(1);
    let (mut checked, mut values_differ) = (0, 0);
    for (i, s) in random_positions(20..26, 11, no_trades()).iter().enumerate() {
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let belief = Belief::from_observation(&obs, 32, i as u64);
        let sampler = WorldSampler::new(&obs, &belief).unwrap();
        let (w1, w2) = (
            sampler.sample(&belief, &mut rng),
            sampler.sample(&belief, &mut rng),
        );
        let legal = searchable(s);
        let e = eval.evaluate(&[
            Leaf {
                world: &w1,
                actor,
                legal: &legal,
            },
            Leaf {
                world: &w2,
                actor,
                legal: &legal,
            },
        ]);
        assert_eq!(e[0].prior, e[1].prior, "position {i}");
        values_differ += (e[0].value != e[1].value) as u32;
        checked += 1;
    }
    assert!(checked > 100, "{checked}");
    assert!(values_differ > 0, "the hidden block reaches the value head");
}

#[test]
fn values_and_priors_are_distributions_in_every_phase() {
    let mut eval = NetEvaluator::new(net());
    let mut trade_phases = 0;
    for s in random_positions(0..2, 13, GameConfig::default()) {
        trade_phases += matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) as u32;
        let legal = searchable(&s);
        let e = &eval.evaluate(&[Leaf {
            world: &s,
            actor: s.current_actor(),
            legal: &legal,
        }])[0];
        assert_eq!(e.prior.len(), legal.len());
        assert!((e.prior.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((e.value.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(e
            .prior
            .iter()
            .chain(&e.value)
            .all(|x| x.is_finite() && *x >= 0.0));
    }
    assert!(trade_phases > 0, "pending offers reached the evaluator");
}

#[test]
fn a_single_legal_move_gets_all_the_prior_and_offers_get_none() {
    let mut eval = NetEvaluator::new(net());
    let s = State::new(3, GameConfig::default());
    let actor = s.current_actor();
    let e = eval.evaluate(&[Leaf {
        world: &s,
        actor,
        legal: &[Action::Roll],
    }]);
    assert_eq!(e[0].prior, vec![1.0]);
    let offer = Action::OfferTrade { give: 0, get: 1 };
    let mixed = [Action::EndTurn, offer, Action::BuyDev];
    let e = eval.evaluate(&[Leaf {
        world: &s,
        actor,
        legal: &mixed,
    }]);
    assert_eq!(e[0].prior[1], 0.0);
    assert!((e[0].prior[0] + e[0].prior[2] - 1.0).abs() < 1e-6);
    assert_eq!(
        masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &[offer]),
        vec![1.0]
    );
}

#[test]
fn a_prior_with_no_finite_logit_is_uniform_over_the_moves_it_may_play() {
    let offer = Action::OfferTrade { give: 0, get: 1 };
    let dead = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
    assert_eq!(
        masked_prior(&dead, &[Action::EndTurn, offer, Action::BuyDev]),
        vec![0.5, 0.0, 0.5]
    );
    assert_eq!(masked_prior(&dead, &[offer, offer]), vec![0.5, 0.5]);
    let mut huge = vec![0.0; ACTION_SPACE_SIZE];
    huge[Action::EndTurn.encode() as usize] = f32::INFINITY;
    let p = masked_prior(&huge, &[Action::EndTurn, offer, Action::BuyDev]);
    assert_eq!(p, vec![1.0, 0.0, 0.0], "the softmax's limit, not NaN");
}

#[test]
fn values_are_rotated_back_to_absolute_seats() {
    let v = absolute_value(&[0.0, 1.0, 2.0, 3.0], 2);
    let e: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect();
    let z: f32 = e.iter().sum();
    // relative seat r is absolute seat (2 + r) % 4
    for (abs_seat, r) in [(2, 0), (3, 1), (0, 2), (1, 3)] {
        assert!((v[abs_seat] - e[r] / z).abs() < 1e-6, "{v:?}");
    }
}

#[test]
fn a_batch_evaluates_each_leaf_as_if_alone() {
    let mut eval = NetEvaluator::new(net());
    let positions = random_positions(3..4, 29, no_trades());
    let legal: Vec<Vec<Action>> = positions.iter().map(searchable).collect();
    let leaves: Vec<Leaf<'_>> = positions
        .iter()
        .zip(&legal)
        .map(|(s, l)| Leaf {
            world: s,
            actor: s.current_actor(),
            legal: l,
        })
        .collect();
    let together = eval.evaluate(&leaves);
    for (i, leaf) in leaves.iter().enumerate() {
        let alone = eval.evaluate(&[Leaf {
            world: leaf.world,
            actor: leaf.actor,
            legal: leaf.legal,
        }]);
        assert_eq!(alone[0], together[i], "leaf {i}");
    }
    assert_eq!(eval.evaluations(), 2 * leaves.len() as u64);
}

fn first(_: &Observation, legal: &[Action]) -> Action {
    legal[0]
}

#[test]
fn rollouts_keep_the_leaf_prior_and_score_the_position_they_reach() {
    let w = net();
    let mut plain = NetEvaluator::new(w.clone());
    let mut rolled = NetEvaluator::with_rollout(w, 4, first);
    let mut differ = 0;
    for s in random_positions(5..6, 17, no_trades()) {
        let legal = searchable(&s);
        let leaf = [Leaf {
            world: &s,
            actor: s.current_actor(),
            legal: &legal,
        }];
        let (a, b) = (plain.evaluate(&leaf), rolled.evaluate(&leaf));
        assert_eq!(a[0].prior, b[0].prior);
        assert!((b[0].value.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        differ += (a[0].value != b[0].value) as u32;
    }
    assert!(differ > 0);
    assert!(rolled.evaluations() > plain.evaluations());
}
