use settler_bots::greedy::GreedyBot;
use settler_bots::heuristic::{features, prior, value, HeuristicEvaluator, DEFAULT_WEIGHTS, NUM_FEATURES};
use settler_bots::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::*;
use settler_search::{search_actions, Belief, Evaluator, Leaf, WorldSampler};

/// Positions from greedy self-play, every 9th move, skipping trade phases.
fn positions(seeds: std::ops::Range<u64>) -> Vec<State> {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let mut out = Vec::new();
    for seed in seeds {
        let mut s = State::new(seed, cfg);
        let mut bots: Vec<GreedyBot> = (0..4).map(GreedyBot::new).collect();
        let mut buf = Vec::new();
        let mut n = 0;
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            let actor = s.current_actor() as usize;
            if n % 9 == 0 {
                out.push(s);
            }
            let a = bots[actor].act(&s.observation(actor as u8), &buf);
            s.apply(a);
            n += 1;
        }
    }
    out
}

#[test]
fn values_are_probabilities_and_favour_the_leader() {
    for s in positions(0..6) {
        let v = value(&s, &DEFAULT_WEIGHTS);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(v.iter().all(|&x| x > 0.0));
        let scores: Vec<f32> = (0..4).map(|p| features(&s, p).iter().zip(&DEFAULT_WEIGHTS).map(|(f, w)| f * w).sum()).collect();
        let best = (0..4).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap();
        assert!((0..4).all(|p| v[best] >= v[p]));
    }
    assert_eq!(NUM_FEATURES, DEFAULT_WEIGHTS.len());
}

#[test]
fn the_prior_is_a_distribution_over_legal_moves() {
    for s in positions(0..3) {
        let actor = s.current_actor();
        let mut legal = Vec::new();
        search_actions(&s.legal_actions(), &mut legal);
        let p = prior(&s.observation(actor), &legal);
        assert_eq!(p.len(), legal.len());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}

#[test]
fn the_prior_depends_only_on_what_the_actor_sees() {
    let mut eval = HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0);
    let mut rng = settler_engine::rng::Rng::new(1);
    let mut checked = 0;
    for s in positions(10..16) {
        if matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) {
            continue;
        }
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let belief = Belief::from_observation(&obs, 32, checked);
        let sampler = WorldSampler::new(&obs, &belief).unwrap();
        let (w1, w2) = (sampler.sample(&belief, &mut rng), sampler.sample(&belief, &mut rng));
        let mut legal = Vec::new();
        search_actions(&s.legal_actions(), &mut legal);
        let e = eval.evaluate(&[Leaf { world: &w1, actor, legal: &legal }, Leaf { world: &w2, actor, legal: &legal }]);
        assert_eq!(e[0].prior, e[1].prior);
        checked += 1;
    }
    assert!(checked > 100);
}
