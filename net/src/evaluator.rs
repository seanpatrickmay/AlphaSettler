//! `NetEvaluator`: `search::Evaluator` backed by the network (spec Section 3). For each leaf it
//! encodes `world.observation(actor)` and the hidden block, runs the batch through one forward
//! pass, masks and softmaxes the policy over `legal`, and rotates the value back to absolute seats.

use crate::encode::{encode_state, rel, HIDDEN_LEN, PUBLIC_LEN};
use crate::forward::{softmax, Forward, Outputs};
use crate::policy::policy_id;
use crate::weights::Weights;
use settler_engine::legal::legal_actions;
use settler_engine::{Action, Observation, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_search::{search_actions, terminal_value, Eval, Evaluator, Leaf};
use std::sync::Arc;

/// Picks a rollout move from an observation and the searchable moves (bots pass GreedyBot's).
pub type RolloutPolicy = fn(&Observation, &[Action]) -> Action;

/// The prior over `legal` for `actor`: a softmax of the policy logits at the moves' policy ids
/// (`policy::policy_id`: steals and trade confirmations seat-relative, as in the training
/// targets). Domestic offers reach it only from a search with `SearchConfig::offers` and read
/// their own logits like any move. With no finite maximum (every logit -inf, or some +inf) it is
/// the softmax's limit: uniform over the moves at the maximum, never NaN.
pub fn masked_prior(logits: &[f32], legal: &[Action], actor: PlayerId) -> Vec<f32> {
    let mut p: Vec<f32> = legal.iter().map(|a| logits[policy_id(*a, actor)]).collect();
    let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_finite() {
        softmax(&mut p);
        return p;
    }
    for x in p.iter_mut() {
        *x = (*x == max) as u8 as f32;
    }
    let n: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= n);
    p
}

/// Win probabilities by absolute seat from the value logits of `actor`'s seat-relative view
/// (rotated with the encoder's own `rel`). Logits with a NaN or no finite maximum give ¼ each.
pub fn absolute_value(logits: &[f32], actor: PlayerId) -> [f32; NUM_PLAYERS] {
    let mut p = [logits[0], logits[1], logits[2], logits[3]];
    let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() || p.iter().any(|x| x.is_nan()) {
        return [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS];
    }
    softmax(&mut p);
    std::array::from_fn(|q| p[rel(q as PlayerId, actor)])
}

/// The net's win probabilities, by absolute seat, for `s` seen by `viewer`: the public block from
/// `s.observation(viewer)`, the hidden block from `s` itself, the value head rotated back to
/// absolute seats. This is the evaluation a search leaf whose world is `s` gets: exactly the
/// value of a one-leaf `NetEvaluator::evaluate`, and within float rounding of the same leaf in a
/// larger batch (the Accelerate forward pass, `forward::dense`, matches a position run alone only
/// to about 1e-5 relative; the same batch run twice is bit-identical).
pub fn position_value(weights: &Weights, s: &State, viewer: PlayerId) -> [f32; NUM_PLAYERS] {
    let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
    encode_state(s, viewer, &mut public, &mut hidden);
    let mut out = Outputs::default();
    Forward::new().run(weights, &public, &hidden, 1, &mut out);
    absolute_value(&out.value[..NUM_PLAYERS], viewer)
}

pub struct NetEvaluator {
    weights: Arc<Weights>,
    rollout: u32,
    policy: Option<RolloutPolicy>,
    forward: Forward,
    out: Outputs,
    public: Vec<f32>,
    hidden: Vec<f32>,
    buf: Vec<Action>,
    moves: Vec<Action>,
    evaluations: u64,
}

impl NetEvaluator {
    /// Weights are shared read-only (`Arc`): load them once per process.
    pub fn new(weights: Arc<Weights>) -> NetEvaluator {
        NetEvaluator {
            weights,
            rollout: 0,
            policy: None,
            forward: Forward::new(),
            out: Outputs::default(),
            public: Vec::new(),
            hidden: Vec::new(),
            buf: Vec::new(),
            moves: Vec::new(),
            evaluations: 0,
        }
    }

    /// The prior still comes from the leaf; the value comes from the position reached after
    /// `rollout` moves of `policy` in the leaf's own sampled world (a finished game scores as
    /// `terminal_value`).
    pub fn with_rollout(
        weights: Arc<Weights>,
        rollout: u32,
        policy: RolloutPolicy,
    ) -> NetEvaluator {
        NetEvaluator {
            rollout,
            policy: Some(policy),
            ..NetEvaluator::new(weights)
        }
    }

    /// Positions run through the network so far.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub fn weights(&self) -> &Arc<Weights> {
        &self.weights
    }

    /// Fill the input buffers for `positions` and run them.
    fn run(&mut self, positions: &[(&State, PlayerId)]) {
        let n = positions.len();
        self.public.resize(n * PUBLIC_LEN, 0.0);
        self.hidden.resize(n * HIDDEN_LEN, 0.0);
        for (i, &(s, actor)) in positions.iter().enumerate() {
            encode_state(
                s,
                actor,
                &mut self.public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
                &mut self.hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
            );
        }
        self.forward
            .run(&self.weights, &self.public, &self.hidden, n, &mut self.out);
        self.evaluations += n as u64;
    }

    fn roll_out(&mut self, world: &State, policy: RolloutPolicy) -> State {
        let mut s = *world;
        for _ in 0..self.rollout {
            if s.is_over() {
                break;
            }
            legal_actions(&s, &mut self.buf);
            search_actions(&self.buf, &mut self.moves);
            let a = policy(&s.observation(s.current_actor()), &self.moves);
            s.apply(a);
        }
        s
    }
}

impl Evaluator for NetEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let positions: Vec<(&State, PlayerId)> =
            leaves.iter().map(|l| (l.world, l.actor)).collect();
        self.run(&positions);
        let mut evals: Vec<Eval> = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| Eval {
                value: absolute_value(
                    &self.out.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
                    l.actor,
                ),
                prior: masked_prior(
                    &self.out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
                    l.legal,
                    l.actor,
                ),
            })
            .collect();
        let rollout_policy = if self.rollout > 0 { self.policy } else { None };
        if let Some(policy) = rollout_policy {
            let mut rolled: Vec<(usize, State)> = Vec::new();
            for (i, l) in leaves.iter().enumerate() {
                let s = self.roll_out(l.world, policy);
                if s.is_over() {
                    evals[i].value = terminal_value(&s);
                } else {
                    rolled.push((i, s));
                }
            }
            if !rolled.is_empty() {
                let positions: Vec<(&State, PlayerId)> =
                    rolled.iter().map(|(_, s)| (s, s.current_actor())).collect();
                self.run(&positions);
                for (j, (i, s)) in rolled.iter().enumerate() {
                    evals[*i].value = absolute_value(
                        &self.out.value[j * NUM_PLAYERS..(j + 1) * NUM_PLAYERS],
                        s.current_actor(),
                    );
                }
            }
        }
        evals
    }
}
