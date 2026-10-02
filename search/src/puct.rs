//! Selection arithmetic, kept apart from the tree walk so it can be tested on synthetic children.

use crate::tree::Child;

/// Value assumed for an unvisited child when nothing at the node has been visited yet.
pub const DEFAULT_FPU: f32 = 0.25;

/// PUCT score of `c` for `actor`. Exploration uses the child's availability count instead of
/// the parent's visit count, as ISMCTS does when legal moves differ between worlds. Pending
/// (virtual-loss) visits count as visits worth nothing.
pub fn score(c: &Child, actor: usize, c_puct: f32, fpu: f32) -> f32 {
    let n = c.visits + c.pending;
    let q = if n == 0 {
        fpu
    } else {
        c.value_sum[actor] / n as f32
    };
    q + c_puct * c.prior * (c.available.max(1) as f32).sqrt() / (1.0 + n as f32)
}

/// First-play urgency: the mean value for `actor` over the node's visited children.
pub fn fpu(children: &[Child], actor: usize) -> f32 {
    let (sum, n) = children.iter().fold((0.0, 0u32), |(s, n), c| {
        (s + c.value_sum[actor], n + c.visits)
    });
    if n == 0 {
        DEFAULT_FPU
    } else {
        sum / n as f32
    }
}

/// The best child for `actor` among those `allowed` (max^n: each player maximises their own
/// value). Ties go to the lower index.
pub fn select(
    children: &[Child],
    allowed: impl Fn(&Child) -> bool,
    actor: usize,
    c_puct: f32,
) -> Option<usize> {
    let f = fpu(children, actor);
    let mut best: Option<(usize, f32)> = None;
    for (i, c) in children.iter().enumerate() {
        if !allowed(c) {
            continue;
        }
        let s = score(c, actor, c_puct, f);
        if best.map_or(true, |(_, b)| s > b) {
            best = Some((i, s));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child(key: u16, visits: u32, available: u32, value_sum: [f32; 4]) -> Child {
        Child {
            visits,
            available,
            value_sum,
            ..Child::new(key, 0.5)
        }
    }

    #[test]
    fn each_player_maximises_their_own_value() {
        let kids = [
            child(0, 10, 10, [0.0, 0.0, 9.0, 0.0]),
            child(1, 10, 10, [9.0, 0.0, 0.0, 0.0]),
        ];
        assert_eq!(select(&kids, |_| true, 2, 0.0), Some(0));
        assert_eq!(select(&kids, |_| true, 0, 0.0), Some(1));
    }

    #[test]
    fn children_not_allowed_in_this_world_are_skipped() {
        let kids = [child(0, 10, 10, [9.0; 4]), child(1, 10, 10, [1.0; 4])];
        assert_eq!(select(&kids, |c| c.key == 1, 0, 1.0), Some(1));
        assert_eq!(select(&kids, |_| false, 0, 1.0), None);
    }

    #[test]
    fn exploration_grows_with_availability() {
        let kids = [child(0, 0, 1, [0.0; 4]), child(1, 0, 100, [0.0; 4])];
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(1));
    }

    #[test]
    fn pending_visits_lower_the_score() {
        let mut kids = [child(0, 4, 8, [2.0; 4]), child(1, 4, 8, [2.0; 4])];
        kids[0].pending = 3;
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(1));
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        let kids = [child(0, 3, 3, [1.0; 4]), child(1, 3, 3, [1.0; 4])];
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(0));
    }
}
