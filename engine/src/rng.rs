//! Small, fast, seedable randomness (SplitMix64). Not cryptographic.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`. `n` must be > 0.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }
}

/// Derive an independent stream seed from `a` and a salt/index `b`.
pub fn mix(a: u64, b: u64) -> u64 {
    Rng::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

const DICE_SALT: u64 = 0xD1CE;

/// The dice for `turn` of the game seeded `seed`. Independent of every bot decision.
pub fn dice_for(seed: u64, turn: u32) -> (u8, u8) {
    let mut r = Rng::new(mix(seed ^ DICE_SALT, turn as u64));
    (1 + r.below(6) as u8, 1 + r.below(6) as u8)
}
