//! Rule variants and limits for one game.

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameConfig {
    pub vp_to_win: u8,
    /// Players holding more than this many cards discard half on a 7.
    pub discard_limit: u8,
    /// Domestic trade offers allowed per turn (0 disables domestic trading).
    pub max_offers_per_turn: u8,
    /// Max cards on each side of a domestic offer (1 or 2).
    pub max_trade_cards: u8,
    /// The game ends with no winner when this many turns have been played.
    pub max_turns: u32,
    /// Discards on a 7 are random instead of chosen, as Catanatron did before version 3.
    /// Catanatron 3.3 lets each player choose, like the default; leave this off against it.
    pub catanatron_compat: bool,
    /// The bank holds 19 cards of each resource and runs short (the default). Off, it never runs
    /// short, as in JSettlers2: production, maritime trades and Year of Plenty always pay, and
    /// `bank` shows what a 19-card bank would hold, 19 less the cards in hands, never below 0.
    pub bank_limit: bool,
}

/// As the derived `Debug` would print, except that `bank_limit` is written only when it is off:
/// the debug text of a state is hashed by the golden-trace pins, which stay unchanged with the
/// limit on (the default).
impl std::fmt::Debug for GameConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("GameConfig");
        d.field("vp_to_win", &self.vp_to_win)
            .field("discard_limit", &self.discard_limit)
            .field("max_offers_per_turn", &self.max_offers_per_turn)
            .field("max_trade_cards", &self.max_trade_cards)
            .field("max_turns", &self.max_turns)
            .field("catanatron_compat", &self.catanatron_compat);
        if !self.bank_limit {
            d.field("bank_limit", &self.bank_limit);
        }
        d.finish()
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            vp_to_win: 10,
            discard_limit: 7,
            max_offers_per_turn: 3,
            max_trade_cards: 2,
            max_turns: 1000,
            catanatron_compat: false,
            bank_limit: true,
        }
    }
}

impl GameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2).contains(&self.max_trade_cards) {
            return Err(format!(
                "max_trade_cards must be 1 or 2, got {}",
                self.max_trade_cards
            ));
        }
        if self.vp_to_win < 3 {
            return Err(format!(
                "vp_to_win must be at least 3, got {}",
                self.vp_to_win
            ));
        }
        if self.max_turns == 0 {
            return Err("max_turns must be positive".into());
        }
        Ok(())
    }
}
