//! Rule variants and limits for one game.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    /// Discards on a 7 are random instead of chosen, as Catanatron does.
    pub catanatron_compat: bool,
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
