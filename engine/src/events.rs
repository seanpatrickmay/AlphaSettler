//! What happened, for logs and observers. `Stole` and `BoughtDev` carry private details.

use crate::types::{DevCard, Hand, PlayerId, Resource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    BuiltSettlement {
        player: PlayerId,
        node: u8,
    },
    BuiltCity {
        player: PlayerId,
        node: u8,
    },
    BuiltRoad {
        player: PlayerId,
        edge: u8,
    },
    Rolled {
        player: PlayerId,
        dice: (u8, u8),
    },
    Produced {
        player: PlayerId,
        resources: Hand,
    },
    Discarded {
        player: PlayerId,
        resource: Resource,
    },
    RobberMoved {
        player: PlayerId,
        tile: u8,
    },
    /// `resource` is `None` in views of players other than thief and victim.
    Stole {
        thief: PlayerId,
        victim: PlayerId,
        resource: Option<Resource>,
    },
    /// `card` is `None` in views of players other than the buyer.
    BoughtDev {
        player: PlayerId,
        card: Option<DevCard>,
    },
    PlayedDev {
        player: PlayerId,
        card: DevCard,
    },
    MonopolyTaken {
        player: PlayerId,
        resource: Resource,
        amount: u8,
    },
    YearOfPlentyTaken {
        player: PlayerId,
        resources: Hand,
    },
    MaritimeTraded {
        player: PlayerId,
        gave: Hand,
        got: Hand,
    },
    TradeOffered {
        player: PlayerId,
        give: Hand,
        get: Hand,
    },
    TradeResponded {
        player: PlayerId,
        accepted: bool,
    },
    TradeConfirmed {
        offerer: PlayerId,
        partner: PlayerId,
        offerer_gave: Hand,
        partner_gave: Hand,
    },
    TradeCancelled {
        player: PlayerId,
    },
    TurnEnded {
        player: PlayerId,
    },
    GameOver {
        winner: Option<PlayerId>,
    },
}

impl Event {
    /// This event as `viewer` is allowed to see it.
    pub fn redacted_for(self, viewer: PlayerId) -> Event {
        match self {
            Event::Stole { thief, victim, .. } if viewer != thief && viewer != victim => {
                Event::Stole {
                    thief,
                    victim,
                    resource: None,
                }
            }
            Event::BoughtDev { player, .. } if viewer != player => {
                Event::BoughtDev { player, card: None }
            }
            e => e,
        }
    }
}
