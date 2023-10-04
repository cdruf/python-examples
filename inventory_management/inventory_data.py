from dataclasses import dataclass


@dataclass
class Data:
    """Basic data with a stationary demand. """
    demand_mu: float
    demand_sig: float
    fixed_replenishment_cost: float
    backlog_cost: float
    holding_cost: float
    lead_time: float
