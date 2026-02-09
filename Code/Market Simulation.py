#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
from collections import defaultdict

# =============================================================================
# MARKET MECHANISM IMPLEMENTATIONS
# =============================================================================

def amm_bid_update(bid: List[float], money_supply: float, offset_supply: float, k: float) -> Tuple[float, float, float, float]:
    """
    AMM bid processing using constant product formula.
    bid = [price, quantity]
    Buyer wants to purchase tokens at up to 'price' per unit.
    """
    price, quantity = bid
    # Maximum tokens buyer can get at their price limit
    max_at_price = max(0, offset_supply - np.sqrt(k / price))
    purchased = min(quantity, max_at_price)
    
    if purchased <= 0:
        return 0, 0, money_supply, offset_supply
    
    # Calculate payment: difference in money_supply before and after
    old_money = money_supply
    offset_supply_new = offset_supply - purchased
    money_supply_new = k / offset_supply_new
    payment = money_supply_new - old_money  # Money that flows in
    
    return purchased, payment, money_supply_new, offset_supply_new


def amm_ask_update(ask: List[float], money_supply: float, offset_supply: float, k: float) -> Tuple[float, float, float, float]:
    """
    AMM ask processing using constant product formula.
    ask = [price, quantity]
    Seller wants to sell tokens at at least 'price' per unit.
    """
    price, quantity = ask
    # Maximum tokens that can be sold at seller's price floor
    max_at_price = max(0, np.sqrt(k / price) - offset_supply)
    sold = min(quantity, max_at_price)
    
    if sold <= 0:
        return 0, 0, money_supply, offset_supply
    
    # Calculate payment: difference in money_supply before and after
    old_money = money_supply
    offset_supply_new = offset_supply + sold
    money_supply_new = k / offset_supply_new
    payment = old_money - money_supply_new  # Money that flows out to seller
    
    return sold, payment, money_supply_new, offset_supply_new


def cda_bid_match(bid: List[float], bids: List[List[float]], asks: List[List[float]], alpha: float) -> Tuple[float, float, List[List[float]], List[List[float]]]:
    """
    CDA bid matching against ask book.
    Returns: (total_payment, trade_volume, updated_bids, updated_asks)
    """
    total_payment = 0.0
    bid_quantity = bid[1]
    
    # Sort asks by price ascending
    asks.sort(key=lambda x: x[0])
    
    for ask in asks:
        if ask[0] <= bid[0] and bid[1] > 0:  # Trade possible
            volume = min(ask[1], bid[1])
            price = (1 - alpha) * ask[0] + alpha * bid[0]
            ask[1] -= volume
            bid[1] -= volume
            total_payment += price * volume
    
    trade_volume = bid_quantity - bid[1]
    
    # Remove filled asks
    asks = [a for a in asks if a[1] > 1e-10]
    
    # Add remaining bid to book
    if bid[1] > 1e-10:
        bids.append(bid)
    
    return total_payment, trade_volume, bids, asks


def cda_ask_match(ask: List[float], bids: List[List[float]], asks: List[List[float]], alpha: float) -> Tuple[float, float, List[List[float]], List[List[float]]]:
    """
    CDA ask matching against bid book.
    Returns: (total_payment, trade_volume, updated_bids, updated_asks)
    """
    total_payment = 0.0
    ask_quantity = ask[1]
    
    # Sort bids by price descending
    bids.sort(key=lambda x: -x[0])
    
    for bid in bids:
        if ask[0] <= bid[0] and ask[1] > 0:  # Trade possible
            volume = min(ask[1], bid[1])
            price = (1 - alpha) * ask[0] + alpha * bid[0]
            ask[1] -= volume
            bid[1] -= volume
            total_payment += price * volume
    
    trade_volume = ask_quantity - ask[1]
    
    # Remove filled bids
    bids = [b for b in bids if b[1] > 1e-10]
    
    # Add remaining ask to book
    if ask[1] > 1e-10:
        asks.append(ask)
    
    return total_payment, trade_volume, bids, asks


def compute_breakeven_indices(bids: np.ndarray, asks: np.ndarray) -> Tuple[Optional[int], Optional[int], any]:
    """
    Find breakeven indices for SDA clearing.
    """
    n = bids.shape[0]
    m = asks.shape[0]
    
    if n == 0 or m == 0:
        return None, None, "No Trade"
    
    # Cumulative quantities
    cum_bid_qty = np.cumsum(bids[:, 1])
    cum_ask_qty = np.cumsum(asks[:, 1])
    
    k = 0
    l = 0
    
    while k < n and l < m:
        cum_bid_k = cum_bid_qty[k]
        cum_ask_l = cum_ask_qty[l]
        bid_price = bids[k, 0]
        ask_price = asks[l, 0]
        
        if cum_bid_k > cum_ask_l:
            if bid_price >= ask_price:
                if k + 1 >= n or ask_price >= bids[k + 1, 0]:
                    if l > 0 and cum_ask_qty[l - 1] <= cum_bid_k <= cum_ask_qty[l]:
                        return k, l, 1
            l += 1
        else:
            if ask_price <= bid_price:
                if l + 1 >= m or bids[k, 0] <= asks[l + 1, 0]:
                    if k > 0 and cum_bid_qty[k - 1] <= cum_ask_l <= cum_bid_qty[k]:
                        return k, l, 2
            k += 1
    
    return None, None, "No Trade"


def execute_sda_case(bids: np.ndarray, asks: np.ndarray, K: int, L: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Execute SDA clearing for both Case I and Case II.
    Returns: (buy_price, sell_price, buy_quantities, sell_quantities)
    """
    buy_price = bids[K, 0]
    sell_price = asks[L, 0]
    
    n = bids.shape[0]
    m = asks.shape[0]
    
    buy_quantities = np.zeros(n)
    sell_quantities = np.zeros(m)
    
    if K < 1 or L < 1:
        return buy_price, sell_price, buy_quantities, sell_quantities
    
    cum_bid_sum = np.sum(bids[:K, 1])
    cum_ask_sum = np.sum(asks[:L, 1])
    
    if cum_bid_sum >= cum_ask_sum:
        # All L-1 sellers sell everything, buyers share reduction
        active_buyers = list(range(K))
        remaining_demand = cum_bid_sum - cum_ask_sum
        
        while remaining_demand > 1e-10 and len(active_buyers) > 0:
            reduction_per_buyer = remaining_demand / len(active_buyers)
            to_remove = []
            
            for i in active_buyers:
                if bids[i, 1] <= reduction_per_buyer:
                    to_remove.append(i)
            
            if len(to_remove) == 0:
                for i in active_buyers:
                    buy_quantities[i] = bids[i, 1] - reduction_per_buyer
                break
            else:
                for i in to_remove:
                    buy_quantities[i] = 0
                    remaining_demand -= bids[i, 1]
                active_buyers = [i for i in active_buyers if i not in to_remove]
        
        for j in range(L):
            sell_quantities[j] = asks[j, 1]
    else:
        # All K-1 buyers buy everything, sellers share reduction
        active_sellers = list(range(L))
        remaining_supply = cum_ask_sum - cum_bid_sum
        
        while remaining_supply > 1e-10 and len(active_sellers) > 0:
            reduction_per_seller = remaining_supply / len(active_sellers)
            to_remove = []
            
            for j in active_sellers:
                if asks[j, 1] <= reduction_per_seller:
                    to_remove.append(j)
            
            if len(to_remove) == 0:
                for j in active_sellers:
                    sell_quantities[j] = asks[j, 1] - reduction_per_seller
                break
            else:
                for j in to_remove:
                    sell_quantities[j] = 0
                    remaining_supply -= asks[j, 1]
                active_sellers = [j for j in active_sellers if j not in to_remove]
        
        for i in range(K):
            buy_quantities[i] = bids[i, 1]
    
    return buy_price, sell_price, buy_quantities, sell_quantities


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SimulationParams:
    """Parameters for the simulation."""
    # Time parameters
    T: int = 1000
    
    # Arrival rates
    lambda_bids: float = 5.0
    lambda_asks: float = 5.0
    
    # Bid price and quantity bounds
    p_bid: float = 8.0
    P_bid: float = 12.0
    q_bid: float = 1.0
    Q_bid: float = 10.0
    
    # Ask price and quantity bounds
    p_ask: float = 8.0
    P_ask: float = 12.0
    q_ask: float = 1.0
    Q_ask: float = 10.0
    
    # CDA parameters
    alpha: float = 0.5
    
    # AMM parameters
    k: float = 10000.0
    initial_offset_supply: float = 100.0
    
    # SDA auction trigger parameters
    t_max: int = 10
    t_min: int = 3
    tau_bid: int = 5
    tau_ask: int = 5
    
    # Random seed
    seed: int = 42


@dataclass
class TradeRecord:
    """Record of a single trade execution."""
    time: int
    price: float
    quantity: float
    buyer_value: float
    seller_cost: float


@dataclass
class MechanismResults:
    """Results from running a single mechanism."""
    trades: List[TradeRecord] = field(default_factory=list)
    total_volume: float = 0.0
    total_value: float = 0.0
    prices: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    times: List[int] = field(default_factory=list)
    final_bid_book: List[List[float]] = field(default_factory=list)
    final_ask_book: List[List[float]] = field(default_factory=list)


@dataclass
class SimulationResults:
    """Combined results from all three mechanisms."""
    params: SimulationParams
    cda: MechanismResults
    amm: MechanismResults
    sda: MechanismResults
    all_bids: List[Tuple[int, float, float]]
    all_asks: List[Tuple[int, float, float]]


# =============================================================================
# ORDER GENERATION
# =============================================================================

def generate_orders(params: SimulationParams) -> Tuple[List[Tuple[int, float, float]], List[Tuple[int, float, float]]]:
    """
    Generate all bids and asks for the simulation according to Poisson arrival process.
    Returns: (all_bids, all_asks) where each entry is (time, price, quantity)
    """
    np.random.seed(params.seed)
    random.seed(params.seed)
    
    all_bids = []
    all_asks = []
    
    for t in range(1, params.T + 1):
        # Generate bids for this timestep
        n_bids = np.random.poisson(params.lambda_bids)
        for _ in range(n_bids):
            price = np.random.uniform(params.p_bid, params.P_bid)
            qty = np.random.uniform(params.q_bid, params.Q_bid)
            all_bids.append((t, price, qty))
        
        # Generate asks for this timestep
        n_asks = np.random.poisson(params.lambda_asks)
        for _ in range(n_asks):
            price = np.random.uniform(params.p_ask, params.P_ask)
            qty = np.random.uniform(params.q_ask, params.Q_ask)
            all_asks.append((t, price, qty))
    
    return all_bids, all_asks


# =============================================================================
# CDA SIMULATION
# =============================================================================

def run_cda_simulation(all_bids: List, all_asks: List, params: SimulationParams) -> MechanismResults:
    """
    Run Continuous Double Auction simulation.
    Orders arrive and are immediately matched against the order book.
    """
    results = MechanismResults()
    
    bids = []  # Order book
    asks = []
    
    # Group orders by timestep
    bids_by_time = defaultdict(list)
    asks_by_time = defaultdict(list)
    
    for t, p, q in all_bids:
        bids_by_time[t].append((p, q))
    for t, p, q in all_asks:
        asks_by_time[t].append((p, q))
    
    np.random.seed(params.seed)
    random.seed(params.seed)
    
    for t in range(1, params.T + 1):
        t_bids = bids_by_time[t]
        t_asks = asks_by_time[t]
        
        # Create order queue with type tags and shuffle
        orders = [('bid', p, q) for p, q in t_bids] + [('ask', p, q) for p, q in t_asks]
        random.shuffle(orders)
        
        for order_type, price, qty in orders:
            if order_type == 'bid':
                bid = [price, qty]
                original_price = price
                payment, volume, bids, asks = cda_bid_match(bid, bids, asks, params.alpha)
                
                if volume > 1e-10:
                    avg_price = payment / volume
                    results.trades.append(TradeRecord(t, avg_price, volume, original_price, avg_price))
                    results.prices.append(avg_price)
                    results.volumes.append(volume)
                    results.times.append(t)
                    results.total_volume += volume
                    results.total_value += payment
            else:
                ask = [price, qty]
                original_price = price
                payment, volume, bids, asks = cda_ask_match(ask, bids, asks, params.alpha)
                
                if volume > 1e-10:
                    avg_price = payment / volume
                    results.trades.append(TradeRecord(t, avg_price, volume, avg_price, original_price))
                    results.prices.append(avg_price)
                    results.volumes.append(volume)
                    results.times.append(t)
                    results.total_volume += volume
                    results.total_value += payment
    
    results.final_bid_book = bids
    results.final_ask_book = asks
    
    return results


# =============================================================================
# AMM SIMULATION
# =============================================================================

def run_amm_simulation(all_bids: List, all_asks: List, params: SimulationParams) -> MechanismResults:
    """
    Run Automated Market Maker simulation.
    Uses constant product formula: money_supply * offset_supply = k
    """
    results = MechanismResults()
    
    # Initialize AMM state
    offset_supply = params.initial_offset_supply
    money_supply = params.k / offset_supply
    k = params.k
    
    # Group orders by timestep
    bids_by_time = defaultdict(list)
    asks_by_time = defaultdict(list)
    
    for t, p, q in all_bids:
        bids_by_time[t].append((p, q))
    for t, p, q in all_asks:
        asks_by_time[t].append((p, q))
    
    np.random.seed(params.seed)
    random.seed(params.seed)
    
    for t in range(1, params.T + 1):
        t_bids = bids_by_time[t]
        t_asks = asks_by_time[t]
        
        # Create order queue with type tags and shuffle
        orders = [('bid', p, q) for p, q in t_bids] + [('ask', p, q) for p, q in t_asks]
        random.shuffle(orders)
        
        for order_type, price, qty in orders:
            if order_type == 'bid':
                bid = [price, qty]
                purchased, payment, money_supply, offset_supply = amm_bid_update(bid, money_supply, offset_supply, k)
                
                if purchased > 1e-10:
                    avg_price = payment / purchased
                    results.trades.append(TradeRecord(t, avg_price, purchased, price, avg_price))
                    results.prices.append(avg_price)
                    results.volumes.append(purchased)
                    results.times.append(t)
                    results.total_volume += purchased
                    results.total_value += payment
            else:
                ask = [price, qty]
                sold, payment, money_supply, offset_supply = amm_ask_update(ask, money_supply, offset_supply, k)
                
                if sold > 1e-10:
                    avg_price = payment / sold
                    results.trades.append(TradeRecord(t, avg_price, sold, avg_price, price))
                    results.prices.append(avg_price)
                    results.volumes.append(sold)
                    results.times.append(t)
                    results.total_volume += sold
                    results.total_value += payment
    
    return results


# =============================================================================
# SDA SIMULATION
# =============================================================================

def compute_clearing_price_and_quantities(bids: np.ndarray, asks: np.ndarray) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    n_bids = bids.shape[0]
    n_asks = asks.shape[0]
    
    if n_bids == 0 or n_asks == 0:
        return None, np.zeros(0), np.zeros(0)
    
    # Check if any trade possible
    if bids[0, 0] < asks[0, 0]:  # Best bid < best ask
        return None, np.zeros(n_bids), np.zeros(n_asks)
    
    # Build demand curve: cumulative quantity at each price level (descending prices)
    # Build supply curve: cumulative quantity at each price level (ascending prices)
    
    # Find intersection: highest price where cumulative demand >= cumulative supply
    # and cumulative supply > 0
    
    cum_demand = np.cumsum(bids[:, 1])
    cum_supply = np.cumsum(asks[:, 1])
    
    # Find the clearing point
    # We want to find k (number of bids) and l (number of asks) such that:
    # - bids[k-1, 0] >= asks[l-1, 0] (trade is profitable)
    # - The market clears optimally
    
    best_volume = 0
    best_k = 0
    best_l = 0
    clearing_price = None
    
    # Try all combinations to find maximum traded volume
    for k in range(1, n_bids + 1):
        bid_price = bids[k-1, 0]
        total_bid_qty = cum_demand[k-1]
        
        for l in range(1, n_asks + 1):
            ask_price = asks[l-1, 0]
            total_ask_qty = cum_supply[l-1]
            
            if bid_price >= ask_price:  # Trade possible
                traded_volume = min(total_bid_qty, total_ask_qty)
                if traded_volume > best_volume:
                    best_volume = traded_volume
                    best_k = k
                    best_l = l
                    clearing_price = (bid_price + ask_price) / 2
    
    if best_volume <= 0:
        return None, np.zeros(n_bids), np.zeros(n_asks)
    
    # Allocate quantities
    buy_quantities = np.zeros(n_bids)
    sell_quantities = np.zeros(n_asks)
    
    total_bid_qty = cum_demand[best_k - 1]
    total_ask_qty = cum_supply[best_l - 1]
    traded_volume = min(total_bid_qty, total_ask_qty)
    
    if total_bid_qty >= total_ask_qty:
        # Sellers are fully filled, buyers are rationed
        for j in range(best_l):
            sell_quantities[j] = asks[j, 1]
        
        # Ration buyers proportionally
        remaining = traded_volume
        for i in range(best_k):
            allocation = min(bids[i, 1], remaining)
            buy_quantities[i] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
    else:
        # Buyers are fully filled, sellers are rationed
        for i in range(best_k):
            buy_quantities[i] = bids[i, 1]
        
        # Ration sellers proportionally
        remaining = traded_volume
        for j in range(best_l):
            allocation = min(asks[j, 1], remaining)
            sell_quantities[j] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
    
    return clearing_price, buy_quantities, sell_quantities


def run_sda_auction(bids_matrix: np.ndarray, asks_matrix: np.ndarray, 
                    t: int, results: MechanismResults) -> Tuple[np.ndarray, np.ndarray, bool]:
    if bids_matrix.shape[0] == 0 or asks_matrix.shape[0] == 0:
        return bids_matrix, asks_matrix, False
    
    # Sort bids descending by price, asks ascending
    bids_sorted = bids_matrix[np.argsort(-bids_matrix[:, 0])]
    asks_sorted = asks_matrix[np.argsort(asks_matrix[:, 0])]
    
    # Find clearing
    clearing_price, buy_quantities, sell_quantities = compute_clearing_price_and_quantities(bids_sorted, asks_sorted)
    
    if clearing_price is None:
        return bids_matrix, asks_matrix, False
    
    # Record trades
    total_trade_volume = np.sum(buy_quantities)
    if total_trade_volume > 1e-10:
        # Calculate buyer/seller surplus for the trade record
        avg_buyer_value = np.sum(bids_sorted[:, 0] * buy_quantities) / total_trade_volume if total_trade_volume > 0 else 0
        avg_seller_cost = np.sum(asks_sorted[:, 0] * sell_quantities) / total_trade_volume if total_trade_volume > 0 else 0
        
        results.trades.append(TradeRecord(t, clearing_price, total_trade_volume, avg_buyer_value, avg_seller_cost))
        results.prices.append(clearing_price)
        results.volumes.append(total_trade_volume)
        results.times.append(t)
        results.total_volume += total_trade_volume
        results.total_value += clearing_price * total_trade_volume
    
    # Update order books
    remaining_bids = []
    for i in range(bids_sorted.shape[0]):
        remaining_qty = bids_sorted[i, 1] - buy_quantities[i]
        if remaining_qty > 1e-10:
            remaining_bids.append([bids_sorted[i, 0], remaining_qty])
    
    remaining_asks = []
    for j in range(asks_sorted.shape[0]):
        remaining_qty = asks_sorted[j, 1] - sell_quantities[j]
        if remaining_qty > 1e-10:
            remaining_asks.append([asks_sorted[j, 0], remaining_qty])
    
    new_bids = np.array(remaining_bids) if remaining_bids else np.zeros((0, 2))
    new_asks = np.array(remaining_asks) if remaining_asks else np.zeros((0, 2))
    
    return new_bids, new_asks, True


def run_sda_simulation(all_bids: List, all_asks: List, params: SimulationParams) -> MechanismResults:
    """
    Run Sequential Double Auction simulation.
    Auctions are triggered based on time and order book depth conditions.
    """
    results = MechanismResults()
    
    # Order books as matrices [price, quantity]
    bids_matrix = np.zeros((0, 2))
    asks_matrix = np.zeros((0, 2))
    
    t_last = 0  # Time of last auction
    
    # Group orders by timestep
    bids_by_time = defaultdict(list)
    asks_by_time = defaultdict(list)
    
    for t, p, q in all_bids:
        bids_by_time[t].append((p, q))
    for t, p, q in all_asks:
        asks_by_time[t].append((p, q))
    
    for t in range(1, params.T + 1):
        # Add new orders to books
        for price, qty in bids_by_time[t]:
            bids_matrix = np.vstack([bids_matrix, [price, qty]]) if bids_matrix.size > 0 else np.array([[price, qty]])
        
        for price, qty in asks_by_time[t]:
            asks_matrix = np.vstack([asks_matrix, [price, qty]]) if asks_matrix.size > 0 else np.array([[price, qty]])
        
        # Check auction trigger conditions
        time_since_last = t - t_last
        n_bids = bids_matrix.shape[0]
        n_asks = asks_matrix.shape[0]
        
        trigger_auction = False
        
        # Condition 1: Maximum time exceeded
        if time_since_last > params.t_max:
            trigger_auction = True
        
        # Condition 2: Threshold exceeded and minimum time passed
        if n_bids >= params.tau_bid and n_asks >= params.tau_ask and time_since_last >= params.t_min:
            trigger_auction = True
        
        # Run auction if triggered
        if trigger_auction and n_bids > 0 and n_asks > 0:
            bids_matrix, asks_matrix, traded = run_sda_auction(bids_matrix, asks_matrix, t, results)
            t_last = t
    
    # Convert final order books
    results.final_bid_book = bids_matrix.tolist() if bids_matrix.size > 0 else []
    results.final_ask_book = asks_matrix.tolist() if asks_matrix.size > 0 else []
    
    return results


def run_pcm_simulation(all_bids: List, all_asks: List, params: SimulationParams, interval: int = 5) -> MechanismResults:
    results = MechanismResults()
    
    # Order books as matrices [price, quantity]
    bids_matrix = np.zeros((0, 2))
    asks_matrix = np.zeros((0, 2))
    
    # Group orders by timestep
    bids_by_time = defaultdict(list)
    asks_by_time = defaultdict(list)
    
    for t, p, q in all_bids:
        bids_by_time[t].append((p, q))
    for t, p, q in all_asks:
        asks_by_time[t].append((p, q))
    
    for t in range(1, params.T + 1):
        # Add new orders to books
        for price, qty in bids_by_time[t]:
            bids_matrix = np.vstack([bids_matrix, [price, qty]]) if bids_matrix.size > 0 else np.array([[price, qty]])
        
        for price, qty in asks_by_time[t]:
            asks_matrix = np.vstack([asks_matrix, [price, qty]]) if asks_matrix.size > 0 else np.array([[price, qty]])
        
        # Trigger auction at fixed intervals
        if t % interval == 0:
            n_bids = bids_matrix.shape[0]
            n_asks = asks_matrix.shape[0]
            
            if n_bids > 0 and n_asks > 0:
                bids_matrix, asks_matrix, traded = run_sda_auction(bids_matrix, asks_matrix, t, results)
    
    # Convert final order books
    results.final_bid_book = bids_matrix.tolist() if bids_matrix.size > 0 else []
    results.final_ask_book = asks_matrix.tolist() if asks_matrix.size > 0 else []
    
    return results


# =============================================================================
# ANALYSIS AND METRICS
# =============================================================================

def compute_price_discovery_metrics(results: MechanismResults) -> dict:
    """Compute price discovery metrics: volatility, spread, convergence."""
    if len(results.prices) < 2:
        return {'volatility': np.nan, 'mean_price': np.nan, 'price_range': np.nan}
    
    prices = np.array(results.prices)
    
    return {
        'volatility': np.std(prices),
        'mean_price': np.mean(prices),
        'price_range': np.max(prices) - np.min(prices)
    }


def compute_efficiency(results: MechanismResults, all_bids: List, all_asks: List) -> float:
    """Compute allocative efficiency: ratio of realized gains from trade to maximum possible."""
    # Make copies of quantities
    bid_data = [(b[1], b[2]) for b in all_bids]  # (price, qty)
    ask_data = [(a[1], a[2]) for a in all_asks]
    
    # Sort bids descending, asks ascending
    bid_data.sort(key=lambda x: -x[0])
    ask_data.sort(key=lambda x: x[0])
    
    # Calculate maximum possible surplus
    max_surplus = 0.0
    bid_idx = 0
    ask_idx = 0
    bid_qtys = [b[1] for b in bid_data]
    ask_qtys = [a[1] for a in ask_data]
    
    while bid_idx < len(bid_data) and ask_idx < len(ask_data):
        bp = bid_data[bid_idx][0]
        ap = ask_data[ask_idx][0]
        
        if bp >= ap:
            bq = bid_qtys[bid_idx]
            aq = ask_qtys[ask_idx]
            trade_qty = min(bq, aq)
            max_surplus += (bp - ap) * trade_qty
            
            bid_qtys[bid_idx] -= trade_qty
            ask_qtys[ask_idx] -= trade_qty
            
            if bid_qtys[bid_idx] <= 1e-10:
                bid_idx += 1
            if ask_qtys[ask_idx] <= 1e-10:
                ask_idx += 1
        else:
            break
    
    # Realized surplus
    realized_surplus = sum((t.buyer_value - t.seller_cost) * t.quantity for t in results.trades)
    
    return realized_surplus / max_surplus if max_surplus > 0 else 0.0



# MAIN SIMULATION

def run_simulation(params: SimulationParams = None) -> SimulationResults:
    """Run complete simulation comparing CDA, AMM, and SDA mechanisms."""
    if params is None:
        params = SimulationParams()
    
    print("=" * 60)
    print("MARKET MECHANISM COMPARISON SIMULATION")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  T = {params.T} time steps")
    print(f"  λ_bids = {params.lambda_bids}, λ_asks = {params.lambda_asks}")
    print(f"  Bid prices: [{params.p_bid}, {params.P_bid}]")
    print(f"  Ask prices: [{params.p_ask}, {params.P_ask}]")
    print(f"  Bid quantities: [{params.q_bid}, {params.Q_bid}]")
    print(f"  Ask quantities: [{params.q_ask}, {params.Q_ask}]")
    print(f"  CDA α = {params.alpha}")
    print(f"  AMM k = {params.k}, initial supply = {params.initial_offset_supply}")
    print(f"  SDA: t_max={params.t_max}, t_min={params.t_min}, τ_bid={params.tau_bid}, τ_ask={params.tau_ask}")
    
    # Generate orders
    print("\nGenerating orders...")
    all_bids, all_asks = generate_orders(params)
    print(f"  Generated {len(all_bids)} bids and {len(all_asks)} asks")
    
    # Run simulations
    print("\nRunning CDA simulation...")
    cda_results = run_cda_simulation(all_bids, all_asks, params)
    
    print("Running AMM simulation...")
    amm_results = run_amm_simulation(all_bids, all_asks, params)
    
    print("Running SDA simulation...")
    sda_results = run_sda_simulation(all_bids, all_asks, params)
    
    return SimulationResults(params, cda_results, amm_results, sda_results, all_bids, all_asks)


def print_results(sim_results: SimulationResults):
    """Print comprehensive comparison of simulation results."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    for name, results in [("CDA", sim_results.cda), 
                          ("AMM", sim_results.amm), 
                          ("SDA", sim_results.sda)]:
        print(f"\n--- {name} Results ---")
        print(f"  Number of trades: {len(results.trades)}")
        print(f"  Total volume traded: {results.total_volume:.2f}")
        print(f"  Total value traded: {results.total_value:.2f}")
        
        if len(results.prices) > 0:
            metrics = compute_price_discovery_metrics(results)
            print(f"  Mean price: {metrics['mean_price']:.4f}")
            print(f"  Price volatility: {metrics['volatility']:.4f}")
            print(f"  Price range: {metrics['price_range']:.4f}")
        
        efficiency = compute_efficiency(results, sim_results.all_bids, sim_results.all_asks)
        print(f"  Allocative efficiency: {efficiency:.4f}")
        
        print(f"  Remaining bids in book: {len(results.final_bid_book)}")
        print(f"  Remaining asks in book: {len(results.final_ask_book)}")
    
    # Comparative summary
    print("\n" + "-" * 60)
    print("COMPARATIVE SUMMARY")
    print("-" * 60)
    
    volumes = [sim_results.cda.total_volume, sim_results.amm.total_volume, sim_results.sda.total_volume]
    names = ["CDA", "AMM", "SDA"]
    
    print("\nVolume Ranking:")
    for i, idx in enumerate(np.argsort(volumes)[::-1]):
        print(f"  {i+1}. {names[idx]}: {volumes[idx]:.2f}")
    
    trade_counts = [len(sim_results.cda.trades), len(sim_results.amm.trades), len(sim_results.sda.trades)]
    print("\nTrade Count Ranking:")
    for i, idx in enumerate(np.argsort(trade_counts)[::-1]):
        print(f"  {i+1}. {names[idx]}: {trade_counts[idx]}")
    
    efficiencies = [
        compute_efficiency(sim_results.cda, sim_results.all_bids, sim_results.all_asks),
        compute_efficiency(sim_results.amm, sim_results.all_bids, sim_results.all_asks),
        compute_efficiency(sim_results.sda, sim_results.all_bids, sim_results.all_asks)
    ]
    print("\nEfficiency Ranking:")
    for i, idx in enumerate(np.argsort(efficiencies)[::-1]):
        print(f"  {i+1}. {names[idx]}: {efficiencies[idx]:.4f}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_simulation():
    """Run an example simulation with default parameters."""
    params = SimulationParams(
        T=500,
        lambda_bids=3.0,
        lambda_asks=3.0,
        p_bid=9.0,
        P_bid=11.0,
        p_ask=9.0,
        P_ask=11.0,
        q_bid=1.0,
        Q_bid=5.0,
        q_ask=1.0,
        Q_ask=5.0,
        alpha=0.5,
        k=10000.0,
        initial_offset_supply=100.0,
        t_max=10,
        t_min=3,
        tau_bid=8,
        tau_ask=8,
        seed=12345
    )
    
    results = run_simulation(params)
    print_results(results)
    
    return results


def plot_results(sim_results: SimulationResults, save_path: str = None):
    """Generate plots comparing the three mechanisms."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    mechanisms = [
        ("CDA (Continuous Double Auction)", sim_results.cda, '#2ecc71', False),
        ("AMM (Automated Market Maker)", sim_results.amm, '#e74c3c', False),
        ("SDA (Sequential Double Auction)", sim_results.sda, '#3498db', True)  # True = show separate buy/sell prices
    ]
    
    # Find global max price for consistent y-axis
    all_prices = []
    for _, results, _, _ in mechanisms:
        if len(results.trades) > 0:
            all_prices.extend([t.buyer_value for t in results.trades])
            all_prices.extend([t.seller_cost for t in results.trades])
    y_max = max(all_prices) * 1.1 if all_prices else 15
    
    for ax, (name, results, color, separate_prices) in zip(axes, mechanisms):
        if len(results.trades) > 0:
            times = [t.time for t in results.trades]
            
            # Extract prices
            buy_prices = [t.buyer_value for t in results.trades]   # Price buyers pay
            sell_prices = [t.seller_cost for t in results.trades]  # Price sellers receive
            
            if separate_prices:
                # For SDA: buyers and sellers get different prices
                ax.plot(times, buy_prices, color='blue', linewidth=1.5, label='Buyer Price (pays)', marker='', alpha=0.8)
                ax.plot(times, sell_prices, color='red', linewidth=1.5, label='Seller Price (receives)', marker='', alpha=0.8)
                
                # Show the spread as a shaded region
                ax.fill_between(times, sell_prices, buy_prices, color='gray', alpha=0.2, label='Spread')
                
                avg_spread = np.mean([bp - sp for bp, sp in zip(buy_prices, sell_prices)])
                stats_text = f'Auctions: {len(results.trades)} | Vol: {results.total_volume:.0f} | Avg Spread: {avg_spread:.3f}'
            else:
                # For CDA/AMM: single trade price
                trade_prices = [(t.buyer_value + t.seller_cost) / 2 for t in results.trades]
                
                # Show the original bid/ask that matched as lighter lines
                ax.plot(times, buy_prices, color='blue', linewidth=1, label='Buyer Valuation', alpha=0.4)
                ax.plot(times, sell_prices, color='red', linewidth=1, label='Seller Cost', alpha=0.4)
                ax.plot(times, trade_prices, color=color, linewidth=2, label='Trade Price', alpha=0.9)
                
                stats_text = f'Trades: {len(results.trades)} | Vol: {results.total_volume:.0f} | Avg Price: {np.mean(trade_prices):.2f} | σ: {np.std(trade_prices):.3f}'
            
            ax.set_ylim(0, y_max)
            ax.set_ylabel('Price', fontsize=11)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.set_ylim(0, y_max)
            ax.text(0.5, 0.5, 'No trades executed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    
    axes[-1].set_xlabel('Time Step', fontsize=11)
    
    plt.suptitle('Market Mechanism Comparison: Trade Prices Over Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return fig


def plot_price_distribution(sim_results: SimulationResults, save_path: str = None):
    """Generate price distribution histograms for each mechanism."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    mechanisms = [
        ("CDA", sim_results.cda, '#2ecc71', False),
        ("AMM", sim_results.amm, '#e74c3c', False),
        ("SDA", sim_results.sda, '#3498db', True)
    ]
    
    # Find global price range for consistent bins
    all_prices = []
    for _, results, _, _ in mechanisms:
        if len(results.trades) > 0:
            all_prices.extend([t.buyer_value for t in results.trades])
            all_prices.extend([t.seller_cost for t in results.trades])
    
    if all_prices:
        price_min, price_max = min(all_prices), max(all_prices)
        bins = np.linspace(price_min - 0.1, price_max + 0.1, 30)
    else:
        bins = 30
    
    for ax, (name, results, color, separate_prices) in zip(axes, mechanisms):
        if len(results.trades) > 0:
            if separate_prices:
                # For SDA: show both buyer and seller price distributions
                buy_prices = [t.buyer_value for t in results.trades]
                sell_prices = [t.seller_cost for t in results.trades]
                
                ax.hist(buy_prices, bins=bins, color='blue', alpha=0.5, edgecolor='black', linewidth=0.5, label=f'Buyer Price (μ={np.mean(buy_prices):.3f})')
                ax.hist(sell_prices, bins=bins, color='red', alpha=0.5, edgecolor='black', linewidth=0.5, label=f'Seller Price (μ={np.mean(sell_prices):.3f})')
                ax.axvline(np.mean(buy_prices), color='blue', linestyle='--', linewidth=2)
                ax.axvline(np.mean(sell_prices), color='red', linestyle='--', linewidth=2)
                
                avg_spread = np.mean([bp - sp for bp, sp in zip(buy_prices, sell_prices)])
                ax.set_title(f'{name}\nAvg Spread: {avg_spread:.4f}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=8)
            else:
                # For CDA/AMM: show single trade price distribution
                trade_prices = results.prices
                ax.hist(trade_prices, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(np.mean(trade_prices), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trade_prices):.3f}')
                ax.set_title(f'{name}\nσ = {np.std(trade_prices):.4f}', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
            
            ax.set_xlabel('Trade Price', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No trades', transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Price Distribution by Mechanism', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return fig


def plot_cumulative_volume(sim_results: SimulationResults, save_path: str = None):
    """Plot cumulative trading volume over time for each mechanism."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mechanisms = [
        ("CDA", sim_results.cda, '#2ecc71'),
        ("AMM", sim_results.amm, '#e74c3c'),
        ("SDA", sim_results.sda, '#3498db')
    ]
    
    for name, results, color in mechanisms:
        if len(results.trades) > 0:
            times = [t.time for t in results.trades]
            volumes = [t.quantity for t in results.trades]
            cum_volumes = np.cumsum(volumes)
            
            ax.plot(times, cum_volumes, color=color, linewidth=2, label=f'{name} (Total: {cum_volumes[-1]:.0f})')
            ax.scatter(times, cum_volumes, color=color, s=10, alpha=0.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Cumulative Volume', fontsize=11)
    ax.set_title('Cumulative Trading Volume Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return fig


def generate_all_plots(sim_results: SimulationResults, output_dir: str = "."):
    """Generate all plots and save them."""
    
    plot_results(sim_results, os.path.join(output_dir, "price_timeseries.png"))
    plot_price_distribution(sim_results, os.path.join(output_dir, "price_distribution.png"))
    plot_cumulative_volume(sim_results, os.path.join(output_dir, "cumulative_volume.png"))
    
    print(f"\nAll plots saved to {output_dir}/")


def plot_efficiency_vs_lambdas(lambda_range=(3, 10), n_points=8, T=100, pcm_interval=5, seed=42, save_dir=None):
    """
    Plot efficiency of each mechanism as a function of lambda_bids and lambda_asks.
    Creates two plots:
    1. Efficiency vs lambda_bids (with lambda_asks fixed at midpoint)
    2. Efficiency vs lambda_asks (with lambda_bids fixed at midpoint)
    
    Parameters:
    - lambda_range: tuple of (min, max) for lambda values
    - n_points: number of points to sample
    - T: number of timesteps per simulation
    - pcm_interval: interval for Periodic Call Market auctions
    - seed: base random seed
    - save_dir: directory to save the plots
    """
    import matplotlib.pyplot as plt
    
    lambda_values = np.linspace(lambda_range[0], lambda_range[1], n_points)
    lambda_mid = (lambda_range[0] + lambda_range[1]) / 2
    
    # Initialize efficiency arrays
    cda_eff_vs_bids = np.zeros(n_points)
    amm_eff_vs_bids = np.zeros(n_points)
    sda_eff_vs_bids = np.zeros(n_points)
    pcm_eff_vs_bids = np.zeros(n_points)
    
    cda_eff_vs_asks = np.zeros(n_points)
    amm_eff_vs_asks = np.zeros(n_points)
    sda_eff_vs_asks = np.zeros(n_points)
    pcm_eff_vs_asks = np.zeros(n_points)
    
    # Initialize volume arrays
    cda_vol_vs_bids = np.zeros(n_points)
    amm_vol_vs_bids = np.zeros(n_points)
    sda_vol_vs_bids = np.zeros(n_points)
    pcm_vol_vs_bids = np.zeros(n_points)
    
    cda_vol_vs_asks = np.zeros(n_points)
    amm_vol_vs_asks = np.zeros(n_points)
    sda_vol_vs_asks = np.zeros(n_points)
    pcm_vol_vs_asks = np.zeros(n_points)
    
    print(f"Running {2 * n_points} simulations...")
    print(f"Lambda range: {lambda_range[0]} to {lambda_range[1]}, {n_points} points")
    print(f"T = {T} timesteps per simulation")
    print(f"PCM interval = {pcm_interval}\n")
    
    # Vary lambda_bids, fix lambda_asks at midpoint
    print("Part 1: Varying λ_bids (fixing λ_asks = {:.1f})".format(lambda_mid))
    for i, lambda_bids in enumerate(lambda_values):
        params = SimulationParams(
            T=T,
            lambda_bids=lambda_bids,
            lambda_asks=lambda_mid,
            p_bid=9.0, P_bid=11.0,
            p_ask=9.0, P_ask=11.0,
            q_bid=1.0, Q_bid=5.0,
            q_ask=1.0, Q_ask=5.0,
            alpha=0.5,
            k=10000.0,
            initial_offset_supply=100.0,
            t_max=10, t_min=3,
            tau_bid=8, tau_ask=8,
            seed=seed + i
        )
        
        all_bids, all_asks = generate_orders(params)
        
        np.random.seed(params.seed)
        random.seed(params.seed)
        cda_results = run_cda_simulation(all_bids, all_asks, params)
        
        np.random.seed(params.seed)
        random.seed(params.seed)
        amm_results = run_amm_simulation(all_bids, all_asks, params)
        
        sda_results = run_sda_simulation(all_bids, all_asks, params)
        pcm_results = run_pcm_simulation(all_bids, all_asks, params, interval=pcm_interval)
        
        cda_eff_vs_bids[i] = compute_efficiency(cda_results, all_bids, all_asks)
        amm_eff_vs_bids[i] = compute_efficiency(amm_results, all_bids, all_asks)
        sda_eff_vs_bids[i] = compute_efficiency(sda_results, all_bids, all_asks)
        pcm_eff_vs_bids[i] = compute_efficiency(pcm_results, all_bids, all_asks)
        
        cda_vol_vs_bids[i] = cda_results.total_volume
        amm_vol_vs_bids[i] = amm_results.total_volume
        sda_vol_vs_bids[i] = sda_results.total_volume
        pcm_vol_vs_bids[i] = pcm_results.total_volume
        
        print(f"  λ_bids={lambda_bids:.1f}: CDA={cda_eff_vs_bids[i]:.3f}, AMM={amm_eff_vs_bids[i]:.3f}, SDA={sda_eff_vs_bids[i]:.3f}, PCM={pcm_eff_vs_bids[i]:.3f}")
    
    # Vary lambda_asks, fix lambda_bids at midpoint
    print("\nPart 2: Varying λ_asks (fixing λ_bids = {:.1f})".format(lambda_mid))
    for j, lambda_asks in enumerate(lambda_values):
        params = SimulationParams(
            T=T,
            lambda_bids=lambda_mid,
            lambda_asks=lambda_asks,
            p_bid=9.0, P_bid=11.0,
            p_ask=9.0, P_ask=11.0,
            q_bid=1.0, Q_bid=5.0,
            q_ask=1.0, Q_ask=5.0,
            alpha=0.5,
            k=10000.0,
            initial_offset_supply=100.0,
            t_max=10, t_min=3,
            tau_bid=8, tau_ask=8,
            seed=seed + n_points + j
        )
        
        all_bids, all_asks = generate_orders(params)
        
        np.random.seed(params.seed)
        random.seed(params.seed)
        cda_results = run_cda_simulation(all_bids, all_asks, params)
        
        np.random.seed(params.seed)
        random.seed(params.seed)
        amm_results = run_amm_simulation(all_bids, all_asks, params)
        
        sda_results = run_sda_simulation(all_bids, all_asks, params)
        pcm_results = run_pcm_simulation(all_bids, all_asks, params, interval=pcm_interval)
        
        cda_eff_vs_asks[j] = compute_efficiency(cda_results, all_bids, all_asks)
        amm_eff_vs_asks[j] = compute_efficiency(amm_results, all_bids, all_asks)
        sda_eff_vs_asks[j] = compute_efficiency(sda_results, all_bids, all_asks)
        pcm_eff_vs_asks[j] = compute_efficiency(pcm_results, all_bids, all_asks)
        
        cda_vol_vs_asks[j] = cda_results.total_volume
        amm_vol_vs_asks[j] = amm_results.total_volume
        sda_vol_vs_asks[j] = sda_results.total_volume
        pcm_vol_vs_asks[j] = pcm_results.total_volume
        
        print(f"  λ_asks={lambda_asks:.1f}: CDA={cda_eff_vs_asks[j]:.3f}, AMM={amm_eff_vs_asks[j]:.3f}, SDA={sda_eff_vs_asks[j]:.3f}, PCM={pcm_eff_vs_asks[j]:.3f}")
    
    print("\nGenerating plots...")
    
    # Plot 1: Efficiency vs lambda_bids
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(lambda_values, cda_eff_vs_bids, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='CDA')
    ax1.plot(lambda_values, amm_eff_vs_bids, 's-', color='#e74c3c', linewidth=2, markersize=8, label='AMM')
    ax1.plot(lambda_values, sda_eff_vs_bids, '^-', color='#3498db', linewidth=2, markersize=8, label='SDA')
    ax1.plot(lambda_values, pcm_eff_vs_bids, 'd-', color='#9b59b6', linewidth=2, markersize=8, label=f'PCM (interval={pcm_interval})')
    
    ax1.set_xlabel('λ_bids (Bid Arrival Rate)', fontsize=12)
    ax1.set_ylabel('Allocative Efficiency', fontsize=12)
    ax1.set_title(f'Efficiency vs. Bid Arrival Rate (λ_asks = {lambda_mid:.1f}, T = {T})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlim(lambda_range[0], lambda_range[1])
    
    plt.tight_layout()
    
    if save_dir:
        path1 = f"{save_dir}/efficiency_vs_lambda_bids.png"
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path1}")
    
    plt.close(fig1)
    
    # Plot 2: Efficiency vs lambda_asks
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(lambda_values, cda_eff_vs_asks, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='CDA')
    ax2.plot(lambda_values, amm_eff_vs_asks, 's-', color='#e74c3c', linewidth=2, markersize=8, label='AMM')
    ax2.plot(lambda_values, sda_eff_vs_asks, '^-', color='#3498db', linewidth=2, markersize=8, label='SDA')
    ax2.plot(lambda_values, pcm_eff_vs_asks, 'd-', color='#9b59b6', linewidth=2, markersize=8, label=f'PCM (interval={pcm_interval})')
    
    ax2.set_xlabel('λ_asks (Ask Arrival Rate)', fontsize=12)
    ax2.set_ylabel('Allocative Efficiency', fontsize=12)
    ax2.set_title(f'Efficiency vs. Ask Arrival Rate (λ_bids = {lambda_mid:.1f}, T = {T})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(lambda_range[0], lambda_range[1])
    
    plt.tight_layout()
    
    if save_dir:
        path2 = f"{save_dir}/efficiency_vs_lambda_asks.png"
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path2}")
    
    plt.close(fig2)
    
    # Plot 3: Volume vs lambda_bids
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ax3.plot(lambda_values, cda_vol_vs_bids, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='CDA')
    ax3.plot(lambda_values, amm_vol_vs_bids, 's-', color='#e74c3c', linewidth=2, markersize=8, label='AMM')
    ax3.plot(lambda_values, sda_vol_vs_bids, '^-', color='#3498db', linewidth=2, markersize=8, label='SDA')
    ax3.plot(lambda_values, pcm_vol_vs_bids, 'd-', color='#9b59b6', linewidth=2, markersize=8, label=f'PCM (interval={pcm_interval})')
    
    ax3.set_xlabel('λ_bids (Bid Arrival Rate)', fontsize=12)
    ax3.set_ylabel('Total Volume Traded', fontsize=12)
    ax3.set_title(f'Volume vs. Bid Arrival Rate (λ_asks = {lambda_mid:.1f}, T = {T})', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, None)
    ax3.set_xlim(lambda_range[0], lambda_range[1])
    
    plt.tight_layout()
    
    if save_dir:
        path3 = f"{save_dir}/volume_vs_lambda_bids.png"
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path3}")
    
    plt.close(fig3)
    
    # Plot 4: Volume vs lambda_asks
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    ax4.plot(lambda_values, cda_vol_vs_asks, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='CDA')
    ax4.plot(lambda_values, amm_vol_vs_asks, 's-', color='#e74c3c', linewidth=2, markersize=8, label='AMM')
    ax4.plot(lambda_values, sda_vol_vs_asks, '^-', color='#3498db', linewidth=2, markersize=8, label='SDA')
    ax4.plot(lambda_values, pcm_vol_vs_asks, 'd-', color='#9b59b6', linewidth=2, markersize=8, label=f'PCM (interval={pcm_interval})')
    
    ax4.set_xlabel('λ_asks (Ask Arrival Rate)', fontsize=12)
    ax4.set_ylabel('Total Volume Traded', fontsize=12)
    ax4.set_title(f'Volume vs. Ask Arrival Rate (λ_bids = {lambda_mid:.1f}, T = {T})', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, None)
    ax4.set_xlim(lambda_range[0], lambda_range[1])
    
    plt.tight_layout()
    
    if save_dir:
        path4 = f"{save_dir}/volume_vs_lambda_asks.png"
        fig4.savefig(path4, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path4}")
    
    plt.close(fig4)
    
    return {
        'lambda_values': lambda_values,
        'cda_vs_bids': cda_eff_vs_bids,
        'amm_vs_bids': amm_eff_vs_bids,
        'sda_vs_bids': sda_eff_vs_bids,
        'pcm_vs_bids': pcm_eff_vs_bids,
        'cda_vs_asks': cda_eff_vs_asks,
        'amm_vs_asks': amm_eff_vs_asks,
        'sda_vs_asks': sda_eff_vs_asks,
        'pcm_vs_asks': pcm_eff_vs_asks,
        'cda_vol_vs_bids': cda_vol_vs_bids,
        'amm_vol_vs_bids': amm_vol_vs_bids,
        'sda_vol_vs_bids': sda_vol_vs_bids,
        'pcm_vol_vs_bids': pcm_vol_vs_bids,
        'cda_vol_vs_asks': cda_vol_vs_asks,
        'amm_vol_vs_asks': amm_vol_vs_asks,
        'sda_vol_vs_asks': sda_vol_vs_asks,
        'pcm_vol_vs_asks': pcm_vol_vs_asks,
    }


def plot_sda_volume_vs_tmax(tmax_range=(1, 10), n_simulations=20, T=100, seed=42, save_dir=None):
    """
    Plot SDA trading volume as a function of t_max with confidence intervals.
    
    Parameters:
    - tmax_range: tuple of (min, max) for t_max values
    - n_simulations: number of simulations per t_max value for confidence intervals
    - T: number of timesteps per simulation
    - seed: base random seed
    - save_dir: directory to save the plots
    """
    import matplotlib.pyplot as plt
    
    tmax_values = list(range(tmax_range[0], tmax_range[1] + 1))
    n_tmax = len(tmax_values)
    
    # Store results for each t_max
    sda_volumes = np.zeros((n_tmax, n_simulations))
    sda_efficiencies = np.zeros((n_tmax, n_simulations))
    sda_num_auctions = np.zeros((n_tmax, n_simulations))
    
    print(f"Running {n_tmax * n_simulations} simulations...")
    print(f"t_max range: {tmax_range[0]} to {tmax_range[1]}")
    print(f"Simulations per t_max: {n_simulations}")
    print(f"T = {T} timesteps per simulation\n")
    
    for i, t_max in enumerate(tmax_values):
        print(f"t_max = {t_max}:", end=" ")
        
        for sim in range(n_simulations):
            params = SimulationParams(
                T=T,
                lambda_bids=6.5,
                lambda_asks=6.5,
                p_bid=9.0, P_bid=11.0,
                p_ask=9.0, P_ask=11.0,
                q_bid=1.0, Q_bid=5.0,
                q_ask=1.0, Q_ask=5.0,
                alpha=0.5,
                k=10000.0,
                initial_offset_supply=100.0,
                t_max=t_max,
                t_min=min(3, t_max),  # t_min can't exceed t_max
                tau_bid=8,
                tau_ask=8,
                seed=seed + i * n_simulations + sim
            )
            
            all_bids, all_asks = generate_orders(params)
            sda_results = run_sda_simulation(all_bids, all_asks, params)
            
            sda_volumes[i, sim] = sda_results.total_volume
            sda_efficiencies[i, sim] = compute_efficiency(sda_results, all_bids, all_asks)
            sda_num_auctions[i, sim] = len(sda_results.trades)
        
        print(f"Vol = {np.mean(sda_volumes[i]):.1f} ± {np.std(sda_volumes[i]):.1f}")
    
    # Calculate statistics
    vol_mean = np.mean(sda_volumes, axis=1)
    vol_std = np.std(sda_volumes, axis=1)
    vol_ci = 1.96 * vol_std / np.sqrt(n_simulations)  # 95% CI
    
    eff_mean = np.mean(sda_efficiencies, axis=1)
    eff_std = np.std(sda_efficiencies, axis=1)
    eff_ci = 1.96 * eff_std / np.sqrt(n_simulations)
    
    auctions_mean = np.mean(sda_num_auctions, axis=1)
    auctions_std = np.std(sda_num_auctions, axis=1)
    auctions_ci = 1.96 * auctions_std / np.sqrt(n_simulations)
    
    print("\nGenerating plots...")
    
    # Plot 1: Volume vs t_max
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(tmax_values, vol_mean, 'o-', color='#3498db', linewidth=2, markersize=8, label='SDA Volume')
    ax1.fill_between(tmax_values, vol_mean - vol_ci, vol_mean + vol_ci, color='#3498db', alpha=0.2, label='95% CI')
    
    ax1.set_xlabel('t_max (Maximum Time Between Auctions)', fontsize=12)
    ax1.set_ylabel('Total Volume Traded', fontsize=12)
    ax1.set_title(f'SDA Volume vs. t_max (T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    ax1.set_xlim(tmax_range[0], tmax_range[1])
    ax1.set_xticks(tmax_values)
    
    plt.tight_layout()
    
    if save_dir:
        path1 = f"{save_dir}/sda_volume_vs_tmax.png"
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path1}")
    
    plt.close(fig1)
    
    # Plot 2: Efficiency vs t_max
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(tmax_values, eff_mean, 's-', color='#2ecc71', linewidth=2, markersize=8, label='SDA Efficiency')
    ax2.fill_between(tmax_values, eff_mean - eff_ci, eff_mean + eff_ci, color='#2ecc71', alpha=0.2, label='95% CI')
    
    ax2.set_xlabel('t_max (Maximum Time Between Auctions)', fontsize=12)
    ax2.set_ylabel('Allocative Efficiency', fontsize=12)
    ax2.set_title(f'SDA Efficiency vs. t_max (T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(tmax_range[0], tmax_range[1])
    ax2.set_xticks(tmax_values)
    
    plt.tight_layout()
    
    if save_dir:
        path2 = f"{save_dir}/sda_efficiency_vs_tmax.png"
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path2}")
    
    plt.close(fig2)
    
    # Plot 3: Number of auctions vs t_max
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ax3.plot(tmax_values, auctions_mean, '^-', color='#9b59b6', linewidth=2, markersize=8, label='Number of Auctions')
    ax3.fill_between(tmax_values, auctions_mean - auctions_ci, auctions_mean + auctions_ci, color='#9b59b6', alpha=0.2, label='95% CI')
    
    ax3.set_xlabel('t_max (Maximum Time Between Auctions)', fontsize=12)
    ax3.set_ylabel('Number of Auctions', fontsize=12)
    ax3.set_title(f'SDA Auction Count vs. t_max (T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, None)
    ax3.set_xlim(tmax_range[0], tmax_range[1])
    ax3.set_xticks(tmax_values)
    
    plt.tight_layout()
    
    if save_dir:
        path3 = f"{save_dir}/sda_auctions_vs_tmax.png"
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path3}")
    
    plt.close(fig3)
    
    return {
        'tmax_values': tmax_values,
        'vol_mean': vol_mean,
        'vol_std': vol_std,
        'vol_ci': vol_ci,
        'eff_mean': eff_mean,
        'eff_std': eff_std,
        'eff_ci': eff_ci,
        'auctions_mean': auctions_mean,
        'auctions_std': auctions_std,
        'auctions_ci': auctions_ci,
    }


def plot_sda_metrics_vs_tmin(tmin_range=(1, 10), t_max=10, n_simulations=20, T=100, seed=42, save_dir=None):
    """
    Plot SDA efficiency, volatility, and volume as a function of t_min with confidence intervals.
    
    Parameters:
    - tmin_range: tuple of (min, max) for t_min values
    - t_max: fixed t_max value
    - n_simulations: number of simulations per t_min value for confidence intervals
    - T: number of timesteps per simulation
    - seed: base random seed
    - save_dir: directory to save the plots
    """
    import matplotlib.pyplot as plt
    
    tmin_values = list(range(tmin_range[0], tmin_range[1] + 1))
    n_tmin = len(tmin_values)
    
    # Store results for each t_min
    sda_volumes = np.zeros((n_tmin, n_simulations))
    sda_efficiencies = np.zeros((n_tmin, n_simulations))
    sda_volatilities = np.zeros((n_tmin, n_simulations))
    
    print(f"Running {n_tmin * n_simulations} simulations...")
    print(f"t_min range: {tmin_range[0]} to {tmin_range[1]}")
    print(f"t_max fixed at: {t_max}")
    print(f"Simulations per t_min: {n_simulations}")
    print(f"T = {T} timesteps per simulation\n")
    
    for i, t_min in enumerate(tmin_values):
        print(f"t_min = {t_min}:", end=" ")
        
        for sim in range(n_simulations):
            params = SimulationParams(
                T=T,
                lambda_bids=6.5,
                lambda_asks=6.5,
                p_bid=9.0, P_bid=11.0,
                p_ask=9.0, P_ask=11.0,
                q_bid=1.0, Q_bid=5.0,
                q_ask=1.0, Q_ask=5.0,
                alpha=0.5,
                k=10000.0,
                initial_offset_supply=100.0,
                t_max=t_max,
                t_min=t_min,
                tau_bid=8,
                tau_ask=8,
                seed=seed + i * n_simulations + sim
            )
            
            all_bids, all_asks = generate_orders(params)
            sda_results = run_sda_simulation(all_bids, all_asks, params)
            
            sda_volumes[i, sim] = sda_results.total_volume
            sda_efficiencies[i, sim] = compute_efficiency(sda_results, all_bids, all_asks)
            
            # Calculate volatility (std of prices)
            if len(sda_results.prices) > 1:
                sda_volatilities[i, sim] = np.std(sda_results.prices)
            else:
                sda_volatilities[i, sim] = np.nan
        
        vol_mean = np.mean(sda_volumes[i])
        eff_mean = np.mean(sda_efficiencies[i])
        volat_mean = np.nanmean(sda_volatilities[i])
        print(f"Vol={vol_mean:.1f}, Eff={eff_mean:.3f}, Volat={volat_mean:.4f}")
    
    # Calculate statistics
    vol_mean = np.mean(sda_volumes, axis=1)
    vol_std = np.std(sda_volumes, axis=1)
    vol_ci = 1.96 * vol_std / np.sqrt(n_simulations)
    
    eff_mean = np.mean(sda_efficiencies, axis=1)
    eff_std = np.std(sda_efficiencies, axis=1)
    eff_ci = 1.96 * eff_std / np.sqrt(n_simulations)
    
    volat_mean = np.nanmean(sda_volatilities, axis=1)
    volat_std = np.nanstd(sda_volatilities, axis=1)
    volat_ci = 1.96 * volat_std / np.sqrt(n_simulations)
    
    print("\nGenerating plots...")
    
    # Plot 1: Volume vs t_min
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(tmin_values, vol_mean, 'o-', color='#3498db', linewidth=2, markersize=8, label='SDA Volume')
    ax1.fill_between(tmin_values, vol_mean - vol_ci, vol_mean + vol_ci, color='#3498db', alpha=0.2, label='95% CI')
    
    ax1.set_xlabel('t_min (Minimum Time Between Auctions)', fontsize=12)
    ax1.set_ylabel('Total Volume Traded', fontsize=12)
    ax1.set_title(f'SDA Volume vs. t_min (t_max={t_max}, T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    ax1.set_xlim(tmin_range[0], tmin_range[1])
    ax1.set_xticks(tmin_values)
    
    plt.tight_layout()
    
    if save_dir:
        path1 = f"{save_dir}/sda_volume_vs_tmin.png"
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path1}")
    
    plt.close(fig1)
    
    # Plot 2: Efficiency vs t_min
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(tmin_values, eff_mean, 's-', color='#2ecc71', linewidth=2, markersize=8, label='SDA Efficiency')
    ax2.fill_between(tmin_values, eff_mean - eff_ci, eff_mean + eff_ci, color='#2ecc71', alpha=0.2, label='95% CI')
    
    ax2.set_xlabel('t_min (Minimum Time Between Auctions)', fontsize=12)
    ax2.set_ylabel('Allocative Efficiency', fontsize=12)
    ax2.set_title(f'SDA Efficiency vs. t_min (t_max={t_max}, T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(tmin_range[0], tmin_range[1])
    ax2.set_xticks(tmin_values)
    
    plt.tight_layout()
    
    if save_dir:
        path2 = f"{save_dir}/sda_efficiency_vs_tmin.png"
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path2}")
    
    plt.close(fig2)
    
    # Plot 3: Volatility vs t_min
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    ax3.plot(tmin_values, volat_mean, '^-', color='#e74c3c', linewidth=2, markersize=8, label='SDA Price Volatility')
    ax3.fill_between(tmin_values, volat_mean - volat_ci, volat_mean + volat_ci, color='#e74c3c', alpha=0.2, label='95% CI')
    
    ax3.set_xlabel('t_min (Minimum Time Between Auctions)', fontsize=12)
    ax3.set_ylabel('Price Volatility (Std Dev)', fontsize=12)
    ax3.set_title(f'SDA Price Volatility vs. t_min (t_max={t_max}, T={T}, n={n_simulations} simulations)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, None)
    ax3.set_xlim(tmin_range[0], tmin_range[1])
    ax3.set_xticks(tmin_values)
    
    plt.tight_layout()
    
    if save_dir:
        path3 = f"{save_dir}/sda_volatility_vs_tmin.png"
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path3}")
    
    plt.close(fig3)
    
    return {
        'tmin_values': tmin_values,
        'vol_mean': vol_mean,
        'vol_std': vol_std,
        'vol_ci': vol_ci,
        'eff_mean': eff_mean,
        'eff_std': eff_std,
        'eff_ci': eff_ci,
        'volat_mean': volat_mean,
        'volat_std': volat_std,
        'volat_ci': volat_ci,
    }


# In[ ]:




