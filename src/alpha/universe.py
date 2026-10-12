"""The stock universe: Nifty 100 constituents.

Caveat (survivorship bias): this is the *current* constituent list. Stocks that were in
the index in 2012 but later dropped out (or were delisted) are missing, and stocks that
joined recently are here because they did well. That flatters any backtest a little.
The fix is point-in-time index membership from NSE's archives; it is listed as future
work in the README.
"""

NIFTY_100 = [
    # Nifty 50
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO",
    "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    # Nifty Next 50
    "ABB", "ADANIENSOL", "ADANIGREEN", "ADANIPOWER", "AMBUJACEM", "BAJAJHLDNG",
    "BANKBARODA", "BOSCHLTD", "BPCL", "BRITANNIA", "CANBK", "CHOLAFIN", "DABUR",
    "DIVISLAB", "DLF", "DMART", "GAIL", "GODREJCP", "HAL", "HAVELLS", "HYUNDAI",
    "ICICIGI", "ICICIPRULI", "INDHOTEL", "INDIGO", "IOC", "IRFC", "JINDALSTEL",
    "JSWENERGY", "LICI", "LODHA", "LTIM", "MOTHERSON", "NAUKRI", "PFC", "PIDILITIND",
    "PNB", "RECLTD", "SHREECEM", "SIEMENS", "SWIGGY", "TATAPOWER", "TORNTPHARM",
    "TVSMOTOR", "UNITDSPR", "VBL", "VEDL", "ZYDUSLIFE",
]

# Symbols that changed on Yahoo after a rename or demerger (tried if the main one fails)
ALIASES = {
    "TATAMOTORS": ["TMPV"],  # passenger-vehicle business after the 2025 demerger
    "LTIM": ["LTM"],
}

# Index levels used as benchmarks (Yahoo symbol -> name)
INDICES = {"^CNX100": "NIFTY100", "^NSEI": "NIFTY50"}


def yahoo_symbol(nse_symbol: str) -> str:
    """NSE symbol -> Yahoo Finance symbol (e.g. 'TCS' -> 'TCS.NS')."""
    return f"{nse_symbol}.NS"
