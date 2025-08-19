import sys, math, json, datetime as dt
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# ------------ CONFIG ------------
SOURCE = "MID"  # "MID" for Elexon Market Index Data, or "AGILE" to infer wholesale from Octopus Agile
REGION_LETTER = "C"  # only used for AGILE (e.g., 'C' = London)
DAYS = 365
FLOORS_PENCE = [5, 8, 10]  # floors in p/kWh
ASSUME_VOLUME_MWH_PER_HH = 1.0  # assumed export volume per half-hour (MWh) for the calc

# Realism knobs:
RESALE_UPLIFT_P = 1.0          # p/kWh you expect to realise above day-ahead (flex/imbalance/trading edge)
STANDING_P_PER_DAY_PENCE = 60.0 # p/day per meter (membership/standing income)
HEDGE_RATIO = 0.5              # 0..1 share of volume effectively hedged >= floor; pays (floor - spot) on that share
# --------------------------------

tz_utc = dt.timezone.utc
end = dt.datetime.now(tz_utc).replace(minute=0, second=0, microsecond=0)
start = end - dt.timedelta(days=DAYS)

def fetch_mid(start, end):
    """
    Fetch Elexon Market Index Data (MID) half-hourly prices.
    Returns DataFrame with columns: ['timestamp' (UTC), 'p_per_kwh']
    """
    import logging
    logger = logging.getLogger(__name__)

    # chunk by <=7-day windows (Elexon MID requires short inclusive ranges)
    def window_chunks(s, e, days=7):
        cur_from = s
        while cur_from < e:
            cur_to = min(cur_from + dt.timedelta(days=days), e)
            yield cur_from, cur_to
            cur_from = cur_to + dt.timedelta(seconds=1)

    primary = "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
    frames = []

    for s, e in window_chunks(start, end):
        params = {"from": s.isoformat().replace("+00:00", "Z"), "to": e.isoformat().replace("+00:00", "Z")}
        try:
            r = requests.get(primary, params=params, timeout=60)
            if r.status_code >= 400:
                logger.warning("MID endpoint returned %s: %s", r.status_code, r.text)
                r.raise_for_status()
            js = r.json()
        except Exception as exc:
            logger.error("MID request failed: %s", exc)
            raise

        rows = js.get("data") if isinstance(js, dict) else js
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        # Normalise schema: prefer settlementDate + settlementPeriod, or timestamp
        if "settlementDate" in df.columns and "settlementPeriod" in df.columns:
            ts = pd.to_datetime(df["settlementDate"], utc=True) + pd.to_timedelta((df["settlementPeriod"] - 1) * 30, unit="m")
            df["timestamp"] = ts
        elif "timestamp" not in df.columns:
            logger.error("Unexpected MID schema for params=%s: columns=%s", params, list(df.columns))
            raise RuntimeError("Unexpected MID schema; please inspect JSON.")

        price_col = "marketIndexPrice" if "marketIndexPrice" in df.columns else ("price" if "price" in df.columns else None)
        if price_col is None:
            logger.error("MID response missing price column for params=%s; columns=%s", params, list(df.columns))
            raise RuntimeError("MID response missing price column.")

        # £/MWh → p/kWh (1 £/MWh = 0.1 p/kWh)
        df["p_per_kwh"] = df[price_col].astype(float) / 10.0
        frames.append(df[["timestamp", "p_per_kwh"]])

    if not frames:
        raise RuntimeError("No MID data returned for the requested period.")

    out = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp")
    out = out[(out["timestamp"] >= start) & (out["timestamp"] < end)].reset_index(drop=True)
    return out

def fetch_agile_wholesale(start, end, region_letter="C"):
    """
    Fetch Octopus Agile half-hourly import prices (inc VAT) and infer wholesale.
    Approximate reverse formula used widely by the community:
      price_inc_vat ≈ (wholesale * 2.2 + peak_adder) * 1.05
    where peak_adder = 12p between 16:00–19:00 Europe/London, else 0.
    Then wholesale ≈ ((price_inc_vat / 1.05) - peak_adder) / 2.2
    """
    prod = "AGILE-FLEX-22-11-25"
    base = f"https://api.octopus.energy/v1/products/{prod}/electricity-tariffs/E-1R-{prod}-{region_letter}/standard-unit-rates/"

    prices = []
    cur_to = end
    while cur_to > start:
        cur_from = max(start, cur_to - dt.timedelta(days=20))  # ~960 half-hours/page
        url = (f"{base}?period_from={cur_from.isoformat().replace('+00:00','Z')}"
               f"&period_to={cur_to.isoformat().replace('+00:00','Z')}&page_size=1500")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        js = r.json()
        results = js.get("results", [])
        for row in results:
            t = pd.to_datetime(row["valid_from"], utc=True)  # each covers 30 min starting at valid_from
            p_inc = float(row["value_inc_vat"])  # p/kWh inc VAT
            prices.append((t, p_inc))
        cur_to = cur_from

    if not prices:
        raise RuntimeError("No Agile price data returned for the requested period.")

    df = pd.DataFrame(prices, columns=["timestamp", "p_inc_vat"]).drop_duplicates("timestamp")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].sort_values("timestamp")

    # Determine peak adder (Europe/London local 16:00–19:00)
    df_local = df.copy()
    df_local["timestamp_local"] = df_local["timestamp"].dt.tz_convert("Europe/London")
    df_local["hour_local"] = df_local["timestamp_local"].dt.hour
    df_local["peak_adder"] = np.where(df_local["hour_local"].isin([16, 17, 18]), 12.0, 0.0)

    # Reverse approximate formula
    vat_multiplier = 1.05
    multiplier = 2.2
    df_local["p_per_kwh"] = ((df_local["p_inc_vat"] / vat_multiplier) - df_local["peak_adder"]) / multiplier
    df_local["p_per_kwh"] = df_local["p_per_kwh"].clip(lower=0)

    return df_local[["timestamp", "p_per_kwh"]].reset_index(drop=True)

def get_price_series(source, start, end, region_letter):
    if source.upper() == "MID":
        return fetch_mid(start, end)
    elif source.upper() == "AGILE":
        return fetch_agile_wholesale(start, end, region_letter=region_letter)
    else:
        raise ValueError("SOURCE must be 'MID' or 'AGILE'.")

def simulate_floors(prices_df, floors_pence, volume_mwh_per_hh=1.0):
    """
    prices_df: ['timestamp', 'p_per_kwh']
    Returns: (totals_df, cum_margins_dict, prices_df)
    Totals are in *pence-equivalent* given the volume scaling.
    """
    if prices_df.empty:
        raise RuntimeError("No prices to simulate.")
    prices_df = prices_df.sort_values("timestamp").reset_index(drop=True)

    # Standing income over the covered days (per meter)
    day_count = max(1, (prices_df["timestamp"].dt.normalize().nunique()))
    standing_income_total = STANDING_P_PER_DAY_PENCE * day_count  # pence

    results = {}
    cum_margins = {}

    spot = prices_df["p_per_kwh"].values.astype(float)
    resale = spot + RESALE_UPLIFT_P  # p/kWh resale assumption
    vol = float(volume_mwh_per_hh)

    for floor in floors_pence:
        # Cashflows per half-hour (in pence per assumed MWh volume unit)
        payout = np.maximum(spot, floor) * vol
        revenue = resale * vol
        base_margin = revenue - payout  # trading spread vs floor

        # Hedge pays when spot < floor on hedged share
        hedge_pay = np.maximum(0.0, floor - spot) * (HEDGE_RATIO * vol)

        # Add standing charge (flat per-meter income over the period)
        # Convert to per-series by adding once at the end; for cumsum, add pro-rata evenly
        per_step_standing = standing_income_total / len(prices_df)

        margin_series = base_margin + hedge_pay + per_step_standing
        cum_margins[floor] = np.cumsum(margin_series)

        results[floor] = {
            "Revenue (p)": float(revenue.sum()),
            "Payout (p)": float(payout.sum()),
            "Hedge Income (p)": float(hedge_pay.sum()),
            "Standing Income (p)": float(standing_income_total),
            "Margin Total (p)": float(margin_series.sum())
        }

    totals_df = pd.DataFrame(results).T[["Revenue (p)", "Payout (p)", "Hedge Income (p)", "Standing Income (p)", "Margin Total (p)"]]
    return totals_df, cum_margins, prices_df

def main():
    print(f"Fetching prices from {SOURCE} for {DAYS} days ending {end.isoformat()}")
    prices = get_price_series(SOURCE, start, end, REGION_LETTER)

    # Ensure continuous HH timeline (fill gaps to avoid cumsum jumps)
    full_index = pd.date_range(prices["timestamp"].min(), prices["timestamp"].max(), freq="30min", tz="UTC")
    prices = prices.set_index("timestamp").reindex(full_index)
    prices["p_per_kwh"] = prices["p_per_kwh"].astype(float).interpolate(limit_direction="both")
    prices = prices.reset_index().rename(columns={"index": "timestamp"})

    totals_df, cum_margins, prices_df = simulate_floors(prices, FLOORS_PENCE, ASSUME_VOLUME_MWH_PER_HH)

    # Print totals
    print("\n=== Floor Simulation Totals (pence-equivalent) ===")
    print(totals_df.round(2).to_string())

    # Plot cumulative margin curves
    plt.figure(figsize=(10, 6))
    for floor, cm in cum_margins.items():
        plt.plot(prices_df["timestamp"], cm, label=f"Floor {floor}p")
    plt.title(f"Cumulative Margin – Source: {SOURCE} ({DAYS} days)\n"
              f"uplift={RESALE_UPLIFT_P}p, hedge={HEDGE_RATIO*100:.0f}%, standing={STANDING_P_PER_DAY_PENCE}p/day",
              fontsize=12, fontweight="bold")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Margin (pence-equivalent)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
        sys.exit(1)
