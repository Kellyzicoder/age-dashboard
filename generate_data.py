"""Generate a realistic synthetic dataset: 1,000 churches and ~500k member records (vectorised, ~2s).

Outputs
  data/churches.csv.gz  - one row per church
  data/members.csv.gz   - one row per member (birth year, join year, leave year)

Run:  python generate_data.py      (writes the CSVs)
The dashboard imports generate() directly when the CSVs are missing, which skips the slow gzip step.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CHURCHES = 1000
START_YEAR, END_YEAR = 2015, 2026
OUT = Path(__file__).parent / "data"

REGIONS = ["Greater Accra", "Ashanti", "Western", "Eastern", "Northern", "Volta", "Central"]
REGION_W = [0.24, 0.20, 0.12, 0.13, 0.10, 0.10, 0.11]
DENOMS = ["Pentecostal", "Methodist", "Presbyterian", "Catholic", "Baptist", "Charismatic", "Anglican"]
DENOM_W = [0.24, 0.14, 0.13, 0.15, 0.08, 0.18, 0.08]
SETTINGS = ["Urban", "Peri-urban", "Rural"]

# Denomination "profile": how young new joiners skew, and churn
DENOM_YOUTH = {"Pentecostal": 0.9, "Charismatic": 1.0, "Baptist": 0.4, "Methodist": 0.1,
               "Presbyterian": 0.0, "Catholic": 0.2, "Anglican": -0.1}

adj = ["Grace", "Victory", "Living Water", "Bethel", "Emmanuel", "Zion", "Covenant", "Calvary",
       "Holy Trinity", "Christ the King", "New Life", "Faith", "Hope", "Resurrection", "Good Shepherd",
       "Redeemed", "Mount Olive", "Shiloh", "Ebenezer", "Harvest"]
kind = ["Chapel", "Assembly", "Tabernacle", "Parish", "Cathedral", "Temple", "Centre", "Church"]

def generate(seed: int = SEED):
    """Return (churches, members) DataFrames."""
    rng = np.random.default_rng(seed)
    churches = []
    for i in range(N_CHURCHES):
        region = rng.choice(REGIONS, p=REGION_W)
        denom = rng.choice(DENOMS, p=DENOM_W)
        setting = rng.choice(SETTINGS, p=[0.45, 0.25, 0.30])
        founded = int(rng.integers(1950, 2014))
        base = int(np.clip(rng.lognormal(5.2, 0.6), 40, 2500))   # members in 2015
        growth = rng.normal(0.02 + 0.015 * DENOM_YOUTH[denom] + (0.01 if setting == "Urban" else -0.005), 0.035)
        youth_bias = DENOM_YOUTH[denom] + rng.normal(0, 0.4) + (0.3 if setting == "Urban" else -0.2)
        churches.append(dict(
            church_id=f"CH{i+1:04d}",
            church_name=f"{rng.choice(adj)} {rng.choice(kind)} {i+1:04d}",
            region=region, denomination=denom, setting=setting, founded=founded,
            base_=base, growth_=growth, youth_=youth_bias))
    ch = pd.DataFrame(churches)

    cid_parts, birth_parts, join_parts = [], [], []
    for c in ch.itertuples():
        # initial members in 2015: age distribution older for traditional churches
        n0 = c.base_
        mean_age = 38 - 5 * c.youth_
        ages = np.clip(rng.normal(mean_age, 19, n0), 0, 95)
        birth = (START_YEAR - ages).astype(int)
        join = np.minimum(START_YEAR, np.maximum(c.founded, birth + rng.integers(0, 30, n0)))
        join = np.minimum(join, START_YEAR)
        cid_parts.append(np.full(n0, c.Index)); birth_parts.append(birth); join_parts.append(join)
        # yearly joiners 2016..2026
        size = n0
        for y in range(START_YEAR + 1, END_YEAR + 1):
            joiners = max(0, int(rng.poisson(size * (0.07 + max(c.growth_, -0.05)))))
            # joiner ages: mix of births (children) and young adults, tilted by youth bias
            kids = rng.binomial(joiners, 0.25)
            adults = joiners - kids
            a_kids = rng.integers(0, 3, kids)
            a_adults = np.clip(rng.normal(30 - 3 * c.youth_, 11, adults), 13, 90)
            a = np.concatenate([a_kids, a_adults])
            cid_parts.append(np.full(len(a), c.Index)); birth_parts.append((y - a).astype(int)); join_parts.append(np.full(len(a), y))
            size = size * (1 + c.growth_)

    m = pd.DataFrame({"church_id": ch.church_id.to_numpy()[np.concatenate(cid_parts)],
                      "birth_year": np.concatenate(birth_parts), "join_year": np.concatenate(join_parts)})
    m.insert(0, "member_id", "M" + pd.Series(np.arange(1, len(m) + 1)).astype(str).str.zfill(7))
    m["gender"] = rng.choice(["F", "M"], size=len(m), p=[0.56, 0.44])

    # departures: annual hazard, higher for 18-30s (migration/university) and 80+ (mortality)
    leave = np.full(len(m), np.nan)
    growth_map = ch.set_index("church_id")["growth_"]
    hzbase_ = 0.06 - m["church_id"].map(growth_map).values * 0.8
    for y in range(START_YEAR + 1, END_YEAR + 1):
        active = (m["join_year"].values < y) & np.isnan(leave)
        age = y - m["birth_year"].values
        hz = np.clip(hzbase_ + np.where((age >= 18) & (age <= 30), 0.04, 0) + np.where(age >= 80, 0.08, 0), 0.01, 0.4)
        gone = active & (rng.random(len(m)) < hz)
        leave[gone] = y
    m["leave_year"] = pd.array(leave, dtype="Int64")

    ch = ch.drop(columns=["base_", "growth_", "youth_"])
    return ch, m


if __name__ == "__main__":
    ch, m = generate()
    OUT.mkdir(exist_ok=True)
    ch.to_csv(OUT / "churches.csv.gz", index=False, compression="gzip")
    m.to_csv(OUT / "members.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 1})
    print(f"{len(ch):,} churches, {len(m):,} member records")
