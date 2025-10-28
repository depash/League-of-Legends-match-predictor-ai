import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
DATA_RAW = Path("backend/data/raw")
DATA_PROCESSED = Path("backend/data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# =========================
# UTILS
# =========================
def load_csv(filename: str) -> pd.DataFrame:
    """Load CSV, clean whitespace, lowercase headers."""
    df = pd.read_csv(DATA_RAW / filename)
    df.columns = df.columns.str.strip().str.lower()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(how="all")
    print(f"✅ Loaded {filename} — {len(df)} rows, {len(df.columns)} cols")
    return df


# =========================
# LOAD RAW DATA
# =========================
champs = load_csv("champs.csv")
participants = load_csv("participants.csv")
matches = load_csv("matches.csv")
stats1 = load_csv("stats1.csv")
stats2 = load_csv("stats2.csv")
teamstats = load_csv("teamstats.csv")
teambans = load_csv("teambans.csv")

stats = pd.concat([stats1, stats2], ignore_index=True)
print(f"✅ Combined stats1 + stats2 — {len(stats)} rows")


# =========================
# MERGE PLAYER + MATCH DATA
# =========================
# 1️⃣ Combine participants with stats
player_df = participants.merge(stats, on="id", how="inner")
print(f"🔗 participants + stats merged → {len(player_df)} rows")

# 2️⃣ Add champion names
player_df = player_df.merge(
    champs, left_on="championid", right_on="id", how="left", suffixes=("", "_champ")
)
player_df.rename(columns={"name": "champion_name"}, inplace=True)
player_df.drop(columns=["id_champ"], inplace=True, errors="ignore")

# 3️⃣ Add match metadata
player_df = player_df.merge(matches, left_on="matchid", right_on="id", how="left", suffixes=("", "_match"))
player_df.drop(columns=["id_match"], inplace=True, errors="ignore")

player_df.columns = player_df.columns.str.strip().str.lower()
player_df = player_df.fillna(0)

print(f"✅ Player-level dataset ready — {len(player_df)} rows, {len(player_df.columns)} cols")


# =========================
# INFER TEAMS (since not in participants)
# =========================
# Sort players by matchid, then player number
player_df = player_df.sort_values(by=["matchid", "player"])

# Assign teamid = 100 for first 5 players, 200 for next 5 in each match
player_df["teamid"] = player_df.groupby("matchid").cumcount().apply(lambda x: 100 if x < 5 else 200)

print("✅ Inferred teamid: 100 and 200 per match")


# =========================
# AGGREGATE TO TEAM LEVEL
# =========================
numeric_cols = player_df.select_dtypes(include="number").columns.tolist()

# We don't want to aggregate these as numeric stats
exclude_cols = ["win", "matchid", "teamid", "player", "championid", "ss1", "ss2"]
agg_cols = [c for c in numeric_cols if c not in exclude_cols]

team_df = (
    player_df.groupby(["matchid", "teamid"], as_index=False)
    .agg({col: "sum" for col in agg_cols})
)

# Determine team win (any player win = 1)
team_win = player_df.groupby(["matchid", "teamid"])["win"].max().reset_index()
team_df = team_df.merge(team_win, on=["matchid", "teamid"], how="left")

print(f"✅ Aggregated to team level — {len(team_df)} rows")


# =========================
# ADD TEAM STATS + BANS
# =========================
team_df = team_df.merge(teamstats, on=["matchid", "teamid"], how="left")
team_df = team_df.merge(teambans, on=["matchid", "teamid"], how="left")
team_df = team_df.fillna(0)

print(f"✅ Merged teamstats + teambans — {len(team_df)} rows")


# =========================
# PIVOT TEAMS SIDE-BY-SIDE
# =========================
teamA = team_df[team_df["teamid"] == 100].copy()
teamB = team_df[team_df["teamid"] == 200].copy()

teamA = teamA.add_prefix("teamA_")
teamB = teamB.add_prefix("teamB_")

# matchid is teamA_matchid == teamB_matchid
match_df = pd.merge(teamA, teamB, left_on="teamA_matchid", right_on="teamB_matchid", suffixes=("", ""))
match_df["matchid"] = match_df["teamA_matchid"]

# Label = whether Team A won
match_df["teamA_win"] = match_df["teamA_win"].astype(int)

print(f"✅ Created match-level dataset — {len(match_df)} rows, {len(match_df.columns)} columns")


# =========================
# SAVE FINAL DATASETS
# =========================
out_path = DATA_PROCESSED / "match_training_data.csv"
match_df.to_csv(out_path, index=False)
print(f"💾 Saved match-level dataset to {out_path}")
