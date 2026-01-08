"""
Data loading utilities for the AI Performance Coach application.
"""
import pandas as pd
import os
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def load_colleagues():
    """Load colleague dimension data."""
    return pd.read_csv(DATA_DIR / "colleagues.csv")


def load_monthly_metrics():
    """Load monthly performance metrics."""
    df = pd.read_csv(DATA_DIR / "monthly_metrics.csv")
    df['Month'] = pd.to_datetime(df['Month'])
    return df


def load_targets():
    """Load tenure-based targets."""
    return pd.read_csv(DATA_DIR / "targets.csv")


def load_objectives():
    """Load colleague objectives."""
    df = pd.read_csv(DATA_DIR / "objectives.csv")
    df['Target_Date'] = pd.to_datetime(df['Target_Date'])
    return df


def load_industry_benchmarks():
    """Load industry benchmark data."""
    return pd.read_csv(DATA_DIR / "industry_benchmarks.csv")


def get_colleague_with_metrics(colleague_id):
    """Get a specific colleague with their latest metrics."""
    colleagues = load_colleagues()
    metrics = load_monthly_metrics()

    colleague = colleagues[colleagues['Colleague_ID'] == colleague_id].iloc[0]
    colleague_metrics = metrics[metrics['Colleague_ID'] == colleague_id].sort_values('Month', ascending=False)

    return colleague, colleague_metrics


def get_colleague_objectives(colleague_id):
    """Get objectives for a specific colleague."""
    objectives = load_objectives()
    return objectives[objectives['Colleague_ID'] == colleague_id]


def get_team_metrics(team_name, month=None):
    """Get metrics for a specific team, optionally filtered by month."""
    colleagues = load_colleagues()
    metrics = load_monthly_metrics()

    team_colleagues = colleagues[colleagues['Team'] == team_name]['Colleague_ID'].tolist()
    team_metrics = metrics[metrics['Colleague_ID'].isin(team_colleagues)]

    if month:
        team_metrics = team_metrics[team_metrics['Month'] == month]

    return team_metrics


def get_tenure_band_metrics(tenure_band, month=None):
    """Get metrics for a specific tenure band, optionally filtered by month."""
    colleagues = load_colleagues()
    metrics = load_monthly_metrics()

    band_colleagues = colleagues[colleagues['Tenure_Band'] == tenure_band]['Colleague_ID'].tolist()
    band_metrics = metrics[metrics['Colleague_ID'].isin(band_colleagues)]

    if month:
        band_metrics = band_metrics[band_metrics['Month'] == month]

    return band_metrics


def get_latest_month():
    """Get the latest month in the data."""
    metrics = load_monthly_metrics()
    return metrics['Month'].max()


def get_all_data():
    """Load all data and merge for comprehensive view."""
    colleagues = load_colleagues()
    metrics = load_monthly_metrics()
    targets = load_targets()

    # Get latest metrics for each colleague
    latest_metrics = metrics.sort_values('Month', ascending=False).groupby('Colleague_ID').first().reset_index()

    # Merge colleague info with latest metrics
    combined = pd.merge(colleagues, latest_metrics, on='Colleague_ID', how='left')

    # Merge with targets based on tenure band
    combined = pd.merge(combined, targets, on='Tenure_Band', how='left')

    return combined
