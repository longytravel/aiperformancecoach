"""
Calculation utilities for derived metrics and performance scoring.
"""
import pandas as pd
import numpy as np


def calculate_metric_score(actual, target, higher_is_better=True):
    """
    Calculate a component score with realistic distribution.

    Scoring logic (designed for bell curve):
    - 10%+ above target: 95-100 (Role Model territory)
    - At target to 10% above: 80-94 (Strong territory)
    - Within 10% below target: 65-79 (On Track territory)
    - 10-20% below target: 50-64 (Focus territory)
    - More than 20% below: 0-49 (Below territory)
    """
    if higher_is_better:
        if target == 0:
            return 0
        ratio = actual / target
    else:
        # For metrics where lower is better (AHT, Hold, ACW)
        if actual == 0:
            return 100
        ratio = target / actual

    if ratio >= 1.10:
        # 10%+ above target: 95-100
        return min(100, 95 + (ratio - 1.10) * 50)
    elif ratio >= 1.0:
        # At target to 10% above: 80-94
        return 80 + (ratio - 1.0) * 140
    elif ratio >= 0.90:
        # Within 10% of target: 65-79
        return 65 + (ratio - 0.90) * 140
    elif ratio >= 0.80:
        # 10-20% below: 50-64
        return 50 + (ratio - 0.80) * 140
    else:
        # More than 20% below: 0-49
        return max(0, ratio * 62.5)


def calculate_performance_score(row, targets_row):
    """
    Calculate overall performance score (0-100) based on weighted metrics.

    Uses realistic scoring where missing targets results in meaningful penalties.

    Weights:
    - Quality: 25%
    - FCR: 20%
    - CSAT: 20%
    - AHT: 15%
    - Adherence: 10%
    - Compliance (Critical Errors): 10%
    """
    scores = []
    weights = []

    # Quality score (25%)
    quality_score = calculate_metric_score(row['Quality_Pct'], targets_row['Quality_Target'], True)
    scores.append(quality_score)
    weights.append(0.25)

    # FCR score (20%)
    fcr_score = calculate_metric_score(row['FCR_Pct'], targets_row['FCR_Target'], True)
    scores.append(fcr_score)
    weights.append(0.20)

    # CSAT score (20%)
    csat_score = calculate_metric_score(row['CSAT_Pct'], targets_row['CSAT_Target'], True)
    scores.append(csat_score)
    weights.append(0.20)

    # AHT score (15%) - lower is better
    aht_score = calculate_metric_score(row['AHT_Min'], targets_row['AHT_Target'], False)
    scores.append(aht_score)
    weights.append(0.15)

    # Adherence score (10%)
    adherence_score = calculate_metric_score(row['Adherence_Pct'], targets_row['Adherence_Target'], True)
    scores.append(adherence_score)
    weights.append(0.10)

    # Compliance score (10%) - 0 errors = 100, each error reduces by 40
    compliance_score = max(0, 100 - (row['Critical_Errors'] * 40))
    scores.append(compliance_score)
    weights.append(0.10)

    # Calculate weighted average
    overall_score = sum(s * w for s, w in zip(scores, weights))

    return round(overall_score, 1)


def get_performance_status(score):
    """
    Convert performance score to status category.

    - Role Model: 90+ (consistently exceeding targets)
    - Strong: 80-89 (meeting most targets well)
    - On Track: 65-79 (generally meeting expectations)
    - Focus: 50-64 (needs support and coaching)
    - Below: <50 (significant concerns, urgent intervention needed)
    """
    if score >= 90:
        return "Role Model"
    elif score >= 80:
        return "Strong"
    elif score >= 65:
        return "On Track"
    elif score >= 50:
        return "Focus"
    else:
        return "Below"


def get_status_color(status):
    """Get color for performance status."""
    colors = {
        "Role Model": "#10B981",  # Green
        "Strong": "#3B82F6",       # Blue
        "On Track": "#6366F1",     # Indigo
        "Focus": "#F59E0B",        # Amber
        "Below": "#EF4444"         # Red
    }
    return colors.get(status, "#6B7280")


def calculate_trend(metrics_df, metric_column):
    """
    Calculate trend direction based on 3-month data.
    Returns: Improving, Stable, or Declining
    """
    if len(metrics_df) < 2:
        return "Stable"

    sorted_df = metrics_df.sort_values('Month')
    values = sorted_df[metric_column].values

    # Calculate trend using linear regression slope
    if len(values) >= 2:
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Determine trend based on slope magnitude
        threshold = 0.02 * np.mean(values)  # 2% of mean as threshold

        if slope > threshold:
            return "Improving"
        elif slope < -threshold:
            return "Declining"

    return "Stable"


def get_trend_icon(trend):
    """Get icon for trend direction."""
    icons = {
        "Improving": "📈",
        "Stable": "➡️",
        "Declining": "📉"
    }
    return icons.get(trend, "➡️")


def calculate_metric_rag(actual, target, higher_is_better=True):
    """
    Calculate RAG status for a metric.
    Returns: Green, Amber, or Red
    """
    if higher_is_better:
        ratio = actual / target if target > 0 else 0
        if ratio >= 1.0:
            return "Green"
        elif ratio >= 0.9:
            return "Amber"
        else:
            return "Red"
    else:
        # For metrics where lower is better (e.g., AHT)
        ratio = target / actual if actual > 0 else 0
        if ratio >= 1.0:
            return "Green"
        elif ratio >= 0.9:
            return "Amber"
        else:
            return "Red"


def get_rag_color(rag):
    """Get color for RAG status."""
    colors = {
        "Green": "#10B981",
        "Amber": "#F59E0B",
        "Red": "#EF4444"
    }
    return colors.get(rag, "#6B7280")


def identify_coaching_priority(row, targets_row):
    """
    Identify the top priority metric for coaching.
    Returns the metric with the largest gap to target.
    """
    metrics = {
        "Quality": (row['Quality_Pct'], targets_row['Quality_Target'], True),
        "FCR": (row['FCR_Pct'], targets_row['FCR_Target'], True),
        "CSAT": (row['CSAT_Pct'], targets_row['CSAT_Target'], True),
        "AHT": (row['AHT_Min'], targets_row['AHT_Target'], False),
        "Adherence": (row['Adherence_Pct'], targets_row['Adherence_Target'], True),
    }

    max_gap = 0
    priority = None

    for metric, (actual, target, higher_better) in metrics.items():
        if higher_better:
            gap = (target - actual) / target if target > 0 else 0
        else:
            gap = (actual - target) / target if target > 0 else 0

        if gap > max_gap:
            max_gap = gap
            priority = metric

    return priority if priority else "Maintain Performance"


def calculate_risk_flag(row):
    """
    Determine if colleague has a risk flag.
    Risk conditions:
    - Critical errors > 0
    - Quality < 75%
    - CSAT < 75%
    - Complaint rate > 7
    """
    risks = []

    if row['Critical_Errors'] > 0:
        risks.append("Compliance Risk")
    if row['Quality_Pct'] < 75:
        risks.append("Quality Risk")
    if row['CSAT_Pct'] < 75:
        risks.append("CX Risk")
    if row['Complaint_Rate'] > 7:
        risks.append("Complaint Risk")

    return risks if risks else None


def calculate_peer_quartile(colleague_metrics, all_metrics_same_band):
    """
    Calculate which quartile the colleague falls into within their tenure band.
    Returns: Q1 (Top 25%), Q2, Q3, or Q4 (Bottom 25%)
    """
    if len(all_metrics_same_band) < 4:
        return "N/A"

    # Use overall performance score for ranking
    scores = all_metrics_same_band['Performance_Score'].values
    colleague_score = colleague_metrics['Performance_Score']

    percentile = (scores < colleague_score).sum() / len(scores) * 100

    if percentile >= 75:
        return "Q1 (Top 25%)"
    elif percentile >= 50:
        return "Q2"
    elif percentile >= 25:
        return "Q3"
    else:
        return "Q4 (Bottom 25%)"


def calculate_goal_summary(objectives_df):
    """
    Calculate summary of goal attainment.
    Returns dict with counts by status.
    """
    if len(objectives_df) == 0:
        return {"total": 0, "achieved": 0, "on_track": 0, "at_risk": 0, "behind": 0}

    summary = {
        "total": len(objectives_df),
        "achieved": len(objectives_df[objectives_df['Status'] == 'Achieved']),
        "on_track": len(objectives_df[objectives_df['Status'] == 'On Track']),
        "at_risk": len(objectives_df[objectives_df['Status'] == 'At Risk']),
        "behind": len(objectives_df[objectives_df['Status'] == 'Behind'])
    }

    return summary


def compare_to_benchmark(actual, benchmark_avg, benchmark_top, benchmark_bottom, higher_is_better=True):
    """
    Compare actual performance to industry benchmarks.
    Returns position relative to benchmarks.
    """
    if higher_is_better:
        if actual >= benchmark_top:
            return "Top Quartile"
        elif actual >= benchmark_avg:
            return "Above Average"
        elif actual >= benchmark_bottom:
            return "Below Average"
        else:
            return "Bottom Quartile"
    else:
        if actual <= benchmark_top:
            return "Top Quartile"
        elif actual <= benchmark_avg:
            return "Above Average"
        elif actual <= benchmark_bottom:
            return "Below Average"
        else:
            return "Bottom Quartile"
