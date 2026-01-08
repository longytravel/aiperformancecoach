"""
AI prompt templates for the Claude-powered coaching features.
"""

SYSTEM_PROMPT = """You are an expert AI Performance Coach for a UK banking contact centre. Your role is to help managers understand colleague performance, identify areas for improvement, and provide actionable coaching suggestions.

You have access to performance data including:
- Productivity metrics (Call Volume, AHT, Hold Time, ACW, Shrinkage, Adherence)
- Quality metrics (Quality Score, Critical Errors, Complaint Rate, Transfer Rate, Repeat Calls, FCR)
- Customer Experience metrics (CSAT, NPS, Sentiment Score)
- Learning metrics (Training Hours, Coaching Actions)
- Yearly objectives and development goals

IMPORTANT - TENURE-ADJUSTED TARGETS:
Targets are adjusted based on colleague tenure. You will be provided with both the colleague's actual performance AND their tenure-specific targets.

CRITICAL: Only flag a metric as an area for improvement if the actual value is BELOW the target for that colleague's tenure band. If a metric is AT or ABOVE target, it should be considered a strength or neutral - never a development area.

Tenure Bands:
- Attaining Foundation (0-3 months): New starters, learning basics - LOWER targets
- Attaining Competence (4-12 months): Building skills, increasing independence - MODERATE targets
- Maintaining Competence (13-24 months): Consistent performer, reliable - HIGHER targets
- Maintaining Excellence (25+ months): Expert, role model potential - HIGHEST targets

When providing coaching advice:
1. ALWAYS compare actuals to the provided tenure-adjusted targets
2. Be specific and actionable
3. Consider tenure - new starters need different support than experienced colleagues
4. Focus on 1-2 priority areas (ONLY metrics below target) rather than overwhelming with feedback
5. Suggest concrete next steps
6. Be encouraging while being honest about areas needing improvement
7. Reference UK banking context and compliance requirements where relevant

Always maintain a professional, supportive tone appropriate for manager-to-colleague coaching conversations."""


def get_colleague_summary_prompt(colleague_data, metrics_data, targets_data, objectives_data, learning_data=None):
    """Generate prompt for colleague performance summary with learning support."""

    # Helper to determine status vs target
    def status(actual, target, higher_is_better=True):
        if higher_is_better:
            if actual >= target: return "ABOVE TARGET"
            else: return "BELOW TARGET"
        else:  # Lower is better (AHT, Hold, ACW, etc.)
            if actual <= target: return "ABOVE TARGET"
            else: return "BELOW TARGET"

    # Format learning recommendations if provided
    learning_section = ""
    if learning_data and len(learning_data) > 0:
        learning_section = "\n\nAVAILABLE SUPPORT & LEARNING (from our Workday system):\n"
        for course in learning_data[:10]:  # Limit to 10 courses
            prereqs = course.get('Prerequisites', 'None')
            manager_pair = course.get('Manager_Pairing', 'None')
            external = course.get('External_Resource_Name', '')
            external_url = course.get('External_Resource_URL', '')

            learning_section += f"""
- {course['Course_ID']}: {course['Course_Name']}
  Category: {course['Category']} | Level: {course['Level']} | Duration: {course['Duration']} | Format: {course['Format']}
  Description: {course['Description']}
  Prerequisites: {prereqs}
  Manager action required: {manager_pair if manager_pair != 'None' else 'No manager action required'}
  External resource: {external} ({external_url})
"""

    return f"""Based on the following colleague data, provide a concise performance summary with strengths, areas for improvement, coaching recommendations, AND a tailored support plan with specific learning recommendations.

COLLEAGUE PROFILE:
- Name: {colleague_data['Name']}
- Team: {colleague_data['Team']}
- Tenure: {colleague_data['Tenure_Months']} months ({colleague_data['Tenure_Band']})

IMPORTANT: This colleague's targets are adjusted for their tenure band. Only flag metrics BELOW TARGET as improvement areas.

PERFORMANCE vs TENURE-ADJUSTED TARGETS:
| Metric | Actual | Target | Status |
|--------|--------|--------|--------|
| Quality Score | {metrics_data['Quality_Pct']}% | {targets_data['Quality_Target']}% | {status(metrics_data['Quality_Pct'], targets_data['Quality_Target'])} |
| FCR | {metrics_data['FCR_Pct']}% | {targets_data['FCR_Target']}% | {status(metrics_data['FCR_Pct'], targets_data['FCR_Target'])} |
| CSAT | {metrics_data['CSAT_Pct']}% | {targets_data['CSAT_Target']}% | {status(metrics_data['CSAT_Pct'], targets_data['CSAT_Target'])} |
| NPS | {metrics_data['NPS']} | {targets_data['NPS_Target']} | {status(metrics_data['NPS'], targets_data['NPS_Target'])} |
| AHT | {metrics_data['AHT_Min']} min | {targets_data['AHT_Target']} min | {status(metrics_data['AHT_Min'], targets_data['AHT_Target'], False)} |
| Adherence | {metrics_data['Adherence_Pct']}% | {targets_data['Adherence_Target']}% | {status(metrics_data['Adherence_Pct'], targets_data['Adherence_Target'])} |
| Hold Time | {metrics_data['Hold_Min']} min | {targets_data['Hold_Target']} min | {status(metrics_data['Hold_Min'], targets_data['Hold_Target'], False)} |
| ACW | {metrics_data['ACW_Min']} min | {targets_data['ACW_Target']} min | {status(metrics_data['ACW_Min'], targets_data['ACW_Target'], False)} |
| Critical Errors | {metrics_data['Critical_Errors']} | 0 | {"ABOVE TARGET" if metrics_data['Critical_Errors'] == 0 else "BELOW TARGET"} |
| Complaint Rate | {metrics_data['Complaint_Rate']} | {targets_data['Complaint_Rate_Target']} | {status(metrics_data['Complaint_Rate'], targets_data['Complaint_Rate_Target'], False)} |

OBJECTIVES STATUS:
{objectives_data}
{learning_section}

Provide a comprehensive support plan with the following sections:

## Performance Summary
A 2-3 sentence overall summary of this colleague's performance.

## Strengths
Top 2 strengths (metrics that are ABOVE TARGET).

## Areas for Support
Top 2 areas needing support (ONLY metrics that are BELOW TARGET - if none, say "Meeting all targets").

## Coaching Recommendations
2-3 specific coaching actions for the manager to take in the next 1:1.

## Recommended Support & Learning Journey
Based on the available courses above, recommend a tailored learning journey for this colleague. For EACH recommended course, include:
1. The Course ID and name
2. Why this course will help (linked to their specific metrics)
3. Any prerequisites they need to complete first
4. Any actions the MANAGER needs to take before or during the course
5. The external resource link they can access immediately

Structure the learning as a sequenced journey (what to do first, second, etc.) with a suggested timeline.

## Manager Actions Required
List any specific actions the manager must take to support this colleague's development (e.g., completing their own training, scheduling specific conversations, setting up shadowing)."""


def get_struggling_analysis_prompt(colleague_data, metrics_history, targets_data, peer_comparison):
    """Generate prompt for analyzing why a colleague is struggling."""
    return f"""Analyze why this colleague may be struggling and suggest interventions.

COLLEAGUE PROFILE:
- Name: {colleague_data['Name']}
- Team: {colleague_data['Team']}
- Tenure: {colleague_data['Tenure_Months']} months ({colleague_data['Tenure_Band']})

TENURE-ADJUSTED TARGETS FOR THIS COLLEAGUE:
- Quality: {targets_data['Quality_Target']}%
- FCR: {targets_data['FCR_Target']}%
- CSAT: {targets_data['CSAT_Target']}%
- NPS: {targets_data['NPS_Target']}
- AHT: {targets_data['AHT_Target']} min
- Adherence: {targets_data['Adherence_Target']}%
- Hold Time: {targets_data['Hold_Target']} min
- Complaint Rate: {targets_data['Complaint_Rate_Target']}

3-MONTH PERFORMANCE TREND:
{metrics_history}

COMPARISON TO PEERS (same tenure band):
{peer_comparison}

IMPORTANT: When analyzing, compare performance to the TENURE-ADJUSTED TARGETS above, not absolute values.

Provide:
1. Root cause analysis - what appears to be driving the performance issues?
2. Is this a skills gap, motivation issue, process issue, or something else?
3. Specific intervention recommendations
4. Suggested timeline for improvement
5. Warning signs to monitor"""


def get_coaching_plan_prompt(colleague_data, focus_area, current_performance, target_performance):
    """Generate prompt for creating a coaching plan."""
    return f"""Create a detailed coaching plan for improving {focus_area}.

COLLEAGUE PROFILE:
- Name: {colleague_data['Name']}
- Tenure: {colleague_data['Tenure_Months']} months ({colleague_data['Tenure_Band']})

CURRENT STATE:
{current_performance}

TARGET STATE:
{target_performance}

Create a coaching plan including:
1. Specific, measurable goal
2. Week-by-week action items (4-week plan)
3. Resources or training needed
4. How to measure progress
5. Conversation starters for the manager
6. Potential barriers and how to overcome them"""


def get_team_analysis_prompt(team_name, team_metrics, benchmarks):
    """Generate prompt for team performance analysis."""
    return f"""Analyze the performance of {team_name} team and identify improvement opportunities.

TEAM METRICS (averages):
{team_metrics}

INDUSTRY BENCHMARKS:
{benchmarks}

Provide:
1. Overall team health assessment
2. Top performing areas
3. Areas needing focus
4. Recommendations for team-level interventions
5. Patterns or trends to watch"""


def get_chat_context_prompt(question, data_context):
    """Generate prompt for chat interactions."""
    return f"""Answer the following question about contact centre performance data.

AVAILABLE DATA CONTEXT:
{data_context}

QUESTION: {question}

Provide a helpful, accurate response based on the data. If you need to make assumptions, state them clearly. If the question cannot be answered with the available data, explain what additional information would be needed."""


def get_comparison_prompt(cohort1_name, cohort1_data, cohort2_name, cohort2_data):
    """Generate prompt for comparing two cohorts."""
    return f"""Compare the performance of these two cohorts:

COHORT 1: {cohort1_name}
{cohort1_data}

COHORT 2: {cohort2_name}
{cohort2_data}

Provide:
1. Key differences in performance
2. Where each cohort excels
3. Recommendations for each cohort
4. Insights on what might be driving the differences"""
