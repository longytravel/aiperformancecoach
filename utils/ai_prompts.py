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

Tenure Bands:
- Attaining Foundation (0-3 months): New starters, learning basics
- Attaining Competence (4-12 months): Building skills, increasing independence
- Maintaining Competence (13-24 months): Consistent performer, reliable
- Maintaining Excellence (25+ months): Expert, role model potential

When providing coaching advice:
1. Be specific and actionable
2. Consider tenure - new starters need different support than experienced colleagues
3. Focus on 1-2 priority areas rather than overwhelming with feedback
4. Suggest concrete next steps
5. Be encouraging while being honest about areas needing improvement
6. Reference UK banking context and compliance requirements where relevant

Always maintain a professional, supportive tone appropriate for manager-to-colleague coaching conversations."""


def get_colleague_summary_prompt(colleague_data, metrics_data, objectives_data):
    """Generate prompt for colleague performance summary."""
    return f"""Based on the following colleague data, provide a concise performance summary with strengths, areas for improvement, and coaching recommendations.

COLLEAGUE PROFILE:
- Name: {colleague_data['Name']}
- Team: {colleague_data['Team']}
- Tenure: {colleague_data['Tenure_Months']} months ({colleague_data['Tenure_Band']})

LATEST PERFORMANCE METRICS:
- Quality Score: {metrics_data['Quality_Pct']}%
- FCR: {metrics_data['FCR_Pct']}%
- CSAT: {metrics_data['CSAT_Pct']}%
- NPS: {metrics_data['NPS']}
- AHT: {metrics_data['AHT_Min']} minutes
- Adherence: {metrics_data['Adherence_Pct']}%
- Critical Errors: {metrics_data['Critical_Errors']}
- Complaint Rate: {metrics_data['Complaint_Rate']} per 1,000 calls

OBJECTIVES STATUS:
{objectives_data}

Provide:
1. A 2-3 sentence overall summary
2. Top 2 strengths
3. Top 2 areas for improvement
4. 2-3 specific coaching recommendations for the next 1:1"""


def get_struggling_analysis_prompt(colleague_data, metrics_history, peer_comparison):
    """Generate prompt for analyzing why a colleague is struggling."""
    return f"""Analyze why this colleague may be struggling and suggest interventions.

COLLEAGUE PROFILE:
- Name: {colleague_data['Name']}
- Team: {colleague_data['Team']}
- Tenure: {colleague_data['Tenure_Months']} months ({colleague_data['Tenure_Band']})

3-MONTH PERFORMANCE TREND:
{metrics_history}

COMPARISON TO PEERS (same tenure band):
{peer_comparison}

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
