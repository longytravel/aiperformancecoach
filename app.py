"""
AI Performance Coach - Main Streamlit Application
A prototype dashboard for contact centre managers to understand colleague performance
and receive AI-powered coaching suggestions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Import utilities
import sys
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import (
    load_colleagues, load_monthly_metrics, load_targets,
    load_objectives, load_industry_benchmarks, get_all_data,
    get_colleague_with_metrics, get_colleague_objectives
)
from utils.calculations import (
    calculate_performance_score, get_performance_status, get_status_color,
    calculate_trend, get_trend_icon, calculate_metric_rag, get_rag_color,
    identify_coaching_priority, calculate_risk_flag, calculate_goal_summary,
    compare_to_benchmark
)
from utils.ai_prompts import (
    SYSTEM_PROMPT, get_colleague_summary_prompt, get_struggling_analysis_prompt,
    get_coaching_plan_prompt, get_chat_context_prompt
)

# Page configuration
st.set_page_config(
    page_title="AI Performance Coach",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Metric Tooltips - explanations for all key metrics
METRIC_TOOLTIPS = {
    "Quality": "Overall quality score from QA evaluations. Measures compliance, accuracy, and customer service standards on calls.",
    "FCR": "First Call Resolution - percentage of calls resolved without the customer needing to call back. Higher is better.",
    "CSAT": "Customer Satisfaction Score - rating from post-call surveys (1-100%). Measures how satisfied customers are with the service.",
    "AHT": "Average Handle Time - total time spent on calls including talk time, hold time, and after-call work. Lower is generally better, but not at expense of quality.",
    "NPS": "Net Promoter Score (-100 to +100) - measures customer loyalty by asking how likely they are to recommend us. Above 0 is good, above 50 is excellent.",
    "Adherence": "Schedule Adherence - percentage of time the colleague follows their assigned schedule. Shows reliability and availability.",
    "Hold_Time": "Average time customers are placed on hold during calls. Lower is better - long holds frustrate customers.",
    "ACW": "After Call Work - time spent on wrap-up tasks after the call ends (notes, admin). Should be efficient but thorough.",
    "Shrinkage": "Time away from phones for non-productive activities (breaks, meetings, training). Lower percentage means more availability.",
    "Transfer": "Percentage of calls transferred to another team/colleague. Lower is better - indicates ability to resolve issues.",
    "Repeat_Call": "Percentage of customers who call back within 7 days. Lower is better - indicates issues being fully resolved.",
    "Complaint_Rate": "Number of formal complaints per 1000 calls. Lower is better - indicates quality of service.",
    "Critical_Errors": "Serious compliance or quality failures requiring immediate attention. Should always be zero.",
    "Sentiment": "AI-analysed customer sentiment from call recordings. Ranges from -1 (negative) to +1 (positive).",
    "Call_Volume": "Total number of calls handled. Used to understand workload and capacity.",
    "Performance_Score": "Overall weighted score (0-100) combining Quality (25%), FCR (20%), CSAT (20%), AHT (15%), Adherence (10%), and Compliance (10%).",
}

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    .green-badge { background-color: #10B981; }
    .amber-badge { background-color: #F59E0B; }
    .red-badge { background-color: #EF4444; }
    .blue-badge { background-color: #3B82F6; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        return Anthropic(api_key=api_key)
    return None


# Load data
@st.cache_data
def load_all_data():
    colleagues = load_colleagues()
    metrics = load_monthly_metrics()
    targets = load_targets()
    objectives = load_objectives()
    benchmarks = load_industry_benchmarks()

    # Calculate performance scores for latest month
    latest_month = metrics['Month'].max()
    latest_metrics = metrics[metrics['Month'] == latest_month].copy()

    scores = []
    statuses = []
    priorities = []
    risks = []

    for _, row in latest_metrics.iterrows():
        colleague = colleagues[colleagues['Colleague_ID'] == row['Colleague_ID']].iloc[0]
        target_row = targets[targets['Tenure_Band'] == colleague['Tenure_Band']].iloc[0]

        score = calculate_performance_score(row, target_row)
        status = get_performance_status(score)
        priority = identify_coaching_priority(row, target_row)
        risk = calculate_risk_flag(row)

        scores.append(score)
        statuses.append(status)
        priorities.append(priority)
        risks.append(risk)

    latest_metrics['Performance_Score'] = scores
    latest_metrics['Performance_Status'] = statuses
    latest_metrics['Coaching_Priority'] = priorities
    latest_metrics['Risk_Flags'] = risks

    # Merge with colleague data
    combined = pd.merge(colleagues, latest_metrics, on='Colleague_ID', how='left')

    return colleagues, metrics, targets, objectives, benchmarks, combined


def call_claude(prompt, system_prompt=SYSTEM_PROMPT):
    """Call Claude API for AI-powered insights."""
    client = get_anthropic_client()
    if not client:
        return "AI features unavailable - API key not configured."

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling AI: {str(e)}"


# ============== PAGE: OVERVIEW DASHBOARD ==============
def show_overview_dashboard(colleagues, metrics, targets, benchmarks, combined):
    st.title("📊 Performance Overview Dashboard")
    st.markdown("---")

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    latest_month = metrics['Month'].max()
    latest_metrics = metrics[metrics['Month'] == latest_month]

    with col1:
        avg_quality = latest_metrics['Quality_Pct'].mean()
        st.metric("Avg Quality", f"{avg_quality:.1f}%",
                  delta=f"{avg_quality - 85:.1f}%" if avg_quality > 85 else f"{avg_quality - 85:.1f}%",
                  help=METRIC_TOOLTIPS["Quality"])

    with col2:
        avg_fcr = latest_metrics['FCR_Pct'].mean()
        st.metric("Avg FCR", f"{avg_fcr:.1f}%", help=METRIC_TOOLTIPS["FCR"])

    with col3:
        avg_csat = latest_metrics['CSAT_Pct'].mean()
        st.metric("Avg CSAT", f"{avg_csat:.1f}%", help=METRIC_TOOLTIPS["CSAT"])

    with col4:
        avg_aht = latest_metrics['AHT_Min'].mean()
        st.metric("Avg AHT", f"{avg_aht:.1f} min", help=METRIC_TOOLTIPS["AHT"])

    with col5:
        total_calls = latest_metrics['Call_Volume'].sum()
        st.metric("Total Calls", f"{total_calls:,}", help=METRIC_TOOLTIPS["Call_Volume"])

    st.markdown("---")

    # Performance Distribution & Cohort Comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Status Distribution")

        status_counts = combined['Performance_Status'].value_counts()
        status_order = ['Role Model', 'Strong', 'On Track', 'Focus', 'Below']
        status_colors = ['#10B981', '#3B82F6', '#6366F1', '#F59E0B', '#EF4444']

        fig = go.Figure(data=[go.Pie(
            labels=[s for s in status_order if s in status_counts.index],
            values=[status_counts.get(s, 0) for s in status_order if s in status_counts.index],
            marker_colors=[status_colors[status_order.index(s)] for s in status_order if s in status_counts.index],
            hole=0.4
        )])
        fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Team vs Industry Benchmark")

        # Get key metrics for comparison
        team_avg = {
            'FCR': latest_metrics['FCR_Pct'].mean(),
            'Quality': latest_metrics['Quality_Pct'].mean(),
            'CSAT': latest_metrics['CSAT_Pct'].mean(),
            'NPS': latest_metrics['NPS'].mean()
        }

        industry_avg = {
            'FCR': benchmarks[benchmarks['Metric'] == 'FCR_Pct']['Industry_Average'].values[0],
            'Quality': benchmarks[benchmarks['Metric'] == 'Quality_Pct']['Industry_Average'].values[0],
            'CSAT': benchmarks[benchmarks['Metric'] == 'CSAT_Pct']['Industry_Average'].values[0],
            'NPS': benchmarks[benchmarks['Metric'] == 'NPS']['Industry_Average'].values[0]
        }

        comparison_df = pd.DataFrame({
            'Metric': list(team_avg.keys()),
            'Our Team': list(team_avg.values()),
            'Industry Average': list(industry_avg.values())
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Our Team', x=comparison_df['Metric'], y=comparison_df['Our Team'], marker_color='#3B82F6'))
        fig.add_trace(go.Bar(name='Industry Average', x=comparison_df['Metric'], y=comparison_df['Industry Average'], marker_color='#9CA3AF'))
        fig.update_layout(barmode='group', height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Risk Alerts
    st.subheader("⚠️ Risk Alerts - Colleagues Needing Attention")

    risk_colleagues = combined[combined['Risk_Flags'].notna()].copy()
    if len(risk_colleagues) > 0:
        for _, row in risk_colleagues.head(5).iterrows():
            risks = row['Risk_Flags']
            if risks:
                with st.expander(f"🚨 {row['Name']} ({row['Team']}) - {row['Performance_Status']}"):
                    st.write(f"**Tenure:** {row['Tenure_Band']} ({row['Tenure_Months']} months)")
                    st.write(f"**Risk Flags:** {', '.join(risks)}")
                    st.write(f"**Coaching Priority:** {row['Coaching_Priority']}")
                    st.write(f"**Performance Score:** {row['Performance_Score']:.1f}/100")
    else:
        st.success("No colleagues with risk flags this month!")

    # Tenure Band Performance
    st.markdown("---")
    st.subheader("Performance by Tenure Band")

    tenure_performance = combined.groupby('Tenure_Band').agg({
        'Performance_Score': 'mean',
        'Quality_Pct': 'mean',
        'FCR_Pct': 'mean',
        'CSAT_Pct': 'mean'
    }).round(1)

    tenure_order = ['Attaining Foundation', 'Attaining Competence', 'Maintaining Competence', 'Maintaining Excellence']
    tenure_performance = tenure_performance.reindex([t for t in tenure_order if t in tenure_performance.index])

    fig = px.bar(tenure_performance, x=tenure_performance.index, y='Performance_Score',
                 color='Performance_Score', color_continuous_scale='Blues',
                 labels={'Performance_Score': 'Avg Score', 'index': 'Tenure Band'})
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============== PAGE: COLLEAGUE EXPLORER ==============
def show_colleague_explorer(colleagues, metrics, combined):
    st.title("👥 Colleague Explorer")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        team_filter = st.selectbox("Filter by Team", ["All"] + list(colleagues['Team'].unique()))

    with col2:
        tenure_filter = st.selectbox("Filter by Tenure Band", ["All"] + list(colleagues['Tenure_Band'].unique()))

    with col3:
        status_filter = st.selectbox("Filter by Status", ["All", "Role Model", "Strong", "On Track", "Focus", "Below"])

    with col4:
        sort_by = st.selectbox("Sort by", ["Performance Score", "Name", "Tenure"])

    # Apply filters
    filtered = combined.copy()

    if team_filter != "All":
        filtered = filtered[filtered['Team'] == team_filter]

    if tenure_filter != "All":
        filtered = filtered[filtered['Tenure_Band'] == tenure_filter]

    if status_filter != "All":
        filtered = filtered[filtered['Performance_Status'] == status_filter]

    # Sort
    if sort_by == "Performance Score":
        filtered = filtered.sort_values('Performance_Score', ascending=False)
    elif sort_by == "Name":
        filtered = filtered.sort_values('Name')
    else:
        filtered = filtered.sort_values('Tenure_Months', ascending=False)

    st.markdown(f"**Showing {len(filtered)} colleagues**")
    st.markdown("---")

    # Display colleagues as cards
    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                row = filtered.iloc[i + j]
                with col:
                    status_color = get_status_color(row['Performance_Status'])

                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 4px solid {status_color};">
                        <h4 style="margin: 0 0 10px 0;">{row['Name']}</h4>
                        <p style="margin: 5px 0; color: #666;"><strong>Team:</strong> {row['Team']}</p>
                        <p style="margin: 5px 0; color: #666;"><strong>Tenure:</strong> {row['Tenure_Band']} ({row['Tenure_Months']}mo)</p>
                        <p style="margin: 5px 0;"><strong>Score:</strong> {row['Performance_Score']:.1f}/100</p>
                        <span style="background-color: {status_color}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 12px;">{row['Performance_Status']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"View Details", key=f"view_{row['Colleague_ID']}"):
                        st.session_state['selected_colleague'] = row['Colleague_ID']
                        st.session_state['page'] = 'Individual View'
                        st.rerun()


# ============== PAGE: INDIVIDUAL COLLEAGUE VIEW ==============
def show_individual_view(colleagues, metrics, targets, objectives, combined):
    st.title("👤 Individual Colleague View")

    # Colleague selector
    colleague_options = {f"{row['Name']} ({row['Team']})": row['Colleague_ID']
                        for _, row in colleagues.iterrows()}

    selected_name = st.selectbox("Select Colleague", list(colleague_options.keys()))
    selected_id = colleague_options[selected_name]

    colleague = colleagues[colleagues['Colleague_ID'] == selected_id].iloc[0]
    colleague_metrics = metrics[metrics['Colleague_ID'] == selected_id].sort_values('Month')
    colleague_objectives = objectives[objectives['Colleague_ID'] == selected_id]
    target_row = targets[targets['Tenure_Band'] == colleague['Tenure_Band']].iloc[0]
    latest = colleague_metrics.iloc[-1]

    # Calculate score and status
    score = calculate_performance_score(latest, target_row)
    status = get_performance_status(score)
    status_color = get_status_color(status)

    st.markdown("---")

    # Profile Header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"### {colleague['Name']}")
        st.write(f"**Team:** {colleague['Team']}")
        st.write(f"**Tenure:** {colleague['Tenure_Band']} ({colleague['Tenure_Months']} months)")

    with col2:
        st.metric("Performance Score", f"{score:.1f}/100", help=METRIC_TOOLTIPS["Performance_Score"])

    with col3:
        st.markdown(f"""
        <div style="background-color: {status_color}; color: white; padding: 10px 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <strong>{status}</strong>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        priority = identify_coaching_priority(latest, target_row)
        st.metric("Focus Area", priority)

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scorecard", "📈 Trends", "🎯 Objectives", "🤖 AI Coaching"])

    with tab1:
        st.subheader("Performance Scorecard")

        # Metrics vs Targets - (display name, actual value, target, unit, higher_is_better, tooltip_key)
        metrics_data = [
            ("Quality Score", latest['Quality_Pct'], target_row['Quality_Target'], "%", True, "Quality"),
            ("FCR", latest['FCR_Pct'], target_row['FCR_Target'], "%", True, "FCR"),
            ("CSAT", latest['CSAT_Pct'], target_row['CSAT_Target'], "%", True, "CSAT"),
            ("AHT", latest['AHT_Min'], target_row['AHT_Target'], "min", False, "AHT"),
            ("Adherence", latest['Adherence_Pct'], target_row['Adherence_Target'], "%", True, "Adherence"),
            ("Hold Time", latest['Hold_Min'], target_row['Hold_Target'], "min", False, "Hold_Time"),
            ("ACW", latest['ACW_Min'], target_row['ACW_Target'], "min", False, "ACW"),
            ("NPS", latest['NPS'], target_row['NPS_Target'], "", True, "NPS"),
        ]

        cols = st.columns(4)
        for i, (name, actual, target, unit, higher_better, tooltip_key) in enumerate(metrics_data):
            rag = calculate_metric_rag(actual, target, higher_better)
            rag_color = get_rag_color(rag)
            tooltip = METRIC_TOOLTIPS.get(tooltip_key, "")

            with cols[i % 4]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 5px 0; border-left: 4px solid {rag_color};" title="{tooltip}">
                    <p style="margin: 0; color: #666; font-size: 12px;">{name} <span style="cursor: help; color: #999;" title="{tooltip}">ⓘ</span></p>
                    <p style="margin: 5px 0; font-size: 20px; font-weight: bold;">{actual}{unit}</p>
                    <p style="margin: 0; color: #999; font-size: 11px;">Target: {target}{unit}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("3-Month Performance Trends")

        # Create trend charts
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Quality Score', 'FCR', 'CSAT', 'AHT'))

        fig.add_trace(go.Scatter(x=colleague_metrics['Month'], y=colleague_metrics['Quality_Pct'],
                                mode='lines+markers', name='Quality', line=dict(color='#3B82F6')), row=1, col=1)
        fig.add_hline(y=target_row['Quality_Target'], line_dash="dash", line_color="gray", row=1, col=1)

        fig.add_trace(go.Scatter(x=colleague_metrics['Month'], y=colleague_metrics['FCR_Pct'],
                                mode='lines+markers', name='FCR', line=dict(color='#10B981')), row=1, col=2)
        fig.add_hline(y=target_row['FCR_Target'], line_dash="dash", line_color="gray", row=1, col=2)

        fig.add_trace(go.Scatter(x=colleague_metrics['Month'], y=colleague_metrics['CSAT_Pct'],
                                mode='lines+markers', name='CSAT', line=dict(color='#8B5CF6')), row=2, col=1)
        fig.add_hline(y=target_row['CSAT_Target'], line_dash="dash", line_color="gray", row=2, col=1)

        fig.add_trace(go.Scatter(x=colleague_metrics['Month'], y=colleague_metrics['AHT_Min'],
                                mode='lines+markers', name='AHT', line=dict(color='#F59E0B')), row=2, col=2)
        fig.add_hline(y=target_row['AHT_Target'], line_dash="dash", line_color="gray", row=2, col=2)

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Trend indicators
        col1, col2, col3, col4 = st.columns(4)
        for col, metric in zip([col1, col2, col3, col4], ['Quality_Pct', 'FCR_Pct', 'CSAT_Pct', 'AHT_Min']):
            trend = calculate_trend(colleague_metrics, metric)
            icon = get_trend_icon(trend)
            with col:
                st.write(f"{icon} {trend}")

    with tab3:
        st.subheader("Yearly Objectives Progress")

        goal_summary = calculate_goal_summary(colleague_objectives)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Achieved", goal_summary['achieved'])
        col2.metric("On Track", goal_summary['on_track'])
        col3.metric("At Risk", goal_summary['at_risk'])
        col4.metric("Behind", goal_summary['behind'])

        st.markdown("---")

        for _, obj in colleague_objectives.iterrows():
            status_colors = {'Achieved': '#10B981', 'On Track': '#3B82F6', 'At Risk': '#F59E0B', 'Behind': '#EF4444'}
            color = status_colors.get(obj['Status'], '#6B7280')

            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <p style="margin: 0; font-weight: bold;">{obj['Objective_Text']}</p>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">{obj['Objective_Type']} - {obj['Category']}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background-color: {color}; color: white; padding: 3px 10px; border-radius: 10px; font-size: 12px;">{obj['Status']}</span>
                        <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">{obj['Progress_Pct']}% complete</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.subheader("AI Coaching Insights")

        if st.button("Generate AI Coaching Summary", type="primary"):
            with st.spinner("Generating insights..."):
                # Prepare objectives summary
                obj_summary = "\n".join([f"- {row['Objective_Text']}: {row['Status']} ({row['Progress_Pct']}%)"
                                        for _, row in colleague_objectives.iterrows()])

                prompt = get_colleague_summary_prompt(
                    colleague.to_dict(),
                    latest.to_dict(),
                    obj_summary
                )

                response = call_claude(prompt)
                st.markdown(response)


# ============== PAGE: TRENDS & ANALYTICS ==============
def show_trends(colleagues, metrics, targets, combined):
    st.title("📈 Trends & Analytics")
    st.markdown("---")

    # Time series selector
    metric_options = {
        "Quality Score": "Quality_Pct",
        "FCR": "FCR_Pct",
        "CSAT": "CSAT_Pct",
        "NPS": "NPS",
        "AHT": "AHT_Min",
        "Call Volume": "Call_Volume"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_metric = st.selectbox("Select Metric", list(metric_options.keys()),
                                       help="Choose a metric to view trends over time. Hover over metric cards elsewhere in the dashboard for detailed explanations.")

    with col2:
        group_by = st.selectbox("Group By", ["Overall", "Team", "Tenure Band"])

    metric_col = metric_options[selected_metric]

    # Generate trend chart
    if group_by == "Overall":
        trend_data = metrics.groupby('Month')[metric_col].mean().reset_index()
        fig = px.line(trend_data, x='Month', y=metric_col, markers=True,
                     title=f"{selected_metric} Trend - Overall")

    elif group_by == "Team":
        merged = metrics.merge(colleagues[['Colleague_ID', 'Team']], on='Colleague_ID')
        trend_data = merged.groupby(['Month', 'Team'])[metric_col].mean().reset_index()
        fig = px.line(trend_data, x='Month', y=metric_col, color='Team', markers=True,
                     title=f"{selected_metric} Trend by Team")

    else:
        merged = metrics.merge(colleagues[['Colleague_ID', 'Tenure_Band']], on='Colleague_ID')
        trend_data = merged.groupby(['Month', 'Tenure_Band'])[metric_col].mean().reset_index()
        fig = px.line(trend_data, x='Month', y=metric_col, color='Tenure_Band', markers=True,
                     title=f"{selected_metric} Trend by Tenure Band")

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Movers analysis
    st.subheader("Top Movers This Month")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📈 Most Improved**")
        # Calculate month-over-month change
        if len(metrics['Month'].unique()) >= 2:
            months = sorted(metrics['Month'].unique())
            latest = metrics[metrics['Month'] == months[-1]]
            previous = metrics[metrics['Month'] == months[-2]]

            merged = latest.merge(previous, on='Colleague_ID', suffixes=('_now', '_prev'))
            merged['Change'] = merged[f'{metric_col}_now'] - merged[f'{metric_col}_prev']

            if metric_col == 'AHT_Min':
                top_improved = merged.nsmallest(5, 'Change')
            else:
                top_improved = merged.nlargest(5, 'Change')

            for _, row in top_improved.iterrows():
                colleague = colleagues[colleagues['Colleague_ID'] == row['Colleague_ID']].iloc[0]
                st.write(f"**{colleague['Name']}** ({colleague['Team']}): {row['Change']:+.1f}")

    with col2:
        st.markdown("**📉 Needs Support**")
        if len(metrics['Month'].unique()) >= 2:
            if metric_col == 'AHT_Min':
                needs_support = merged.nlargest(5, 'Change')
            else:
                needs_support = merged.nsmallest(5, 'Change')

            for _, row in needs_support.iterrows():
                colleague = colleagues[colleagues['Colleague_ID'] == row['Colleague_ID']].iloc[0]
                st.write(f"**{colleague['Name']}** ({colleague['Team']}): {row['Change']:+.1f}")


# ============== PAGE: STRUGGLING COLLEAGUES ==============
def show_struggling_colleagues(colleagues, metrics, targets, objectives, combined):
    st.title("⚠️ Colleagues Needing Support")
    st.markdown("---")

    # Filter to struggling colleagues (Focus or Below status)
    struggling = combined[combined['Performance_Status'].isin(['Focus', 'Below'])].sort_values('Performance_Score')

    st.write(f"**{len(struggling)} colleagues currently need additional support**")

    for _, row in struggling.iterrows():
        colleague_metrics = metrics[metrics['Colleague_ID'] == row['Colleague_ID']].sort_values('Month')
        colleague_objectives = objectives[objectives['Colleague_ID'] == row['Colleague_ID']]

        status_color = get_status_color(row['Performance_Status'])

        with st.expander(f"🔴 {row['Name']} - Score: {row['Performance_Score']:.1f}/100 ({row['Performance_Status']})"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write(f"**Team:** {row['Team']}")
                st.write(f"**Tenure:** {row['Tenure_Band']} ({row['Tenure_Months']}mo)")
                st.write(f"**Coaching Priority:** {row['Coaching_Priority']}")

                if row['Risk_Flags']:
                    st.write("**Risk Flags:**")
                    for risk in row['Risk_Flags']:
                        st.write(f"- ⚠️ {risk}")

            with col2:
                # Mini trend chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=colleague_metrics['Month'], y=colleague_metrics['Quality_Pct'],
                                        mode='lines+markers', name='Quality'))
                fig.update_layout(height=200, margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            # AI Analysis button
            if st.button(f"Get AI Analysis", key=f"ai_{row['Colleague_ID']}"):
                with st.spinner("Analyzing..."):
                    metrics_summary = colleague_metrics.to_string()
                    prompt = get_struggling_analysis_prompt(
                        row.to_dict(),
                        metrics_summary,
                        "Average in tenure band"
                    )
                    response = call_claude(prompt)
                    st.markdown(response)


# ============== PAGE: MANAGER TRAINING HUB ==============
def show_training_hub():
    st.title("📚 Manager Training Hub")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Coaching Basics", "Metric Guide", "Conversation Templates", "Best Practices"])

    with tab1:
        st.subheader("Coaching Fundamentals")

        st.markdown("""
        ### The GROW Model for Coaching Conversations

        **G - Goal**: What do you want to achieve?
        - "What would success look like for you this month?"
        - "Where do you want to be with your quality scores?"

        **R - Reality**: Where are you now?
        - "Let's look at your current performance data together"
        - "What's been working well? What's been challenging?"

        **O - Options**: What could you do?
        - "What approaches have you tried?"
        - "What else might work?"

        **W - Will**: What will you do?
        - "What specific actions will you take?"
        - "When will you start? How will we measure progress?"

        ---

        ### Coaching by Tenure Band

        **Attaining Foundation (0-3 months)**
        - Focus on foundational skills and confidence
        - Provide frequent feedback and encouragement
        - Pair with experienced mentors
        - Celebrate small wins

        **Attaining Competence (4-12 months)**
        - Build independence gradually
        - Focus on handling complex scenarios
        - Introduce stretch targets
        - Encourage peer learning

        **Maintaining Competence (13-24 months)**
        - Focus on consistency and reliability
        - Address any emerging bad habits
        - Develop specialist skills
        - Prepare for mentoring role

        **Maintaining Excellence (25+ months)**
        - Focus on leadership and mentoring
        - Involve in process improvement
        - Stretch with challenging cases
        - Recognise expertise publicly
        """)

    with tab2:
        st.subheader("Understanding Your Metrics")

        metrics_guide = {
            "Quality Score": {
                "What it measures": "Overall quality of customer interactions based on QA evaluation",
                "Why it matters": "Ensures consistent, compliant service delivery",
                "How to improve": "Listen to call recordings, identify specific areas, practice techniques"
            },
            "FCR (First Call Resolution)": {
                "What it measures": "Percentage of calls resolved without customer needing to call back",
                "Why it matters": "Higher FCR = better customer experience and operational efficiency",
                "How to improve": "Thorough needs analysis, complete resolution, clear next steps"
            },
            "CSAT": {
                "What it measures": "Customer satisfaction rating from post-call surveys",
                "Why it matters": "Direct measure of customer perception and loyalty",
                "How to improve": "Empathy, ownership, setting expectations, following up"
            },
            "AHT (Average Handle Time)": {
                "What it measures": "Average total time spent on calls including hold and wrap",
                "Why it matters": "Balances efficiency with quality - not just about being fast",
                "How to improve": "System proficiency, call control techniques, efficient documentation"
            },
            "NPS": {
                "What it measures": "Customer likelihood to recommend (scale -100 to +100)",
                "Why it matters": "Predicts customer loyalty and business growth",
                "How to improve": "Exceed expectations, personal connection, memorable service"
            }
        }

        for metric, details in metrics_guide.items():
            with st.expander(metric):
                st.write(f"**What it measures:** {details['What it measures']}")
                st.write(f"**Why it matters:** {details['Why it matters']}")
                st.write(f"**How to improve:** {details['How to improve']}")

    with tab3:
        st.subheader("1:1 Conversation Templates")

        st.markdown("""
        ### Opening a Performance Conversation

        **Positive start:**
        > "Thanks for meeting with me. I wanted to check in on how things are going and look at some of your recent performance data together. How are you feeling about work at the moment?"

        **Addressing concerns:**
        > "I've noticed some changes in your [metric] recently. I wanted to understand what might be going on and see how I can support you."

        ---

        ### Discussing Specific Metrics

        **Quality concerns:**
        > "Your quality score has dropped to [X%] this month. Let's listen to a couple of calls together and identify what's happening. What do you think might be contributing to this?"

        **FCR improvements:**
        > "Great news - your FCR has improved by [X%]! What have you been doing differently? This is something we could share with the team."

        ---

        ### Setting Action Items

        > "Based on our conversation, let's agree on 2-3 specific things you'll focus on before our next 1:1. What feels most important to you?"

        > "I'll commit to [support action]. Let's check in on [date] to see how you're progressing."

        ---

        ### Closing Strong

        > "Thanks for being open about this. I'm confident you can turn this around, and I'm here to support you. Is there anything else you need from me?"
        """)

    with tab4:
        st.subheader("Coaching Best Practices")

        st.markdown("""
        ### Do's ✅

        - **Be specific**: Use data and examples, not generalisations
        - **Listen first**: Understand their perspective before offering solutions
        - **Focus forward**: What can we change, not what went wrong
        - **Recognise effort**: Acknowledge improvements, even small ones
        - **Follow up**: Check in on agreed actions
        - **Document**: Keep records of coaching conversations

        ---

        ### Don'ts ❌

        - **Don't surprise**: Address issues in 1:1s, not in public
        - **Don't overload**: Focus on 1-2 priority areas
        - **Don't compare to others**: Compare to their own targets
        - **Don't just tell**: Ask questions to guide self-discovery
        - **Don't ignore context**: Consider tenure, circumstances, workload
        - **Don't skip praise**: Balanced feedback is more effective

        ---

        ### Using This Dashboard for Coaching

        1. **Before the 1:1**: Review the colleague's dashboard page
        2. **Identify patterns**: Look at trends, not just single months
        3. **Check objectives**: Are they on track for yearly goals?
        4. **Use AI insights**: Generate coaching suggestions
        5. **During the 1:1**: Share screen to review data together
        6. **After the 1:1**: Document agreed actions
        """)


# ============== PAGE: AI COACH CHATBOT ==============
def show_ai_coach(colleagues, metrics, targets, objectives, benchmarks, combined):
    st.title("🤖 AI Performance Coach")
    st.markdown("Ask me anything about colleague performance, coaching strategies, or get AI-powered insights.")
    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about performance, coaching, or get insights..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                # Check if a specific colleague is mentioned
                mentioned_colleagues = []
                for _, row in colleagues.iterrows():
                    first_name = row['Name'].split()[0].lower()
                    full_name = row['Name'].lower()
                    if first_name in prompt.lower() or full_name in prompt.lower():
                        mentioned_colleagues.append(row['Colleague_ID'])

                # Build comprehensive context
                context_parts = []

                # Team summary
                context_parts.append(f"""
TEAM SUMMARY:
- Total colleagues: {len(colleagues)}
- Teams: {', '.join(colleagues['Team'].unique())}
- Average performance score: {combined['Performance_Score'].mean():.1f}
- Colleagues needing support: {len(combined[combined['Performance_Status'].isin(['Focus', 'Below'])])}
""")

                # If specific colleagues mentioned, include their FULL details
                if mentioned_colleagues:
                    context_parts.append("\n=== DETAILED DATA FOR MENTIONED COLLEAGUES ===\n")
                    for cid in mentioned_colleagues:
                        colleague_row = colleagues[colleagues['Colleague_ID'] == cid].iloc[0]
                        colleague_combined = combined[combined['Colleague_ID'] == cid].iloc[0]
                        colleague_metrics = metrics[metrics['Colleague_ID'] == cid].sort_values('Month')
                        colleague_objectives = objectives[objectives['Colleague_ID'] == cid]
                        target_row = targets[targets['Tenure_Band'] == colleague_row['Tenure_Band']].iloc[0]

                        context_parts.append(f"""
--- {colleague_row['Name']} ({cid}) ---
Profile:
  - Team: {colleague_row['Team']}
  - Tenure: {colleague_row['Tenure_Months']} months ({colleague_row['Tenure_Band']})
  - Start Date: {colleague_row['Start_Date']}
  - Overall Performance Score: {colleague_combined['Performance_Score']:.1f}/100
  - Performance Status: {colleague_combined['Performance_Status']}
  - Coaching Priority: {colleague_combined['Coaching_Priority']}

Latest Month Metrics (December 2024):
  - Quality Score: {colleague_combined['Quality_Pct']}% (Target: {target_row['Quality_Target']}%)
  - FCR: {colleague_combined['FCR_Pct']}% (Target: {target_row['FCR_Target']}%)
  - CSAT: {colleague_combined['CSAT_Pct']}% (Target: {target_row['CSAT_Target']}%)
  - NPS: {colleague_combined['NPS']} (Target: {target_row['NPS_Target']})
  - AHT: {colleague_combined['AHT_Min']} min (Target: {target_row['AHT_Target']} min)
  - Hold Time: {colleague_combined['Hold_Min']} min (Target: {target_row['Hold_Target']} min)
  - ACW: {colleague_combined['ACW_Min']} min (Target: {target_row['ACW_Target']} min)
  - Adherence: {colleague_combined['Adherence_Pct']}% (Target: {target_row['Adherence_Target']}%)
  - Critical Errors: {colleague_combined['Critical_Errors']}
  - Complaint Rate: {colleague_combined['Complaint_Rate']} per 1000 calls
  - Transfer Rate: {colleague_combined['Transfer_Pct']}%
  - Repeat Call Rate: {colleague_combined['Repeat_Call_Pct']}%
  - Sentiment Score: {colleague_combined['Sentiment_Score']}
  - Training Hours: {colleague_combined['Training_Hours']}
  - Coaching Actions Open: {colleague_combined['Coaching_Open']}
  - Coaching Actions Closed: {colleague_combined['Coaching_Closed']}

3-Month Trend Data:
""")
                        for _, m in colleague_metrics.iterrows():
                            month_str = m['Month'].strftime('%Y-%m') if hasattr(m['Month'], 'strftime') else str(m['Month'])[:7]
                            context_parts.append(f"  {month_str}: Quality={m['Quality_Pct']}%, FCR={m['FCR_Pct']}%, CSAT={m['CSAT_Pct']}%, AHT={m['AHT_Min']}min, NPS={m['NPS']}, Errors={m['Critical_Errors']}")

                        context_parts.append("\nYearly Objectives:")
                        for _, obj in colleague_objectives.iterrows():
                            context_parts.append(f"  - {obj['Objective_Text']}: {obj['Status']} ({obj['Progress_Pct']}% complete)")
                        context_parts.append("")

                # All colleagues summary with key metrics
                context_parts.append("\n=== ALL COLLEAGUES - KEY METRICS (Latest Month) ===")
                metrics_cols = ['Name', 'Team', 'Tenure_Band', 'Performance_Score', 'Performance_Status',
                               'Quality_Pct', 'FCR_Pct', 'CSAT_Pct', 'NPS', 'AHT_Min', 'Critical_Errors', 'Coaching_Priority']
                context_parts.append(combined[metrics_cols].to_string())

                # Industry benchmarks
                context_parts.append(f"""

=== INDUSTRY BENCHMARKS (UK Banking) ===
{benchmarks.to_string()}

=== TENURE BAND TARGETS ===
{targets.to_string()}
""")

                full_context = "\n".join(context_parts)
                full_prompt = get_chat_context_prompt(prompt, full_context)
                response = call_claude(full_prompt)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Example questions
    st.markdown("---")
    st.markdown("**Example questions you can ask:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Who needs the most support right now?
        - How is the Card Services team performing?
        - What should I focus on with new starters?
        - Compare our FCR to industry benchmarks
        """)
    with col2:
        st.markdown("""
        - Generate a coaching plan for improving quality
        - Which colleagues are role models I can learn from?
        - What trends should I be concerned about?
        - How can I help someone improve their AHT?
        """)


# ============== MAIN APP ==============
def main():
    # Load data
    colleagues, metrics, targets, objectives, benchmarks, combined = load_all_data()

    # Sidebar navigation
    st.sidebar.image("https://via.placeholder.com/150x50?text=AI+Coach", width=150)
    st.sidebar.title("Navigation")

    pages = {
        "Overview Dashboard": "📊",
        "Colleague Explorer": "👥",
        "Individual View": "👤",
        "Trends & Analytics": "📈",
        "Struggling Colleagues": "⚠️",
        "Manager Training": "📚",
        "AI Coach": "🤖"
    }

    # Check for page in session state
    if 'page' not in st.session_state:
        st.session_state['page'] = "Overview Dashboard"

    selected_page = st.sidebar.radio(
        "Select Page",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}",
        index=list(pages.keys()).index(st.session_state['page'])
    )

    st.session_state['page'] = selected_page

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Total Colleagues", len(colleagues), help="Total number of colleagues in the dashboard")
    st.sidebar.metric("Avg Performance", f"{combined['Performance_Score'].mean():.1f}", help=METRIC_TOOLTIPS["Performance_Score"])

    needing_support = len(combined[combined['Performance_Status'].isin(['Focus', 'Below'])])
    st.sidebar.metric("Needing Support", needing_support, help="Colleagues in 'Focus' or 'Below' status who need additional coaching and support")

    # Render selected page
    if selected_page == "Overview Dashboard":
        show_overview_dashboard(colleagues, metrics, targets, benchmarks, combined)
    elif selected_page == "Colleague Explorer":
        show_colleague_explorer(colleagues, metrics, combined)
    elif selected_page == "Individual View":
        show_individual_view(colleagues, metrics, targets, objectives, combined)
    elif selected_page == "Trends & Analytics":
        show_trends(colleagues, metrics, targets, combined)
    elif selected_page == "Struggling Colleagues":
        show_struggling_colleagues(colleagues, metrics, targets, objectives, combined)
    elif selected_page == "Manager Training":
        show_training_hub()
    elif selected_page == "AI Coach":
        show_ai_coach(colleagues, metrics, targets, objectives, benchmarks, combined)


if __name__ == "__main__":
    main()
