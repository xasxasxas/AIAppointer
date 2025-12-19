"""
XAI Helper Functions for Career Progression Patterns

This module provides utilities for extracting and displaying
Markov-based career progression patterns in the UI.
"""

import pandas as pd
import streamlit as st
from src.data_processor import DataProcessor


def extract_career_history(officer_data):
    """
    Extract career history from officer data.
    
    Args:
        officer_data: Dict, Series, or DataFrame row containing officer information
        
    Returns:
        List of role titles in chronological order
    """
    dp = DataProcessor()
    
    # Convert to dict if Series
    if isinstance(officer_data, pd.Series):
        officer_data = officer_data.to_dict()
    
    # Try multiple sources for career history
    career_history = []
    
    # 1. Check for Appointment_history string (most common in raw data)
    if 'Appointment_history' in officer_data:
        appt_hist = officer_data['Appointment_history']
        if appt_hist and not pd.isna(appt_hist):
            appointments = dp.parse_history_column(appt_hist)
            if appointments:
                career_history = [appt['title'] for appt in appointments if appt.get('title')]
    
    # 2. Check for snapshot_history (from transitions)
    if not career_history and 'snapshot_history' in officer_data:
        snapshot = officer_data['snapshot_history']
        if snapshot:
            career_history = [h.get('title', '') for h in snapshot if h.get('title')]
    
    # 3. Fallback to current appointment
    if not career_history and 'current_appointment' in officer_data:
        curr = officer_data.get('current_appointment', '')
        if curr and not pd.isna(curr):
            career_history = [curr]
    
    return career_history


def get_markov_pattern_info(predictor, officer_data, target_role):
    """
    Get Markov progression pattern information for an officer.
    
    Args:
        predictor: Predictor instance with markov_engine
        officer_data: Officer information
        target_role: Target role being predicted
        
    Returns:
        Dict with pattern information or None if no pattern found
    """
    if not predictor.markov_engine:
        return None
    
    # Extract career history
    career_history = extract_career_history(officer_data)
    
    if not career_history or len(career_history) < 1:
        return None
    
    # Get Markov transition info
    info = predictor.markov_engine.get_transition_info(career_history)
    
    # Get probabilities for target role
    probs = predictor.markov_engine.predict_proba(career_history, candidate_roles=[target_role])
    target_prob = probs.get(target_role, 0.0)
    
    return {
        'career_history': career_history,
        'order_used': info.get('order_used', 0),
        'context': info.get('context', ()),
        'context_seen': info.get('context_seen', False),
        'target_probability': target_prob,
        'fallback_reason': info.get('fallback_reason', '')
    }


def display_career_progression_tab(predictor, officer_data, target_role, feature_dict=None):
    """
    Display the Career Progression Pattern tab in XAI.
    
    Args:
        predictor: Predictor instance
        officer_data: Officer information
        target_role: Target role being predicted
        feature_dict: Optional feature dictionary with Markov probabilities
    """
    st.markdown("### 🔄 Career Progression Pattern")
    st.caption("AI-identified career sequence patterns from historical data")
    
    # Get pattern info
    pattern_info = get_markov_pattern_info(predictor, officer_data, target_role)
    
    if not pattern_info or not pattern_info['context_seen']:
        # No pattern found
        st.info("📊 **No specific progression pattern found**")
        st.markdown("""
        This recommendation is based on other factors:
        - Rank compatibility
        - Branch and pool alignment
        - Training and experience
        - Historical transition probabilities
        """)
        
        # Show career history if available
        if pattern_info and pattern_info['career_history']:
            with st.expander("📜 View Career History"):
                st.write("**Career Path:**")
                for i, role in enumerate(pattern_info['career_history'], 1):
                    st.markdown(f"{i}. {role}")
        return
    
    # Pattern found - display it
    order = pattern_info['order_used']
    context = pattern_info['context']
    career_history = pattern_info['career_history']
    target_prob = pattern_info['target_probability']
    
    # Display pattern order
    st.markdown(f"**Pattern Type:** {order}-step sequence")
    
    # Visual progression
    st.markdown("**Recognized Sequence:**")
    
    # Build visual sequence
    sequence_html = '<div style="font-size: 16px; padding: 15px; background: #f0f2f6; border-radius: 8px; margin: 10px 0;">'
    
    for i, role in enumerate(context):
        sequence_html += f'<span style="background: #4CAF50; color: white; padding: 8px 12px; border-radius: 4px; margin: 4px; display: inline-block;">{role}</span>'
        if i < len(context) - 1:
            sequence_html += ' <span style="font-size: 20px; margin: 0 8px;">→</span> '
    
    sequence_html += f' <span style="font-size: 20px; margin: 0 8px;">→</span> '
    sequence_html += f'<span style="background: #2196F3; color: white; padding: 8px 12px; border-radius: 4px; margin: 4px; display: inline-block; font-weight: bold;">{target_role}</span>'
    sequence_html += '</div>'
    
    st.markdown(sequence_html, unsafe_allow_html=True)
    
    # Pattern strength
    st.markdown("**Pattern Strength:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            f"{order}-Step Pattern Probability",
            f"{target_prob * 100:.1f}%",
            help=f"Probability of moving to {target_role} given this {order}-step career sequence"
        )
    
    with col2:
        # Get feature values if available
        if feature_dict:
            markov_avg = feature_dict.get('markov_avg_prob', target_prob)
            st.metric(
                "Overall Markov Confidence",
                f"{markov_avg * 100:.1f}%",
                help="Weighted average across all Markov orders"
            )
    
    # Insight box
    st.success(f"""
    💡 **AI Insight:** Based on analyzing {len(predictor.markov_engine.all_roles)} unique roles 
    across historical career data, officers who followed the sequence **{' → '.join(context)}** 
    typically moved to **{target_role}** next with {target_prob * 100:.1f}% probability.
    """)
    
    # Full career history
    with st.expander("📜 View Full Career History"):
        st.write(f"**Complete Career Path ({len(career_history)} roles):**")
        for i, role in enumerate(career_history, 1):
            is_in_pattern = role in context
            if is_in_pattern:
                st.markdown(f"**{i}. {role}** ⭐ *(part of recognized pattern)*")
            else:
                st.markdown(f"{i}. {role}")


def display_feature_impact_tab(explainer, contribs, base_value, feats, key_suffix=""):
    """
    Display the Feature Impact tab with SHAP waterfall.
    
    Args:
        explainer: ModelExplainer instance
        contribs: SHAP contributions dict
        base_value: Base value for waterfall
        feats: Feature dictionary
        key_suffix: Unique key suffix for plotly chart
    """
    st.markdown("### 📊 Feature Impact Analysis")
    st.caption("How each factor influenced this prediction (Green = positive, Red = negative)")
    
    # Create waterfall chart with unique key
    import time
    unique_key = f"waterfall_{key_suffix}_{int(time.time() * 1000000)}"
    fig = explainer.create_shap_waterfall(contribs, base_value=base_value, feats=feats)
    st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Top factors summary
    st.markdown("**Top Contributing Factors:**")
    
    # Get top 5 positive and negative
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positive Factors** 🟢")
        positive = [(k, v) for k, v in sorted_contribs if v > 0][:3]
        if positive:
            for feat, val in positive:
                st.markdown(f"- **{feat}**: +{val:.3f}")
        else:
            st.caption("None")
    
    with col2:
        st.markdown("**Negative Factors** 🔴")
        negative = [(k, v) for k, v in sorted_contribs if v < 0][:3]
        if negative:
            for feat, val in negative:
                st.markdown(f"- **{feat}**: {val:.3f}")
        else:
            st.caption("None")


def display_xai_section(predictor, explainer, officer_data, target_role, 
                        score, contribs, base_value, feats, mode="employee"):
    """
    Display complete XAI section with all tabs.
    
    Args:
        predictor: Predictor instance
        explainer: ModelExplainer instance
        officer_data: Officer information
        target_role: Target role
        score: Prediction score
        contribs: SHAP contributions
        base_value: Base value
        feats: Feature dictionary
        mode: "employee", "billet", or "simulation"
    """
    with st.expander("🔍 **Deep Dive Analysis (XAI)** - Why this prediction?", expanded=False):
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Feature Impact", "🔄 Career Progression"])
        
        with tab1:
            display_feature_impact_tab(
                explainer, contribs, base_value, feats, 
                key_suffix=f"{mode}_{hash(str(officer_data))}"
            )
        
        with tab2:
            display_career_progression_tab(
                predictor, officer_data, target_role, feats
            )
