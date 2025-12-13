import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add root to path so we can import src
sys.path.append(os.getcwd())

from src.predictor import Predictor
from config import DATASET_PATH, UI_PAGE_TITLE, UI_LAYOUT

st.set_page_config(page_title=UI_PAGE_TITLE, layout=UI_LAYOUT)

# Custom CSS for better responsiveness
st.markdown("""
<style>
    /* Table responsiveness */
    .stDataFrame {
        width: 100%;
    }
    .stDataFrame table {
        width: 100% !important;
    }
    .stDataFrame td, .stDataFrame th {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
        overflow-wrap: break-word !important;
    }
    
    /* Metric styling with auto-sizing */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        min-height: 80px;
    }
    .stMetric label {
        font-size: 14px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 20px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        font-size: 12px;
    }
    
    /* Column text wrapping */
    div[data-testid="column"] {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

from src.predictor import Predictor
from src.explainer import Explainer
# Config already imported at top, verified.
# st.set_page_config already called at top.

# Cache loaders
@st.cache_resource(ttl=3600)


# ... (omitted imports)



@st.cache_resource
def load_explainer_v4(df, titles=None):
    return Explainer(df, known_titles=titles)


# --- CACHE CLEARING UTILITY ---
def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

@st.cache_resource(ttl=3600)
@st.cache_resource
def load_predictor_v7():
    return Predictor()

# ...

@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

def main():
    st.title("🚀 AI Appointer Assist")
    st.markdown("AI-powered system for recommending next-best assignments based on career history.")
    
    # Load Resources
    with st.spinner("Loading AI Models..."):
        predictor = load_predictor_v7()
        df = load_data()
        
        # Extract known titles for fuzzy normalization
        # Combine historical titles AND valid target roles to ensure full coverage
        h_titles = list(predictor.transition_stats['title_trans'].keys()) if hasattr(predictor, 'transition_stats') else []
        t_titles = list(predictor.valid_roles) if hasattr(predictor, 'valid_roles') else []
        known_titles = list(set(h_titles + t_titles))
        
        explainer = load_explainer_v4(df, known_titles)
        
    # Navigation
    mode = st.sidebar.radio("Mode", ["Employee Lookup", "Simulation", "Billet Lookup", "Branch Analytics"])
    
    # ... (Cache control) ...
    
    # Rank Flexibility Control (Global)
    with st.expander("⚙️ Advanced Settings (Rank Flexibility)"):
        rank_flex_up = st.slider("Rank Flexibility (Up)", 0, 2, 0, help="Allow promotion of N levels from predicted rank")
        rank_flex_down = st.slider("Rank Flexibility (Down)", 0, 2, 0, help="Allow demotion of N levels from predicted rank")
    
    if st.sidebar.button("🔄 Reload Models & Cache", help="Click if recent updates are not showing"):
        clear_cache()
        st.rerun()
    
    # =========================================================================
    # MODE 1: EMPLOYEE LOOKUP (FILTERED ITERATION)
    # =========================================================================
    if mode == "Employee Lookup":
        st.header("Search & Predict")
        
        # 1. Filters
        st.markdown("### 🔍 Filter Officers")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        all_ranks = sorted(df['Rank'].unique())
        all_branches = sorted(df['Branch'].unique())
        all_entries = sorted(df['Entry_type'].unique())
        
        with col_f1:
            sel_rank = st.selectbox("Rank", ["All"] + list(all_ranks))
        with col_f2:
            sel_branch = st.selectbox("Branch", ["All"] + list(all_branches))
        with col_f3:
            sel_entry = st.selectbox("Entry Type", ["All"] + list(all_entries))
            
        # 2. Filter Data
        filtered_df = df.copy()
        if sel_rank != "All": filtered_df = filtered_df[filtered_df['Rank'] == sel_rank]
        if sel_branch != "All": filtered_df = filtered_df[filtered_df['Branch'] == sel_branch]
        if sel_entry != "All": filtered_df = filtered_df[filtered_df['Entry_type'] == sel_entry]
        
        count = len(filtered_df)
        st.success(f"Found {count} officers matching criteria.")
        
        if count > 0:
            # 3. Iteration Controls
            if 'curr_idx' not in st.session_state: st.session_state.curr_idx = 0
            
            # Reset index if filters change (naive check: if index out of bounds)
            if st.session_state.curr_idx >= count: st.session_state.curr_idx = 0
            
            col_prev, col_stat, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("⬅️ Previous"):
                    st.session_state.curr_idx = (st.session_state.curr_idx - 1) % count
            with col_next:
                if st.button("Next ➡️"):
                    st.session_state.curr_idx = (st.session_state.curr_idx + 1) % count
            with col_stat:
                st.markdown(f"<div style='text-align: center; font-weight: bold; padding-top: 10px;'>Officer {st.session_state.curr_idx + 1} of {count}</div>", unsafe_allow_html=True)
                
            # 4. Display Current Officer
            row = filtered_df.iloc[st.session_state.curr_idx]
            
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rank", row['Rank'])
                st.metric("Branch", row['Branch'])
            with col2:
                st.metric("Current Role", row['current_appointment'])
                st.metric("Pool", row['Pool'])
            with col3:
                st.metric("ID", row['Employee_ID'])
                st.text(f"Entry: {row['Entry_type']}")
                
            # History
            with st.expander("📜 Career History", expanded=False):
                st.text_area("History", row['Appointment_history'], height=100, disabled=True)

            # 5. AUTO-PREDICT
            st.markdown("### 🔮 Predicted Trajectory")
            try:
                # Convert Series to DF for predictor
                input_df = pd.DataFrame([row])
                results = predictor.predict(input_df, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                
                # Check top confidence
                top_conf = results.iloc[0]['Confidence']
                
                # Dynamic Alert
                if top_conf > 0.4:
                    st.success(f"Strong Recommendation: **{results.iloc[0]['Prediction']}**")
                else:
                    st.warning(f"Weak Signal ({top_conf:.1%}). Model suggests: **{results.iloc[0]['Prediction']}**")

                st.subheader(f"Top 10 Recommendations")
                
                # Interactive List with Explainability
                # Logic mirrors Billet Lookup
                
                for idx, r_row in results.head(10).iterrows():
                    score = r_row['Confidence']
                    score_color = "🟢" if score > 0.5 else "🟡" if score > 0.1 else "⚪"
                    role_name = r_row['role'] if 'role' in r_row else r_row['Prediction']
                    
                    with st.expander(f"{score_color} {score:.1%} | {role_name}"):
                         # 1. Breakdown Chart
                        st.markdown("#### 📊 Why this recommendation?")
                        feats = r_row.get('_Feats', {})
                        if feats:
                            
                            # Extract XAI Contribs
                            contribs = r_row.get('_Contribs', None)
                            
                            metrics = explainer.format_feature_explanation(feats, score=top_conf, constraints=predictor.constraints, contribs=contribs)
                            # Create comparison chart
                            c1, c2, c3, c4, c5 = st.columns(5)
                            
                            m_ai = metrics.get('AI Score', {'value': '0%', 'desc': 'No Data'})
                            c1.metric("AI Score", m_ai['value'], help=m_ai['desc'])
                            
                            m_hist = metrics.get('History Strength', {'value': '0%', 'desc': 'No Data'})
                            c2.metric("History Match", m_hist['value'], help=m_hist['desc'])
                            
                            m_train = metrics.get('Training Match', {'value': '-', 'desc': 'No Data'})
                            c3.metric("Training", m_train['value'], help=m_train['desc'])
                            
                            m_bp = metrics.get('Branch & Pool Fit', {'value': '-', 'desc': 'No Data'})
                            c4.metric("Branch & Pool", m_bp['value'], help=m_bp['desc'])
                            
                            m_re = metrics.get('Rank & Entry Fit', {'value': '-', 'desc': 'No Data'})
                            c5.metric("Rank & Entry", m_re['value'], help=m_re['desc'])
                            
                            # Deep Dive XAI
                            if contribs:
                                with st.expander("🔍 Deep Dive Analysis (XAI)"):
                                    st.markdown("This chart shows exactly how each feature pushed the score up (Green) or down (Red).")
                                    # Pass Base Value for accurate Waterfall starting point
                                    # This answers "Why is score -1.46?" vs "33%" (Raw vs Norm)
                                    base_v = r_row.get('_BaseVal', 0.0)
                                    fig = explainer.create_shap_waterfall(contribs, base_value=base_v, feats=feats)
                                    st.plotly_chart(fig, use_container_width=True, key=f"xai_chart_{int(score*100)}_{role_name}")

                        # 2. Historical Precedents
                        st.markdown("#### 📜 Historical Precedents")
                        
                        # Use exact title from Predictor context (matches Tooltip)
                        ctx = feats.get('_Context', {})
                        context_title = ctx.get('From_Title')
                        
                        # Fallback to row if context missing
                        curr_title = context_title if context_title else row.get('current_appointment', row.get('last_role_title'))
                        
                        if curr_title:
                            precedents = explainer.get_precedents(curr_title, role_name)
                            if precedents:
                                st.write(f"Officers who moved from **{curr_title}** to **{role_name}**:")
                                cols = ['Employee_ID', 'Rank', 'Name', 'Branch', 'Pool', 'Entry_type', 'Appointment_history', 'Training_history']
                                st.dataframe(pd.DataFrame(precedents)[cols], hide_index=True)
                            else:
                                st.caption("No exact historical precedents found for this specific transition.")
                
            except Exception as e:
                st.error(f"Prediction unavailable: {e}")
                
        else:
            st.warning("No officers match. Adjust filters.")

    # =========================================================================
    # MODE 4: BRANCH ANALYTICS (DASHBOARD)
    # =========================================================================
    elif mode == "Branch Analytics":
        st.header("📊 Workforce Flow Analytics")
        import plotly.graph_objects as go
        
        # Tabs for Dashboard
        tab_flow, tab_ai = st.tabs(["🔀 Flow Chart (Sankey)", "🧠 AI Promo Drivers (Global XAI)"])
        
        with tab_flow:
            # Rich Sankey: Entry -> Rank -> Branch -> Pool
            
            # 1. Controls
            all_branches = sorted(df['Branch'].unique())
            sel_branches = st.multiselect("Select Branches to Visualize", all_branches, default=all_branches[:3])
            
            if not sel_branches:
                st.warning("Please select at least one branch.")
            else:
                sankey_df = df[df['Branch'].isin(sel_branches)]
                
                if len(sankey_df) > 1000:
                    st.info(f"Visualizing random sample of 1000 officers from {len(sankey_df)} total to improve performance.")
                    sankey_df = sankey_df.sample(1000)
                    
                # 2. Aggregations
                # Flow 1: Entry -> Rank
                f1 = sankey_df.groupby(['Entry_type', 'Rank']).size().reset_index(name='count')
                # Flow 2: Rank -> Branch
                f2 = sankey_df.groupby(['Rank', 'Branch']).size().reset_index(name='count')
                # Flow 3: Branch -> Pool (Top 10 Pools)
                top_pools = sankey_df['Pool'].value_counts().head(10).index.tolist()
                f3 = sankey_df[sankey_df['Pool'].isin(top_pools)].groupby(['Branch', 'Pool']).size().reset_index(name='count')
                
                # Nodes
                labels = []
                labels.extend(sorted(f1['Entry_type'].unique())) # 0..A
                entry_end = len(labels)
                
                labels.extend(sorted(f1['Rank'].unique())) # A..B
                rank_end = len(labels)
                
                labels.extend(sorted(f2['Branch'].unique())) # B..C
                branch_end = len(labels)
                
                labels.extend(sorted(f3['Pool'].unique())) # C..D
                
                label_map = {l: i for i, l in enumerate(labels)}
                
                sources = []
                targets = []
                values = []
                colors = []
                
                # Link Generation
                # 1: Entry -> Rank
                for _, r in f1.iterrows():
                    sources.append(label_map[r['Entry_type']])
                    targets.append(label_map[r['Rank']])
                    values.append(r['count'])
                    colors.append("rgba(31, 119, 180, 0.3)")
                    
                # 2: Rank -> Branch
                for _, r in f2.iterrows():
                    src = r['Rank']
                    tgt = r['Branch']
                    if src in label_map and tgt in label_map:
                        sources.append(label_map[src])
                        targets.append(label_map[tgt])
                        values.append(r['count'])
                        colors.append("rgba(255, 127, 14, 0.3)")
                        
                # 3: Branch -> Pool
                for _, r in f3.iterrows():
                    src = r['Branch']
                    tgt = r['Pool']
                    if src in label_map and tgt in label_map:
                        sources.append(label_map[src])
                        targets.append(label_map[tgt])
                        values.append(r['count'])
                        colors.append("rgba(44, 160, 44, 0.3)")
                
                fig = go.Figure(data=[go.Sankey(
                    node = dict(
                      pad = 15,
                      thickness = 20,
                      line = dict(color = "black", width = 0.5),
                      label = labels,
                      color = "lightgray"
                    ),
                    link = dict(
                      source = sources,
                      target = targets,
                      value = values,
                      color = colors
                  ))])
                  
                fig.update_layout(title_text="Workforce Flow: Entry -> Rank -> Branch -> Pool", font_size=10, height=600)
                st.plotly_chart(fig)
        
        with tab_ai:
            st.markdown("### 🧠 What drives career patterns?")
            st.info("The AI model learns different rules for different groups. Use the filters below to investigate the 'Laws of Promotion' for specific cohorts.")
            
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                ai_branch = st.selectbox("Branch", ["All"] + sorted(df['Branch'].unique()))
            with c_f2:
                ai_pool = st.selectbox("Pool", ["All"] + sorted(df['Pool'].unique()))
            with c_f3:
                ai_entry = st.selectbox("Entry Type", ["All"] + sorted(df['Entry_type'].unique()))

            if st.button("Run Global SHAP Analysis"):
                filter_desc = f"{ai_branch}/{ai_pool}/{ai_entry}"
                with st.spinner(f"Computing drivers for {filter_desc}..."):
                     # Call get_global_context with all filters
                     X_global = predictor.get_global_context(
                         n=100, 
                         branch_filter=ai_branch if ai_branch != "All" else None,
                         pool_filter=ai_pool if ai_pool != "All" else None,
                         entry_filter=ai_entry if ai_entry != "All" else None
                     )
                     
                     if X_global is not None:
                         expl_obj = predictor.xai.get_explanation_object(X_global)
                         
                         c1, c2 = st.columns(2)
                         with c1:
                             st.markdown("**Feature Impact (Beeswarm)**")
                             fig_bee = explainer.create_global_beeswarm_plot(expl_obj)
                             st.pyplot(fig_bee)
                         with c2:
                             st.markdown("**Feature Importance (Bar)**")
                             fig_bar = explainer.create_global_bar_plot(expl_obj)
                             st.pyplot(fig_bar)
                     else:
                         st.error(f"Insufficient data found for filter combination: {filter_desc}. Try relaxing constraints.")
            
    elif mode == "Simulation":
        st.header("🧪 Experiments Lab")
        st.markdown("Design hypothetical officers, tweak parameters, and analyze how the AI reacts to specific career profiles.")
        
        # Layout: Control Panel (Left), Results (Right)
        col_inputs, col_results = st.columns([1, 1.2])
        
        with col_inputs:
            st.markdown("### 1. Officer Profile")
            c1, c2 = st.columns(2)
            with c1:
                rank = st.selectbox("Rank", sorted(df['Rank'].unique()), index=sorted(df['Rank'].unique()).index('Lieutenant') if 'Lieutenant' in df['Rank'].unique() else 0)
                branch = st.selectbox("Branch", sorted(df['Branch'].unique()))
            with c2:
                pool = st.selectbox("Pool", sorted(df['Pool'].unique()))
                entry = st.selectbox("Entry Type", sorted(df['Entry_type'].unique()))
            
            st.markdown("### 2. Career Markers")
            years_rank = st.slider("Time in Current Rank (Years)", 0.0, 10.0, 3.5, 0.5, help="Directly affects 'years_in_current_rank' feature.")
            total_service = st.slider("Total Service (Years)", 0, 40, 8, help="Affects 'years_service' feature.")
            
            st.markdown("### 3. History & Qualifications")
            
            # Smart Role Selector (Context Aware)
            import json
            with open('models/all_constraints.json') as f:
                constraints = json.load(f)
            
            valid_roles = []
            for role_name, role_const in constraints.items():
                if rank in role_const.get('ranks', []) and branch in role_const.get('branches', []):
                    valid_roles.append(role_name)
            
            if not valid_roles: valid_roles = [f"Generic {rank} Role"]
            
            last_role = st.selectbox("Current Job Title", sorted(valid_roles), help="The text of the title matters for 'Title Similarity'.")
            
            common_training = [
                "Command School", "Advanced Tactical", "Warp Theory", 
                "Diplomatic Protocol", "Bridge Officer Test", "Strategic Operations", 
                "First Contact Procedures", "Engineering Classification", "Medical Administration",
                "Department Head Course", "Executive Officer Course"
            ]
            
            training = st.multiselect("Completed Training", common_training, default=["Bridge Officer Test", "Advanced Tactical"])
            
            
            run_btn = st.button("🚀 Run Prediction Analysis", type="primary", use_container_width=True)
            
            # --- PREDICTION LOGIC (Session State) ---
            if run_btn:
                # --- SYNTHESIZE DATA ---
                now = datetime.now()
                appt_start = (now - timedelta(days=365*years_rank))
                appt_str = appt_start.strftime("%d %b %Y").upper()
                appt_hist = f"{last_role} ({appt_str} - )"
                train_hist_parts = [f"{t} (01 JAN 2020 - 01 FEB 2020)" for t in training]
                train_hist = ", ".join(train_hist_parts)
                
                dummy_row = {
                    'Employee_ID': [999999], 'Rank': [rank], 'Branch': [branch], 'Pool': [pool],
                    'Entry_type': [entry], 'Appointment_history': [appt_hist], 'Training_history': [train_hist],
                    'current_appointment': [last_role], 'appointed_since': [appt_start.strftime("%d/%m/%Y")],
                    'DOB': ["01/01/1980"]
                }
                dummy_df = pd.DataFrame(dummy_row)
                
                with st.spinner("Analyzing patterns..."):
                    try:
                        results = predictor.predict(dummy_df, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                        st.session_state['lab_results'] = results
                        st.session_state['lab_dummy_df'] = dummy_df
                    except Exception as e:
                        st.error(f"Simulation Failed: {e}")
            
            # Show Results List in Left Column if they exist
            if 'lab_results' in st.session_state and not st.session_state['lab_results'].empty:
                st.divider()
                st.markdown("### Top Recommendations")
                
                res_df = st.session_state['lab_results']
                # Create a display label for selectbox
                res_df['Label'] = res_df.apply(lambda x: f"{x['Confidence']:.1%} | {x.get('role', x.get('Prediction', 'Unknown'))}", axis=1)
                
                # Selection
                sel_idx = st.selectbox(
                    "Select Role to Analyze", 
                    options=res_df.index,
                    format_func=lambda i: res_df.loc[i, 'Label'],
                    index=0
                )
                
                st.dataframe(res_df[['Prediction', 'Confidence']], hide_index=True, use_container_width=True)

            else:
                sel_idx = None
            
        with col_results:
            st.markdown("### 🔬 Analysis Console")
            
            if sel_idx is not None and 'lab_results' in st.session_state:
                results = st.session_state['lab_results']
                top_res = results.loc[sel_idx]
                
                score = top_res['Confidence']
                pred_role = top_res.get('role', top_res.get('Prediction', 'Unknown'))
                
                # 1. Score Card
                st.success(f"**Recommendation**: {pred_role}")
                sc_col1, sc_col2 = st.columns(2)
                sc_col1.metric("Confidence", f"{score:.1%}")
                sc_col2.metric("Raw AI Score", f"{top_res.get('_RawScore', 0):.2f}")
                
                # 2. Explainability
                st.markdown("#### 📊 Decision Factors")
                
                feats = top_res.get('_Feats', {})
                contribs = top_res.get('_Contribs', {})
                
                if contribs:
                    base_v = top_res.get('_BaseVal', 0.0)
                    
                    # TABS: Individual vs Global
                    t_ind, t_glob = st.tabs(["👤 Individual Analysis", "🌍 Global Model Insights"])
                    
                    with t_ind:
                        # A. WATERFALL
                        st.markdown("##### 1. Decision Path (Waterfall)")
                        fig = explainer.create_shap_waterfall(contribs, base_value=base_v, feats=feats)
                        st.plotly_chart(fig, use_container_width=True, key=f"lab_shap_waterfall_{sel_idx}")
                        
                        # B. FORCE PLOT
                        st.markdown("##### 2. Force Analysis")
                        try:
                            force_df = pd.DataFrame([feats])
                            force_shap = pd.DataFrame([contribs])
                            common_cols = force_df.columns.intersection(force_shap.columns)
                            html = explainer.create_force_plot_html(
                                base_value=base_v,
                                shap_values=force_shap[common_cols].values[0],
                                features=force_df[common_cols]
                            )
                            import streamlit.components.v1 as components
                            components.html(html, height=120, scrolling=True)
                        except Exception as e:
                            st.error(f"Force plot error: {e}")
                    
                    with t_glob:
                        st.markdown("**How does the model think generally?**")
                        X_global = predictor.get_global_context(n=100)
                        if X_global is not None:
                            expl_obj = predictor.xai.get_explanation_object(X_global)
                            st.markdown("##### 1. Feature Impact (Beeswarm)")
                            fig_bee = explainer.create_global_beeswarm_plot(expl_obj)
                            st.pyplot(fig_bee)
                            st.markdown("##### 2. Feature Importance (Bar)")
                            fig_bar = explainer.create_global_bar_plot(expl_obj)
                            st.pyplot(fig_bar)
                        else:
                            st.error("Could not load global context.")

                    # 3. Lab Notes
                    st.divider()
                    st.markdown("#### 📝 Lab Notes")
                    suggestions = []
                    
                    if feats.get('years_in_current_rank', 0) < 2.0 and score < 0.5:
                        suggestions.append(f"⏱️ **Tenure**: Low time in rank ({years_rank} yrs) is likely hurting the score.")
                    
                    if feats.get('prior_title_prob', 0) == 0:
                        suggestions.append(f"📜 **History**: The jump from '{last_role}' to '{pred_role}' has NO historical precedent.")
                    else:
                        suggestions.append(f"📜 **History**: Valid historical path found ({feats.get('prior_title_prob',0):.1%}).")
                        
                    if feats.get('branch_match') == 0:
                        suggestions.append("🔀 **Branch**: Cross-branch move detected.")
                        
                    for note in suggestions:
                        st.info(note)
                        
                else:
                    st.warning("No XAI data available.")
            else:
                st.info("👈 Configure your officer profile on the left and click 'Run Prediction Analysis' to see results.")
            
    elif mode == "Billet Lookup":
        st.header("Find Candidates for Role")
        st.markdown("Reverse search: Select a target role to find the best fit officers.")
        
        all_roles = sorted(predictor.target_encoder.classes_)
        target_role = st.selectbox("Target Appointment", all_roles)
        
        # Show constraints for this role if available
        constraints = predictor.constraints.get(target_role, {})
        allowed_branches = constraints.get('branches', [])
        allowed_ranks = constraints.get('ranks', [])
        
        if allowed_branches:
            st.info(f"Typically held by: **{', '.join(allowed_branches)}** Branch")
        if allowed_ranks:
            st.info(f"Typically held by Ranks: **{', '.join(allowed_ranks)}**")
            
        if st.button("Find Top Candidates"):
            with st.spinner(f"Scanning workforce for '{target_role}'..."):
                # Use the new predict_for_role method
                candidates = df.copy()
                
                # Pre-filter by branch to reduce computation
                if allowed_branches:
                    candidates = candidates[candidates['Branch'].isin(allowed_branches)]
                
                if candidates.empty:
                    st.warning("No candidates found matching Branch requirements.")
                else:
                    # Get confidence for this specific role across all candidates
                    match_df = predictor.predict_for_role(candidates, target_role, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                    
                    if not match_df.empty:
                        # Limit to top 20
                        match_df = match_df.head(20)
                        
                        st.subheader(f"Top Recommended Candidates ({len(match_df)})")
                        
                        # Interactive List with Explainability
                        for idx, row in match_df.iterrows():
                            # Header with Score
                            score = row['Confidence']
                            score_color = "🟢" if score > 0.5 else "🟡" if score > 0.1 else "⚪"
                            
                            with st.expander(f"{score_color} {score:.1%} | {row['Employee_ID']} - {row['Rank']} {row['Name']} ({row['Branch']}) - Currently {row.get('CurrentRole', 'Unknown')}"):
                                # 1. Breakdown Chart
                                st.markdown("#### 📊 Why this recommendation?")
                                feats = row.get('_Feats', {})
                                if feats:
                                    
                                    # Extract XAI Contribs
                                    contribs = row.get('_Contribs', None)
                                    
                                    metrics = explainer.format_feature_explanation(feats, score=score, constraints=predictor.constraints, contribs=contribs)
                                    # Create comparison chart
                                    c1, c2, c3, c4, c5 = st.columns(5)
                                    
                                    m_ai = metrics.get('AI Score', {'value': '0%', 'desc': 'No Data'})
                                    c1.metric("AI Score", m_ai['value'], help=m_ai['desc'])
                                    
                                    m_hist = metrics.get('History Strength', {'value': '0%', 'desc': 'No Data'})
                                    c2.metric("History Match", m_hist['value'], help=m_hist['desc'])
                                    
                                    m_train = metrics.get('Training Match', {'value': '-', 'desc': 'No Data'})
                                    c3.metric("Training", m_train['value'], help=m_train['desc'])
                                    
                                    m_bp = metrics.get('Branch & Pool Fit', {'value': '-', 'desc': 'No Data'})
                                    c4.metric("Branch & Pool", m_bp['value'], help=m_bp['desc'])
                                    
                                    m_re = metrics.get('Rank & Entry Fit', {'value': '-', 'desc': 'No Data'})
                                    c5.metric("Rank & Entry", m_re['value'], help=m_re['desc'])
                                    
                                    # Deep Dive XAI
                                    if contribs:
                                        with st.expander("🔍 Deep Dive Analysis (XAI)"):
                                            st.markdown("This chart shows exactly how each feature pushed the score up (Green) or down (Red).")
                                            
                                            # TABS
                                            xb_t1, xb_t2 = st.tabs(["Waterfall", "Force Plot"])
                                            
                                            with xb_t1:
                                                # Pass Base Value for accurate Waterfall starting point
                                                base_v = row.get('_BaseVal', 0.0)
                                                # Pass feats for tooltips
                                                fig_wf = explainer.create_shap_waterfall(contribs, base_value=base_v, feats=feats)
                                                st.plotly_chart(fig_wf, use_container_width=True, key=f"xai_billet_{idx}_{row['Employee_ID']}")
                                            
                                            with xb_t2:
                                                try:
                                                    force_df = pd.DataFrame([feats])
                                                    force_shap = pd.DataFrame([contribs])
                                                    common_cols = force_df.columns.intersection(force_shap.columns)
                                                    html = explainer.create_force_plot_html(
                                                        base_value=row.get('_BaseVal', 0.0),
                                                        shap_values=force_shap[common_cols].values[0],
                                                        features=force_df[common_cols]
                                                    )
                                                    import streamlit.components.v1 as components
                                                    components.html(html, height=120, scrolling=True)
                                                except Exception as e:
                                                    st.error(f"Force Plot Error: {e}")
                                
                                # 2. Historical Precedents
                                st.markdown("#### 📜 Historical Precedents")
                                
                                # Use exact title from Predictor context (matches Tooltip)
                                ctx = feats.get('_Context', {})
                                curr_clean = ctx.get('From_Title')
                                
                                # Semantic fallback if needed
                                if not curr_clean:
                                    raw = row.get('Appointment_history', '')
                                    if raw:
                                        import re
                                        parts = raw.split('>')
                                        if parts:
                                            # last part
                                            curr_clean = parts[-1].strip()
                                        else:
                                            # try regex on Appointment history
                                            pass
                                
                                if curr_clean:
                                    precedents = explainer.get_precedents(curr_clean, target_role)
                                    if precedents:
                                        st.write(f"Officers who moved from **{curr_clean}** to **{target_role}**:")
                                        cols = ['Employee_ID', 'Rank', 'Name', 'Branch', 'Pool', 'Entry_type', 'Appointment_history', 'Training_history']
                                        st.dataframe(pd.DataFrame(precedents)[cols], hide_index=True)
                                    else:
                                        st.caption("No exact historical precedents found.")

if __name__ == "__main__":
    main()
                                
