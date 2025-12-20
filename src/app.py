import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add root to path so we can import src
sys.path.append(os.getcwd())

from src.predictor import Predictor
from src.xai_helpers import display_xai_section
from config import DATASET_PATH, UI_PAGE_TITLE, UI_LAYOUT


st.set_page_config(page_title=UI_PAGE_TITLE, layout=UI_LAYOUT)

# Custom CSS for professional TalentSync AI styling
st.markdown("""
<style>
    /* === COLOR SCHEME === */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --background-light: #f8f9fa;
        --text-dark: #1a1a2e;
    }
    
    /* === HEADER STYLING === */
    h1 {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: var(--text-dark) !important;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
    }
    
    /* === TABLE RESPONSIVENESS === */
    .stDataFrame {
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame table {
        width: 100% !important;
    }
    .stDataFrame td, .stDataFrame th {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
        overflow-wrap: break-word !important;
        padding: 8px 12px !important;
    }
    .stDataFrame th {
        background-color: var(--background-light) !important;
        font-weight: 600 !important;
    }
    
    /* === METRIC CARDS === */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        min-height: 90px;
    }
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    .stMetric label {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #666 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: var(--primary-color) !important;
    }
    
    /* === SIDEBAR (Light Theme) === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] h1 {
        color: #1f77b4 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 2px 0;
        color: #333 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(31, 119, 180, 0.1);
    }
    
    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 15px !important;
        background-color: var(--background-light) !important;
        border-radius: 8px !important;
    }
    
    /* === BUTTONS === */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1f77b4 0%, #2196f3 100%);
        border: none;
        font-weight: 600;
        padding: 8px 24px;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    /* === INFO/SUCCESS/WARNING BOXES === */
    .stAlert {
        border-radius: 8px !important;
        border-left-width: 4px !important;
    }
    
    /* === TEXT AREAS === */
    .stTextArea textarea {
        font-size: 13px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* === COLUMN LAYOUT === */
    div[data-testid="column"] {
        overflow: visible !important;
    }
    
    /* === DIVIDERS === */
    hr {
        margin: 24px 0 !important;
        border-color: #e0e0e0 !important;
    }
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

from src.explainer import Explainer

# === Cache Functions ===
@st.cache_resource
def load_explainer_v4(df, titles=None):
    return Explainer(df, known_titles=titles)

@st.cache_resource
def load_predictor_v7():
    return Predictor()

@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

def clear_cache():
    """Clear all cached data and resources."""
    st.cache_data.clear()
    st.cache_resource.clear()


# === Main Application ===



def main():
    st.title("🎯 TalentSync AI")
    st.markdown("*AI-powered talent placement system for optimal career progression and role matching.*")
    
    # Load Resources
    with st.spinner("Loading AI Models..."):
        predictor = load_predictor_v7()
        df = load_data()
        
        # Extract known titles for fuzzy normalization
        # Combine historical titles AND valid target roles to ensure full coverage
        h_titles = list(predictor.transition_stats['title_trans'].keys()) if hasattr(predictor, 'transition_stats') else []
        t_titles = list(predictor.valid_roles) if hasattr(predictor, 'valid_roles') else []
        known_titles = list(set(h_titles + t_titles))
    with st.sidebar:
        st.title("🎯 TalentSync AI")
        st.caption(f"v4.0 • {len(df):,} Officers • {len(predictor.constraints):,} Billets")
        
        # Welcome message for first-time users
        if 'first_visit' not in st.session_state:
            st.session_state.first_visit = True
        
        if st.session_state.first_visit:
            with st.expander("👋 Welcome! First time here?", expanded=True):
                st.markdown("""
                **TalentSync AI** uses machine learning to predict optimal career progressions.
                
                **Start with Dashboard** to get an overview!
                """)
                if st.button("Got it! Don't show again"):
                    st.session_state.first_visit = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📍 Navigation")
        
        mode = st.radio(
            "Select Mode", 
            ["🏠 Dashboard", "👤 Employee Lookup", "🎯 Billet Lookup", "🔍 Semantic AI Search", "📊 Analytics & Explorer", "🔄 Simulation", "⚙️ Admin Console"],
            help="Select a mode to navigate between different features",
            label_visibility="collapsed"
        )
        
        # Rank Flexibility (Moved to Sidebar)
        rank_flex_up = 0
        rank_flex_down = 0
        if mode in ["👤 Employee Lookup", "🔄 Simulation", "🎯 Billet Lookup", "🔍 Semantic AI Search"]:
            st.markdown("---")
            with st.expander("Advanced Settings"):
                st.caption("Adjust Rank Flexibility for predictions.")
                rank_flex_up = st.slider(
                    "Flexibility (Up)", 
                    0, 2, 0, 
                    help="Allow predictions for roles up to N ranks above current rank. Higher = more aspirational roles."
                )
                rank_flex_down = st.slider(
                    "Flexibility (Down)", 
                    0, 2, 0, 
                    help="Allow predictions for roles up to N ranks below current rank. Usually kept at 0."
                )
        
        st.markdown("---")
        
        explainer = load_explainer_v4(df, known_titles)
        
    # Navigation
    
    # ... (Cache control) ...
    
    # Rank Flexibility Control (Moved to Sidebar)
    
    if st.sidebar.button("Reload Models & Cache", help="Click if recent updates are not showing"):
        clear_cache()
        st.rerun()
    
    # Dataset Explorer removed (Merged into Analytics & Explorer)

    # =========================================================================
    # MODE 0: DASHBOARD (Home/Landing Page)
    # =========================================================================
    if mode == "🏠 Dashboard":
        st.header("📊 Dashboard")
        st.markdown("Welcome to **TalentSync AI** — Your intelligent talent placement assistant.")
        
        # Key Metrics Row
        st.markdown("### 📈 System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Officers", f"{len(df):,}")
        with col2:
            st.metric("🎯 Available Billets", f"{len(predictor.constraints):,}")
        with col3:
            unique_branches = df['Branch'].nunique()
            st.metric("🌿 Branches", f"{unique_branches}")
        with col4:
            unique_ranks = df['Rank'].nunique()
            st.metric("🎖️ Rank Levels", f"{unique_ranks}")
        
        st.divider()
        
        # Data Breakdown
        st.markdown("### 📋 Data Breakdown")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Officers by Branch**")
            branch_counts = df['Branch'].value_counts()
            st.bar_chart(branch_counts)
        
        with col_b:
            st.markdown("**Officers by Rank**")
            rank_counts = df['Rank'].value_counts()
            st.bar_chart(rank_counts)
        
        st.divider()
        
        # Navigation Guide
        st.markdown("### 🧭 How to Use TalentSync AI")
        st.info("Use the **sidebar navigation** to switch between modes. Here's what each mode does:")
        
        guide_col1, guide_col2 = st.columns(2)
        
        with guide_col1:
            with st.expander("👤 **Employee Lookup** — *Most Common*", expanded=True):
                st.markdown("""
                **Purpose:** Find the best next role for a specific officer.
                
                **When to use:**
                - Annual posting recommendations
                - Career progression planning
                - "What's next for Officer X?"
                
                **How:** Select an officer → See AI-recommended roles with explanations.
                """)
            
            with st.expander("🎯 **Billet Lookup** — *Vacancy Filling*"):
                st.markdown("""
                **Purpose:** Find the best candidates for an open position.
                
                **When to use:**
                - Filling a vacant billet
                - Finding qualified officers for a role
                - "Who should fill Position Y?"
                
                **How:** Select a billet → See ranked candidates with AI scores.
                """)
            
            with st.expander("🔍 **Semantic AI Search** — *Advanced*"):
                st.markdown("""
                **Purpose:** Natural language search for officers, billets, or similar profiles.
                
                **When to use:**
                - Complex queries: "Masters + HoD experience, NOT staff"
                - Finding officers with similar careers
                - Exploring billet requirements
                
                **How:** Use INCLUDE/EXCLUDE fields or select reference officer.
                """)
        
        with guide_col2:
            with st.expander("📊 **Analytics & Explorer** — *Insights*"):
                st.markdown("""
                **Purpose:** Explore career patterns and dataset insights.
                
                **When to use:**
                - Understanding career transitions
                - Analyzing officer demographics
                - Data exploration
                
                **How:** Browse charts and filters to discover patterns.
                """)
            
            with st.expander("🔄 **Simulation** — *What-If*"):
                st.markdown("""
                **Purpose:** Test hypothetical scenarios.
                
                **When to use:**
                - "What if we promoted Officer X?"
                - Exploring alternative career paths
                - Policy impact analysis
                
                **How:** Modify officer attributes → See predicted outcomes.
                """)
            
            with st.expander("⚙️ **Admin Console** — *Administrators Only*"):
                st.markdown("""
                **Purpose:** Retrain AI model and manage deployments.
                
                **When to use:**
                - Loading new HR data
                - Updating the AI model
                - System maintenance
                
                **How:** Upload CSV → Train → Deploy to production.
                """)
        
        st.divider()
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        qa_col1, qa_col2, qa_col3 = st.columns(3)
        
        with qa_col1:
            st.markdown("**Find Next Role for Officer**")
            st.caption("Go to 👤 Employee Lookup")
        
        with qa_col2:
            st.markdown("**Fill an Open Position**")
            st.caption("Go to 🎯 Billet Lookup")
        
        with qa_col3:
            st.markdown("**Search by Skills**")
            st.caption("Go to 🔍 Semantic AI Search")
        
        st.divider()
        st.caption("💡 **Tip:** Use the Advanced Settings in the sidebar to adjust rank flexibility for predictions.")

    # =========================================================================
    # MODE 6: ADMIN CONSOLE (RETRAINING HUD)
    # =========================================================================
    if mode == "⚙️ Admin Console":
        st.header("⚙️ Model Management Console")
        st.markdown("Upload new data, retrain the AI, and manage deployments.")
        
        from src.training_manager import TrainingManager
        manager = TrainingManager(base_dir=os.getcwd())
        
        # Session State for Training Flow
        if 'training_session' not in st.session_state:
            st.session_state.training_session = None
        
        # 1. Dataset Upload
        st.subheader("1. Update Knowledge Base")
        uploaded_file = st.file_uploader("Upload Updated HR Dataset (CSV)", type="csv")
        
        if uploaded_file:
            # Save temp
            path = manager.save_uploaded_dataset(uploaded_file)
            st.success(f"File uploaded: {os.path.basename(path)}")
            
            # Validate
            valid, msg = manager.validate_dataset(path)
            if valid:
                st.success("[OK] Dataset Structure Validated")
                
                # 2. Training Trigger
                st.subheader("2. Retrain Model")
                st.info("Training will generate a new 'Staging' model. Production is not affected until you deploy.")
                
                if st.button("Start Training Pipeline"):
                    with st.status("Running Training Pipeline...", expanded=True) as status:
                        st.write("Initializing...")
                        
                        # Run Training
                        session_id, result = manager.train_staging_model(path)
                        
                        if session_id:
                            st.write("[OK] Data Processing Complete")
                            st.write("[OK] LTR Model Trained")
                            st.write("[OK] Artifacts Generated")
                            status.update(label="Training Complete!", state="complete", expanded=False)
                            
                            st.session_state.training_session = session_id
                            st.session_state.training_metrics = result # Save metrics
                            st.success(f"Training Success! Session ID: {session_id}")
                        else:
                            status.update(label="Training Failed", state="error")
                            st.error(f"Pipeline Error: {result}")
            else:
                st.error(f"[ERROR] Validation Failed: {msg}")
                
        # 3. Deployment
        if st.session_state.training_session:
            st.divider()
            st.subheader("3. Review & Deploy")
            st.info(f"Staging Model Ready: {st.session_state.training_session}")
            
            # Show Metrics
            if 'training_metrics' in st.session_state and isinstance(st.session_state.training_metrics, dict):
                m = st.session_state.training_metrics
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("New Model Accuracy", f"{m.get('accuracy', 0):.2%}")
                m_col2.metric("New Model AUC", f"{m.get('auc', 0):.4f}")
            else:
                st.warning("Metrics unavailable.")
            
            st.markdown("---")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("Deploy to Production", type="primary"):
                    success, msg = manager.commit_model(st.session_state.training_session)
                    if success:
                        st.success(f"DEPLOYED! {msg}")
                        st.cache_resource.clear()
                        st.balloons()
                    else:
                        st.error(f"Deployment Failed: {msg}")
            
            with col_d2:
                if st.button("Discard Staging Model"):
                    st.session_state.training_session = None
                    st.rerun()

        # 4. Rollback
        st.divider()
        with st.expander("Danger Zone: Rollback"):
            st.warning("Restore the previous model version if the current one is unstable.")
            if st.button("Rollback to Last Backup"):
                success, msg = manager.rollback()
                if success:
                    st.success(msg)
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(msg)
    
    # =========================================================================
    # MODE 1: EMPLOYEE LOOKUP
    # =========================================================================
    if mode == "👤 Employee Lookup":
        st.header("👤 Employee Lookup")
        st.info("🎯 **Find the best next role for an officer** based on their career history, skills, and organizational patterns. The AI analyzes historical career progressions and identifies optimal next steps.")
        
        # Compact 2-Column Layout
        col_left, col_right = st.columns([1, 2])
        
        # === LEFT PANEL: Filters & Officer Selection ===
        with col_left:
            st.markdown("#### 🔎 Find Officer")
            
            # Quick search
            employee_id_search = st.text_input("🆔 Employee ID", placeholder="e.g., 200094", key="emp_id_search")
            name_search = st.text_input("👤 Name", placeholder="e.g., Kirk", key="emp_name_search")
            
            # Filters in expander
            with st.expander("🎚️ Filters", expanded=False):
                all_ranks = sorted(df['Rank'].unique())
                all_branches = sorted(df['Branch'].unique())
                all_entries = sorted(df['Entry_type'].unique())
                
                sel_rank = st.selectbox("Rank", ["All"] + list(all_ranks), key="emp_rank")
                sel_branch = st.selectbox("Branch", ["All"] + list(all_branches), key="emp_branch")
                sel_entry = st.selectbox("Entry", ["All"] + list(all_entries), key="emp_entry")
                pattern_filter = st.selectbox("Pattern", ["All", "Has Pattern", "High Confidence (>80%)", "No Pattern"], key="emp_pattern")
            
            # Apply filters
            filtered_df = df.copy()
            if sel_rank != "All": filtered_df = filtered_df[filtered_df['Rank'] == sel_rank]
            if sel_branch != "All": filtered_df = filtered_df[filtered_df['Branch'] == sel_branch]
            if sel_entry != "All": filtered_df = filtered_df[filtered_df['Entry_type'] == sel_entry]
            if employee_id_search:
                filtered_df = filtered_df[filtered_df['Employee_ID'].astype(str).str.contains(employee_id_search, case=False, na=False)]
            if name_search:
                filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]
            
            count = len(filtered_df)
            
            # Officer list with navigation
            st.markdown("---")
            st.caption(f"📋 {count} officers found")
            
            if count > 0:
                if 'curr_idx' not in st.session_state: st.session_state.curr_idx = 0
                if st.session_state.curr_idx >= count: st.session_state.curr_idx = 0
                
                # Navigation
                nav_cols = st.columns([1, 2, 1])
                with nav_cols[0]:
                    if st.button("◀ Prev", key="prev_emp"):
                        st.session_state.curr_idx = (st.session_state.curr_idx - 1) % count
                with nav_cols[1]:
                    st.markdown(f"<p style='text-align:center; font-weight:bold;'>{st.session_state.curr_idx + 1} / {count}</p>", unsafe_allow_html=True)
                with nav_cols[2]:
                    if st.button("Next ▶", key="next_emp"):
                        st.session_state.curr_idx = (st.session_state.curr_idx + 1) % count
                
                row = filtered_df.iloc[st.session_state.curr_idx]
        
        # === RIGHT PANEL: Officer Info + Predictions ===
        with col_right:
            if count > 0:
                # OFFICER INFO AT TOP
                st.markdown(f"### 👤 {row['Rank']} {row['Name']}")
                
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("🆔 ID", row['Employee_ID'])
                with info_cols[1]:
                    st.metric("🌿 Branch", row['Branch'])
                with info_cols[2]:
                    st.metric("🌊 Pool", row['Pool'])
                with info_cols[3]:
                    st.metric("📅 Entry", row['Entry_type'])
                
                st.caption(f"📍 **Current Role:** {row['current_appointment']}")
                
                with st.expander("📜 Career History", expanded=False):
                    st.text(str(row['Appointment_history']))
                
                st.divider()
                
                # PREDICTIONS
                st.markdown("#### 🎯 Recommended Next Roles")
                
                try:
                    input_df = pd.DataFrame([row])
                    results = predictor.predict(input_df, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                    top_conf = results.iloc[0]['Confidence']
                    
                    # Top recommendation highlight
                    if top_conf > 0.4:
                        st.success(f"✅ **Top Recommendation:** {results.iloc[0]['Prediction']} ({top_conf:.0%})")
                    else:
                        st.warning(f"⚠️ **Top Recommendation:** {results.iloc[0]['Prediction']} ({top_conf:.0%})")
                    
                    from src.xai_helpers import get_markov_pattern_info
                    
                    # Build results with patterns
                    results_with_patterns = []
                    for idx, r_row in results.head(10).iterrows():
                        score = r_row['Confidence']
                        role_name = r_row['role'] if 'role' in r_row else r_row['Prediction']
                        pattern_info = get_markov_pattern_info(predictor, row, role_name)
                        results_with_patterns.append({
                            'idx': idx, 'row': r_row, 'score': score,
                            'role_name': role_name, 'pattern_info': pattern_info
                        })
                    
                    # Apply pattern filter
                    if pattern_filter == "Has Pattern":
                        results_with_patterns = [r for r in results_with_patterns if r['pattern_info'] and r['pattern_info']['context_seen']]
                    elif pattern_filter == "High Confidence (>80%)":
                        results_with_patterns = [r for r in results_with_patterns if r['pattern_info'] and r['pattern_info']['context_seen'] and r['pattern_info']['target_probability'] > 0.8]
                    elif pattern_filter == "No Pattern":
                        results_with_patterns = [r for r in results_with_patterns if not r['pattern_info'] or not r['pattern_info']['context_seen']]
                    
                    if not results_with_patterns:
                        st.info(f"No results match '{pattern_filter}' filter.")
                    
                    # Results list with full details
                    for result_data in results_with_patterns:
                        r_row = result_data['row']
                        score = result_data['score']
                        role_name = result_data['role_name']
                        pattern_info = result_data['pattern_info']
                        
                        # Pattern badge
                        pattern_badge = ""
                        if pattern_info and pattern_info['context_seen']:
                            prob = pattern_info['target_probability']
                            order = pattern_info['order_used']
                            pattern_badge = f" 🔗 {order}-step ({prob*100:.0f}%)"
                        
                        # Score indicator
                        score_icon = "🟢" if score > 0.5 else "🟡" if score > 0.1 else "🔴"
                        
                        with st.expander(f"{score_icon} {score:.0%} | {role_name}{pattern_badge}"):
                            feats = r_row.get('_Feats', {})
                            contribs = r_row.get('_Contribs', None)
                            
                            if feats:
                                metrics = explainer.format_feature_explanation(feats, score=top_conf, constraints=predictor.constraints, contribs=contribs)
                                
                                # Metrics with TOOLTIPS restored
                                st.markdown("**Why this recommendation?**")
                                m_cols = st.columns(5)
                                
                                m_ai = metrics.get('AI Score', {'value': '-', 'desc': 'No Data'})
                                m_cols[0].metric("AI Score", m_ai['value'], help=m_ai.get('desc', ''))
                                
                                m_hist = metrics.get('History Strength', {'value': '-', 'desc': 'No Data'})
                                m_cols[1].metric("History", m_hist['value'], help=m_hist.get('desc', ''))
                                
                                m_train = metrics.get('Training Match', {'value': '-', 'desc': 'No Data'})
                                m_cols[2].metric("Training", m_train['value'], help=m_train.get('desc', ''))
                                
                                m_bp = metrics.get('Branch & Pool Fit', {'value': '-', 'desc': 'No Data'})
                                m_cols[3].metric("Branch & Pool", m_bp['value'], help=m_bp.get('desc', ''))
                                
                                m_re = metrics.get('Rank & Entry Fit', {'value': '-', 'desc': 'No Data'})
                                m_cols[4].metric("Rank & Entry", m_re['value'], help=m_re.get('desc', ''))
                                
                                # XAI Section
                                if contribs:
                                    display_xai_section(
                                        predictor=predictor, explainer=explainer,
                                        officer_data=row, target_role=role_name,
                                        score=score, contribs=contribs,
                                        base_value=r_row.get('_BaseVal', 0.0),
                                        feats=feats, mode="employee"
                                    )
                            
                            # FULL Precedent Table restored
                            st.markdown("#### 📊 Historical Precedents")
                            ctx = feats.get('_Context', {}) if feats else {}
                            curr_title = ctx.get('From_Title') or row.get('current_appointment')
                            if curr_title:
                                precedents = explainer.get_precedents(curr_title, role_name)
                                if precedents:
                                    st.write(f"Officers who moved from **{curr_title}** to **{role_name}**:")
                                    prec_cols = ['Employee_ID', 'Rank', 'Name', 'Branch', 'Pool', 'Entry_type']
                                    st.dataframe(pd.DataFrame(precedents)[prec_cols], hide_index=True, use_container_width=True)
                                else:
                                    st.caption("No exact historical precedents found for this transition.")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.warning("No officers match. Adjust filters.")

    # =========================================================================
    # MODE 4: ANALYTICS & EXPLORER
    # =========================================================================
    elif mode == "📊 Analytics & Explorer":
        st.header("📊 Analytics & Data Explorer")
        st.info("📈 **Explore organizational data, career flows, and appointment patterns**. Analyze career progression trends, organizational structure, and historical appointment timelines.")
        st.caption("✨ Strategic insights, organizational structure, and detailed dataset exploration.")
        
        import plotly.graph_objects as go
        import plotly.express as px
        from src.data_processor import DataProcessor
        from streamlit_echarts import st_echarts
        from src.gantt_viz import create_comprehensive_officer_timeline, create_billet_occupancy_timeline
        import networkx as nx
        
        # Tabs
        t_data, t_stats, t_flow, t_map, t_void, t_org, t_gantt = st.tabs([
            "Data Browser",
            "Statistics",
            "Career Explorer",
            "Appointments Map",
            "Void Analysis",
            "Org Structure",
            "Timeline Explorer"
        ])
        
        # =====================================================================
        # TAB 1: DATA BROWSER
        # =====================================================================
        with t_data:
            st.markdown("###  Dataset Explorer")
            st.caption("Inspect raw records, filter datasets, and download reports.")
            
            # Use standard Streamlit Dataframe for stability
            st.dataframe(
                df,
                use_container_width=True,
                height=500,
                column_config={
                    "Employee_ID": st.column_config.TextColumn("ID"),
                    "Date_of_appointment": st.column_config.DateColumn("Appointed Since"),
                    "Appointment_history": st.column_config.TextColumn("History", help="Full career path"),
                }
            )

        # =====================================================================
        # TAB 2: STATISTICS
        # =====================================================================
        with t_stats:
            st.markdown("###  Workforce Analytics")
            st.caption("High-level demographics, service duration, and rank distribution insights.")
            
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Total Personnel", len(df))
            s_col2.metric("Branches", df['Branch'].nunique())
            s_col3.metric("Avg Career Depth", int(df['Appointment_history'].apply(lambda x: len(str(x).split(','))).mean()))
            
            st.markdown("---")
            
            # --- New Charts ---
            st.subheader(" Deep Dive Analytics")
            
            # 1. Prepare Data
            stats_df = df.copy()
            import re
            
            # Helper for Service Years (First Appt Year)
            def get_start_year(hist):
                # Find all years (YYYY)
                matches = re.findall(r'\((\d{4})', str(hist))
                if matches:
                    # Return min year
                    return min([int(m) for m in matches])
                return 2025 # Fallback
                
            # Helper for Current Role Years
            def get_curr_year(row):
                val = str(row.get('Date_of_appointment', row.get('appointed_since', '')))
                if len(val) >= 4 and val[:4].isdigit():
                    return int(val[:4])
                return 2025
                
            stats_df['Start_Year'] = stats_df['Appointment_history'].apply(get_start_year)
            stats_df['Current_Role_Year'] = stats_df.apply(get_curr_year, axis=1)
            
            stats_df['Years_of_Service'] = 2025 - stats_df['Start_Year']
            stats_df['Years_in_Current_Role'] = 2025 - stats_df['Current_Role_Year']
            
            # remove calculated negatives
            stats_df = stats_df[stats_df['Years_of_Service'] >= 0]
            stats_df = stats_df[stats_df['Years_in_Current_Role'] >= 0]
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**⏳ Years of Service Distribution**")
                # Histogram of Service Years
                fig_service = px.histogram(
                    stats_df, 
                    x='Years_of_Service', 
                    color='Branch',
                    nbins=20,
                    title="Workforce Experience Profile",
                    labels={'Years_of_Service': 'Total Years of Service'},
                    template="plotly_dark",
                    barmode='stack'
                )
                st.plotly_chart(fig_service, use_container_width=True)
                st.caption("Distribution of total career length across the force.")

            with c2:
                st.markdown("** Stagnation Analysis (Time in Rank)**")
                # Box Plot of Years in Current Role by Rank
                fig_stagnation = px.box(
                    stats_df,
                    x='Rank',
                    y='Years_in_Current_Role',
                    color='Rank',
                    title="Time in Current Role by Rank",
                    labels={'Years_in_Current_Role': 'Years in Role'},
                    template="plotly_dark"
                )
                st.plotly_chart(fig_stagnation, use_container_width=True)
                st.caption("How long officers have held their current rank/role.")

            st.markdown("---")
            
            # Existing Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rank Distribution**")
                fig_rank = px.bar(df['Rank'].value_counts(), orientation='h', template="plotly_dark", color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig_rank, use_container_width=True)
                
            with col2:
                st.markdown("**Branch Composition**")
                fig_branch = px.pie(df, names='Branch', hole=0.4, template="plotly_dark")
                st.plotly_chart(fig_branch, use_container_width=True)

        # =====================================================================
        # TAB 3: CAREER EXPLORER
        # =====================================================================
        with t_flow:
            st.markdown("###  Detailed Career Explorer")
            st.caption("Explore exact career paths. Use filters to drill down.")
            
            # --- 1. Advanced Filters ---
            with st.expander(" Filter Population", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                all_branches = sorted(df['Branch'].unique())
                sel_branches = c1.multiselect("Branch", all_branches, default=all_branches)  # Auto-select all
                all_pools = sorted(df['Pool'].unique())
                sel_pools = c2.multiselect("Pool", all_pools, default=all_pools)  # Auto-select all
                all_ranks = sorted(df['Rank'].unique())
                sel_ranks = c3.multiselect("Current Rank", all_ranks, default=all_ranks)  # Auto-select all
                all_entries = sorted(df['Entry_type'].unique())
                sel_entries = c4.multiselect("Entry Type", all_entries, default=all_entries)  # Auto-select all
            
            # Filter Data
            filtered_df = df.copy()
            if sel_branches: filtered_df = filtered_df[filtered_df['Branch'].isin(sel_branches)]
            if sel_pools: filtered_df = filtered_df[filtered_df['Pool'].isin(sel_pools)]
            if sel_ranks: filtered_df = filtered_df[filtered_df['Rank'].isin(sel_ranks)]
            if sel_entries: filtered_df = filtered_df[filtered_df['Entry_type'].isin(sel_entries)]
            
            if filtered_df.empty:
                st.warning("No records match these filters.")
            else:
                st.info(f"Analyzing {len(filtered_df)} officers...")
                
                with st.spinner("Tracing career histories..."):
                    dp = DataProcessor()
                    paths_list = []
                    max_depth = 20
                    
                    # Store officer names associated with each path for tooltips
                    path_objects = []
                    
                    for _, row in filtered_df.iterrows():
                        hist = row.get('Appointment_history', '')
                        parsed = dp.parse_history_column(hist)
                        emp_name = row.get('Name', row['Employee_ID']) # Fallback if Name missing
                        
                        chain = []
                        if parsed:
                            parsed.sort(key=lambda x: x['start_date'] if pd.notna(x['start_date']) else pd.Timestamp.min)
                            for p in parsed:
                                t = str(p['title']).strip()
                                chain.append(t)
                        
                        chain.append(str(row['current_appointment']).strip())
                        
                        clean = []
                        for x in chain:
                            if not clean or clean[-1] != x:
                                clean.append(x)
                        
                        if len(clean) > max_depth:
                            clean = clean[-max_depth:]
                            
                        path_objects.append({
                            "path": clean,
                            "name": emp_name
                        })
                    
                    # --- Build ECharts Tree ---
                    # Root
                    tree = {"name": "Career Start", "children": []}
                    
                    # Helper to find child
                    def find_child(node, name):
                        for c in node['children']:
                            if c['name'] == name: return c
                        return None
                    
                    for obj in path_objects:
                        curr = tree
                        for step in obj['path']:
                            child = find_child(curr, step)
                            if not child:
                                child = {"name": step, "children": [], "value": 0, "officers": []}
                                curr['children'].append(child)
                            child['value'] += 1
                            child['officers'].append(obj['name'])
                            curr = child
                            
                    # Prune or Format
                    # Limit officers list in tooltip
                    def format_tree(node):
                        # Tooltip formatting
                        off_list = node.get('officers', [])
                        count = len(off_list)
                        if count > 10:
                            shown = off_list[:10]
                            node['tooltip_html'] = f"{count} Officers:<br/>" + "<br/>".join(shown) + f"<br/>...and {count-10} more"
                        else:
                            node['tooltip_html'] = f"{count} Officers:<br/>" + "<br/>".join(off_list)
                            
                        for c in node['children']:
                            format_tree(c)
                            
                    format_tree(tree)
                    
                    option = {
                        "tooltip": {
                            "trigger": "item",
                            "triggerOn": "mousemove",
                            "formatter": "{b}<br/>{c} records" 
                        },
                        "series": [
                            {
                                "type": "tree",
                                "data": [tree],
                                "layout": "orthogonal",
                                "orient": "LR",
                                "symbolSize": 7,
                                "initialTreeDepth": 3,
                                "label": {
                                    "position": "left",
                                    "verticalAlign": "middle",
                                    "align": "right",
                                    "fontSize": 10
                                },
                                "leaves": {
                                    "label": {
                                        "position": "right",
                                        "verticalAlign": "middle",
                                        "align": "left"
                                    }
                                },
                                "expandAndCollapse": True,
                                "animationDuration": 550,
                                "animationDurationUpdate": 750
                            }
                        ]
                    }
                    st_echarts(option, height="700px")

        # =====================================================================
        # TAB 4: APPOINTMENTS MAP
        # =====================================================================
        with t_map:
            st.markdown("###  Live Career Network")
            st.caption("Direct visualization of career paths. Filters update the graph in real-time.")
            
            # 1. Direct Filtering
            f_col1, f_col2, f_col3 = st.columns(3)
            
            all_b_vals = sorted(df['Branch'].unique())
            all_r_vals = sorted(df['Rank'].unique())
            
            sel_b = f_col1.multiselect("Branches", all_b_vals, default=all_b_vals, key="map_new_b")
            sel_r = f_col2.multiselect("Ranks", all_r_vals, default=all_r_vals, key="map_new_r")
            
            # Tuning
            min_w = f_col3.slider("Min Traffic", 1, 50, 1, help="Hide paths with fewer than N officers")

            # 2. Filter Data Frame
            m_df = df.copy()
            
            if sel_b:
                m_df = m_df[m_df['Branch'].isin(sel_b)]
            if sel_r:
                m_df = m_df[m_df['Rank'].isin(sel_r)]
                
            st.metric("Records Analysis", len(m_df))
            
            # 3. Graph Logic (Corner Layout + Running Text Tooltips)
            if m_df.empty:
                st.warning("No data found for these filters.")
            else:
                import networkx as nx
                from streamlit_echarts import st_echarts
                from collections import Counter, defaultdict
                import re
                import math

                # Helper to normalize & extract date
                def parse_role_date(t_str):
                    t_str = str(t_str).strip()
                    # Extract date if present
                    date_match = re.search(r'\((\d{4})', t_str)
                    year = int(date_match.group(1)) if date_match else 9999
                    
                    # Clean title
                    clean = re.sub(r'\s*\(.*?\)', '', t_str)
                    clean = re.sub(r'\s*\/.*', '', clean) 
                    clean = re.sub(r'\s*USS .*', '', clean)
                    clean = re.sub(r'\s*Starbase .*', '', clean) 
                    return clean.strip(), year

                transitions = []
                node_meta = {} # store branch votes
                node_officers = defaultdict(list) # list of details per node
                
                for _, row in m_df.iterrows():
                    hist = row.get("Appointment_history", "")
                    curr_b = row['Branch']
                    off_name = row['Name']
                    off_id = row['Employee_ID']
                    off_rank = row['Rank']
                    
                    # Current Appt Year
                    # Fix: Handle missing column name safely (dataset uses 'appointed_since')
                    curr_date = str(row.get('Date_of_appointment', row.get('appointed_since', '')))
                    curr_year = int(curr_date[:4]) if len(curr_date) >= 4 and curr_date[:4].isdigit() else 9999
                    
                    if not isinstance(hist, str): continue
                    
                    # Parse History
                    raw_parts = [x.strip() for x in hist.split(',')]
                    path_nodes = []
                    
                    for raw in raw_parts:
                        role, year = parse_role_date(raw)
                        final_yr = year if year != 9999 else "N/A"
                        if len(role) > 2:
                            path_nodes.append(role)
                            # Add to officer record
                            # Format: Rank Name (ID) (Branch) (Year)
                            detail = f"{off_rank} {off_name} ({off_id}) ({curr_b}) ({final_yr})"
                            node_officers[role].append((year, detail))
                            
                    # Add current
                    curr_role, _ = parse_role_date(row['current_appointment'])
                    final_curr_yr = curr_year if curr_year != 9999 else "N/A"
                    if not path_nodes or path_nodes[-1] != curr_role:
                        if len(curr_role) > 2:
                            path_nodes.append(curr_role)
                            detail = f"{off_rank} {off_name} ({off_id}) ({curr_b}) ({final_curr_yr})"
                            node_officers[curr_role].append((curr_year, detail))

                    # Metadata & Edges
                    if len(path_nodes) >= 1:
                        for p in path_nodes:
                            if p not in node_meta: node_meta[p] = []
                            node_meta[p].append(curr_b)
                            
                    if len(path_nodes) >= 2:
                        for i in range(len(path_nodes)-1):
                            transitions.append((path_nodes[i], path_nodes[i+1]))
                
                if not transitions:
                    st.info("No transitions found in selected data.")
                else:
                    # Graph Construction
                    G = nx.DiGraph()
                    counts = Counter(transitions)
                    
                    for (u,v), w in counts.items():
                        if w >= min_w:
                            G.add_edge(u, v, weight=w)

                    # Branch Stats
                    unique_branches = sorted(list(set([b for votes in node_meta.values() for b in votes])))
                    categories = [{"name": b} for b in unique_branches]
                    
                    # Identify Top 4 Branches
                    branch_counts = Counter([b for votes in node_meta.values() for b in votes])
                    top_branches = [b for b, c in branch_counts.most_common(4)]
                    
                    # --- CORNER LAYOUT STRATEGY ---
                    # 4 Corners in normalized -1..1 space
                    corners = [(-0.8, 0.8), (0.8, 0.8), (0.8, -0.8), (-0.8, -0.8)]
                    centers = {}
                    
                    # Assign top branches to corners
                    for i, b in enumerate(top_branches):
                        centers[b] = corners[i]
                        
                    # Remaining branches in center or scattered?
                    # Center (0,0) is fine, they will float in middle
                    
                    # 2. Assign initial positions based on dominant branch
                    initial_pos = {}
                    for n in G.nodes():
                        votes = node_meta.get(n, [])
                        if votes:
                            b_mode = Counter(votes).most_common(1)[0][0]
                            # Use corner if top branch, else 0,0
                            cx, cy = centers.get(b_mode, (0,0))
                            import random
                            # Small jitter to prevent stacking
                            initial_pos[n] = (cx + random.uniform(-0.1, 0.1), cy + random.uniform(-0.1, 0.1))
                        else:
                            initial_pos[n] = (0,0)

                    # 3. Run Spring Layout
                    # k=0.15 is standard. We use default scale=1.
                    with st.spinner("Optimizing map (Corner Layout)..."):
                        pos = nx.spring_layout(G, pos=initial_pos, k=0.2, iterations=100, seed=42)
                    
                    # Nodes
                    echarts_nodes = []
                    SCALE = 4000 # Expand to large coordinate space
                    
                    for n in G.nodes():
                        x, y = pos[n]
                        
                        # Category
                        votes = node_meta.get(n, [])
                        if votes:
                            b_mode = Counter(votes).most_common(1)[0][0]
                            cat_idx = unique_branches.index(b_mode) if b_mode in unique_branches else 0
                        else:
                            cat_idx = 0
                        
                        # Tooltip Content (Running Text)
                        raw_officers = node_officers.get(n, [])
                        raw_officers.sort(key=lambda x: x[0]) # Sort by Year
                        
                        # Create comma separated list
                        entries = [x[1] for x in raw_officers]
                        
                        # Limit length for performance/display
                        if len(entries) > 50:
                            entries = entries[:50] + [f"...and {len(entries)-50} more"]
                            
                        running_text = ", ".join(entries)
                        
                        header = f"<b>{n}</b> <span style='color:#aaa'>({unique_branches[cat_idx]})</span>"
                        # Wrapper div
                        tooltip_html = f"""
                        <div style="width:350px; white-space:normal; font-family:sans-serif; font-size:11px; line-height:1.4;">
                            {header}<br/>
                            <hr style="margin:4px 0; border:0; border-top:1px solid #555;">
                            {running_text}
                        </div>
                        """
                            
                        deg = G.degree(n)
                        # Size scaling
                        sz = min(10 + deg*1.5, 60)
                        
                        echarts_nodes.append({
                            "name": n,
                            "x": x * SCALE,
                            "y": y * SCALE,
                            "value": tooltip_html, # Store HTML in value
                            "category": cat_idx,
                            "symbolSize": sz,
                            "draggable": True,
                            "label": {"show": deg > 5},
                            "tooltip": {"formatter": "{c}"} # Display HTML
                        })
                        
                    # Links
                    echarts_links = []
                    for u, v, d in G.edges(data=True):
                        w = d['weight']
                        echarts_links.append({
                            "source": u,
                            "target": v,
                            "value": w,
                            "lineStyle": {"width": min(1 + w/2, 5)},
                            "tooltip": {"formatter": "{b} (Traffic: {c})"}
                        })
                        
                    # ECharts Option
                    option = {
                        "title": {"text": "Career Relationship Network", "subtext": "Clustered by Branch"},
                        "tooltip": {
                            "trigger": "item",
                            "encode": {"tooltip": "value"}, # Ensure value is used
                            "confine": True, # Keep tooltip inside canvas
                            "textStyle": {"fontSize": 10}
                        }, 
                        "legend": [{"data": unique_branches, "orient": "horizontal", "bottom": 0}],
                        "animationDuration": 1500,
                        "animationEasingUpdate": "quinticInOut",
                        "series": [{
                            "name": "Roles",
                            "type": "graph",
                            "layout": "none",
                            "data": echarts_nodes,
                            "links": echarts_links,
                            "categories": categories,
                            "roam": True,
                            "label": {
                                "position": "right",
                                "formatter": "{b}"
                            },
                            "labelLayout": {
                                "hideOverlap": True
                            },
                            "scaleLimit": {
                                "min": 0.1,
                                "max": 5
                            },
                            "lineStyle": {
                                "color": "source",
                                "curveness": 0.3
                            },
                            "emphasis": {
                                "focus": "adjacency",
                                "lineStyle": {
                                    "width": 10
                                }
                            }
                        }]
                    }
                    
                    st_echarts(option, height="800px")
                    st.caption("**Tip**: Hover to see distinct officer history. Nodes are clustered by Branch.")
                    
        # =====================================================================
        # TAB 5: VOID ANALYSIS
        # =====================================================================
        with t_void:
            st.markdown("###  Critical Gap & Void Detection")
            st.info("Identify branches or ranks with dangerous personnel shortages.")
            
            if 'Rank' in df.columns and 'Branch' in df.columns:
                heatmap_data = pd.crosstab(df['Branch'], df['Rank'])
                
                sorter = ["Ensign", "Lieutenant (jg)", "Lieutenant", "Lieutenant Commander", "Commander", "Captain", "Commodore", "Rear Admiral", "Vice Admiral", "Admiral"]
                existing_ranks = [r for r in sorter if r in heatmap_data.columns]
                heatmap_data = heatmap_data[existing_ranks]
                
                fig_void = px.imshow(
                    heatmap_data,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdYlGn", 
                    title="Officer Density Heatmap (Red = Potential Void)"
                )
                fig_void.update_layout(height=600)
                st.plotly_chart(fig_void, use_container_width=True)
                
                voids = []
                for b in heatmap_data.index:
                    for r in heatmap_data.columns:
                        val = heatmap_data.loc[b, r]
                        if val < 3:
                            voids.append(f"**{b}** - **{r}** ({val} officers)")
                 
                if voids:
                    with st.expander(f"[WARNING] Critical Voids Detected ({len(voids)})", expanded=False):
                        st.write(", ".join(voids))
            else:
                st.error("Missing Rank/Branch data.")

        # --- TAB 3: ORG STRUCTURE ---
        with t_org:
            st.markdown("###  Organizational Unit Browser")
            st.info("Drill down from high-level Units to individual Posts.")
            
            def extract_unit(appt):
                if pd.isna(appt): return "Unknown", "Unknown"
                s = str(appt)
                if '/' in s:
                    parts = s.split('/')
                    left = parts[0].strip()
                    post = parts[1].strip()
                    return left, post
                else:
                    return s, "General Staff"

            if 'Unit' not in df.columns:
                # df[['Org_Unit', 'Org_Post']] = df['current_appointment'].apply(lambda x: pd.Series(extract_unit(x)))
                # Using standard loop to avoid 'Unit' key error if apply fails or setting issues
                res = df['current_appointment'].apply(extract_unit)
                df['Org_Unit'] = [x[0] for x in res]
                df['Org_Post'] = [x[1] for x in res]
            
            org_view = df.groupby(['Org_Unit', 'Org_Post']).size().reset_index(name='count')
            
            search_unit = st.text_input("Search Unit", "")
            if search_unit:
                org_view = org_view[org_view['Org_Unit'].str.contains(search_unit, case=False)]
            
            if len(org_view) > 0:
                fig_tree = px.treemap(org_view, path=['Org_Unit', 'Org_Post'], values='count',
                                      color='count', color_continuous_scale='Blues',
                                      title="Organizational Command Structure")
                fig_tree.update_layout(height=800)
                fig_tree.update_traces(textinfo="label+value", root_color="lightgrey")
                st.plotly_chart(fig_tree, use_container_width=True)
                
                st.divider()
                st.markdown("#### Officer: Unit Roster")
                sel_unit = st.selectbox("Select Unit to Inspect", sorted(org_view['Org_Unit'].unique()))
                
                if sel_unit:
                    roster = df[df['Org_Unit'] == sel_unit]
                    st.dataframe(roster[['Employee_ID', 'Rank', 'Name', 'current_appointment', 'Branch']], hide_index=True)
            else:
                st.warning("No units match search.")
        
        # =====================================================================
        # TAB 7: APPOINTMENT TIMELINE (GANTT CHART)
        # =====================================================================
        with t_gantt:
            st.markdown("### Timeline Explorer")
            st.caption("Complete HR temporal view: appointments, training, and promotions")
            
            # View selector
            view_mode = st.radio(
                "View Mode",
                ["By Officer", "By Billet/Position"],
                horizontal=True,
                help="Officer view shows complete career timeline per person. Billet view shows who held each position over time."
            )
            
            # Filters
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                all_branches_g = ["All"] + sorted(df['Branch'].unique().tolist())
                filter_branch_g = st.selectbox("Filter by Branch", all_branches_g, key="gantt_branch")
            with col_f2:
                start_date_g = st.date_input("Start Date (Optional)", value=None, key="gantt_start")
            with col_f3:
                end_date_g = st.date_input("End Date (Optional)", value=None, key="gantt_end")
            
            # Convert dates
            start_dt = datetime.combine(start_date_g, datetime.min.time()) if start_date_g else None
            end_dt = datetime.combine(end_date_g, datetime.max.time()) if end_date_g else None
            
            # Legend
            if view_mode == "By Officer":
                st.info("**Legend:** Blue bars = Appointments | Green bars = Training courses | Gold stars (⭐) = Promotions")
            else:
                st.info("**Legend:** Each row shows a position/billet. Bars show when different officers held that position.")
            
            try:
                if view_mode == "By Officer":
                    fig_gantt = create_comprehensive_officer_timeline(
                        df,
                        filter_branch=filter_branch_g if filter_branch_g != "All" else None,
                        start_date=start_dt,
                        end_date=end_dt
                    )
                else:
                    fig_gantt = create_billet_occupancy_timeline(
                        df,
                        filter_branch=filter_branch_g if filter_branch_g != "All" else None,
                        start_date=start_dt,
                        end_date=end_dt
                    )
                
                # Display chart based on type
                if view_mode == "By Officer":
                    chart, height = fig_gantt
                    # Altair handles height via properties, but we can set container height if needed?
                    # Streamlit handles altair chart sizing well.
                    # We can use the calculated height to set chart size if not set in properties.
                    # In gantt_viz, we set width='container', but not height explicitly in properties.
                    # Let's trust Altair interactive zooming.
                    st.altair_chart(chart, use_container_width=True, theme="streamlit")
                else:
                    st.plotly_chart(fig_gantt, use_container_width=True)
                
                # Usage instructions
                with st.expander("How to use this chart"):
                    st.markdown("""
                    **Interactions:**
                    - **Scroll** (Mouse Wheel) to Zoom in/out
                    - **Drag** to Pan the timeline
                    - **Hover** for details
                    """)
                    
                    st.markdown("""
                    **By Officer View:**
                    - Each row = one officer's complete career
                    - Blue bars = appointment periods
                    - Green bars = training courses (with duration)
                    - Gold stars (⭐) = promotions
                    
                    **By Billet View:**
                    - Each row = one position/billet
                    - Shows succession of officers over time
                    - Identifies gaps in coverage
                    - Color-coded by branch
                    """)
            except Exception as e:
                st.error(f"Error creating chart: {e}")
                st.caption("Ensure appointment history includes dates in format: Role (YYYY-MM-DD)")

        # --- TAB 4: STATISTICS ---
        with t_stats:
            st.markdown("###  Dataset Demographics")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Officers", len(df))
            k2.metric("Unique Roles", df['current_appointment'].nunique())
            k3.metric("Branches", df['Branch'].nunique())
            k4.metric("Avg Service", "N/A") 
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rank Distribution**")
                st.bar_chart(df['Rank'].value_counts())
            with c2:
                st.markdown("**Branch Distribution**")
                st.bar_chart(df['Branch'].value_counts())

        # --- TAB 5: DATA BROWSER ---
        with t_data:
            st.markdown("###  Detailed Dataset Browser")
            st.caption("Filter, sort, and analyze the raw officer data.")
            
            with st.expander(" Filters & Controls", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    all_ranks = sorted(df['Rank'].unique())
                    sel_ranks_tab = st.multiselect("Filter Ranks", all_ranks, key="tab_rank")
                with c2:
                    all_branches = sorted(df['Branch'].unique())
                    sel_branches_tab = st.multiselect("Filter Branches", all_branches, key="tab_branch")
                with c3:
                    search_term = st.text_input("Search (Name/Role)", key="tab_search")
                with c4:
                    all_cols = list(df.columns)
                    vis_cols = st.multiselect("Visible Columns", all_cols, default=['Employee_ID', 'Rank', 'Name', 'current_appointment', 'Branch'])

            filtered_tab_df = df.copy()
            if sel_ranks_tab:
                filtered_tab_df = filtered_tab_df[filtered_tab_df['Rank'].isin(sel_ranks_tab)]
            if sel_branches_tab:
                filtered_tab_df = filtered_tab_df[filtered_tab_df['Branch'].isin(sel_branches_tab)]
            if search_term:
                mask = filtered_tab_df['Name'].str.contains(search_term, case=False, na=False) | \
                       filtered_tab_df['current_appointment'].str.contains(search_term, case=False, na=False)
                filtered_tab_df = filtered_tab_df[mask]

            m1, m2 = st.columns([1, 1])
            m1.metric("Visible Rows", len(filtered_tab_df))
            
            st.data_editor(
                filtered_tab_df[vis_cols], 
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
    elif mode == "🔄 Simulation":
        st.header("🔄 Experiments Lab")
        st.markdown("🧪 Design hypothetical officers, tweak parameters, and analyze how the AI reacts to specific career profiles.")
        
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
            
            # Background Defaults (User defined)
            years_rank = 3.5
            total_service = 8
            
            st.markdown("### 2. History & Qualifications")
            
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
            
            # Dynamic Training List from Dataset
            @st.cache_data
            def extract_training_set(d_frame):
                t_set = set()
                # Optimised loop
                # Use unique to reduce iteration if many duplicates
                for item in d_frame['Training_history'].astype(str).unique():
                    if item == 'nan': continue
                    for part in item.split(','):
                        # Clean: Remove parens with date e.g. "Course (Date)" -> "Course"
                        clean = part.split('(')[0].strip()
                        if len(clean) > 2:
                            t_set.add(clean)
                return sorted(list(t_set))

            common_training = extract_training_set(df)
            if not common_training: 
                common_training = ["Bridge Officer Test", "Advanced Tactical"] # Fallback
            
            training = st.multiselect("Completed Training", common_training, default=common_training[:2] if len(common_training) >= 2 else common_training)
            
            
            run_btn = st.button(" Run Prediction Analysis", type="primary", use_container_width=True)
            
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
            st.markdown("###  Analysis Console")
            
            if sel_idx is not None and 'lab_results' in st.session_state:
                results = st.session_state['lab_results']
                top_res = results.loc[sel_idx]
                
                score = top_res['Confidence']
                pred_role = top_res.get('role', top_res.get('Prediction', 'Unknown'))
                
                # FEATURE MAP for Insights
                FEATURE_MAP = {
                    'years_in_current_rank': 'Time in Current Rank',
                    'years_service': 'Total Years of Service',
                    'title_sim': 'Job Title Matching',
                    'branch_match': 'Branch Alignment',
                    'pool_match': 'Officer Pool Fit',
                    'prior_title_prob': 'Historical Career Precedent',
                    'rank_match': 'Rank Compatibility',
                    'entry_match': 'Commission Entry Type',
                    'training_match': 'Training Qualifications'
                }
                
                # 1. Score Card
                st.success(f"**Recommendation**: {pred_role}")
                sc_col1, sc_col2 = st.columns(2)
                sc_col1.metric("Confidence", f"{score:.1%}")
                sc_col2.metric("Raw AI Score", f"{top_res.get('_RawScore', 0):.2f}")
                
                # 2. Explainability
                st.markdown("####  Decision Factors")
                
                feats = top_res.get('_Feats', {})
                contribs = top_res.get('_Contribs', {})
                
                if contribs:
                    base_v = top_res.get('_BaseVal', 0.0)
                    
                    # TABS: Individual vs Global
                    t_ind, t_glob = st.tabs([" Individual Analysis", " Global Model Insights"])
                    
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
                    st.divider()
                    st.markdown("####  Analysis Notes")
                    notes = []
                    
                    # 1. Strength Assessment
                    if score > 0.6:
                        st.success(f"**Strong Candidate**: {score:.1%} confidence indicates a high match with historical patterns.")
                    elif score > 0.3:
                        st.warning(f"**Moderate Fit**: {score:.1%} confidence. Viable, but may have stronger competitors.")
                    else:
                        st.error(f"**Low Probability**: {score:.1%} confidence. Atypical profile for this role.")
                    
                    # 2. Factor Analysis (Top Drivers)
                    if contribs:
                        # Sort by absolute impact
                        sorted_c = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                        pros = [k for k,v in sorted_c if v > 0.05][:3]
                        cons = [k for k,v in sorted_c if v < -0.05][-3:]
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            if pros:
                                st.markdown("**[OK] Primary Strengths:**")
                                for p in pros: 
                                    label = FEATURE_MAP.get(p, p)
                                    st.markdown(f"- **{label}**: Positive driver.")
                            else:
                                st.markdown("No distinct primary strengths.")
                                
                        with c2:
                            if cons:
                                st.markdown("**[WARNING] Key Risks:**")
                                for c in cons: 
                                    label = FEATURE_MAP.get(c, c)
                                    st.markdown(f"- **{label}**: Negative impact.")
                            else:
                                st.markdown("No significant negative factors.")
                            
                    # 3. Context
                    if feats.get('prior_title_prob', 0) > 0:
                        st.info(f" **Precedent**: {feats.get('prior_title_prob'):.1%} of officers moved from *{last_role}* to this role.")
                    elif feats.get('prior_title_prob', 0) == 0:
                         st.info(f" **No Precedent**: Evaluating this move based on skills/branch similarity rather than direct history.")
                        
                else:
                    st.warning("No XAI data available.")
            else:
                st.info(" Configure your officer profile on the left and click 'Run Prediction Analysis' to see results.")
            
    elif mode == "🎯 Billet Lookup":
        st.header("🎯 Billet Lookup")
        st.info("👥 **Find the best candidates for a specific role**. The AI ranks officers by fit, considering career patterns, skills, branch/rank requirements, and historical precedents. Perfect for filling critical positions.")
        st.markdown("🔀 Reverse search: Select a target role to find the best fit officers.")
        
        # 1. Filters for Target Role List
        with st.expander(" Filter Target Role List", expanded=True):
            bf_col1, bf_col2, bf_col3, bf_col4 = st.columns(4)
            all_ranks_f = sorted(df['Rank'].unique())
            all_branches_f = sorted(df['Branch'].unique())
            
            f_rank = bf_col1.selectbox("Filter by Role Rank", ["All"] + all_ranks_f)
            f_branch = bf_col2.selectbox("Filter by Role Branch", ["All"] + all_branches_f)
            f_min_inc = bf_col3.number_input("Min. Incumbents (History)", min_value=0, value=0, step=1, help="Show only roles with at least N historical/current incumbents.")
            pattern_filter_billet = bf_col4.selectbox("Career Pattern", ["All", "Has Pattern", "High Confidence (>80%)", "No Pattern"])
        
        # 2. Prepare Role Options with Counts (Historical & Current)
        @st.cache_data
        def get_historical_counts(d_frame):
            from collections import Counter
            all_r = []
            # Current
            all_r.extend(d_frame['current_appointment'].dropna().astype(str).tolist())
            # History
            for h in d_frame['Appointment_history'].dropna().astype(str):
                # Heuristic split by comma, clean dates e.g. "Role (Date)"
                parts = h.split(',')
                for p in parts:
                    clean = p.split('(')[0].strip()
                    if len(clean) > 2:
                        all_r.append(clean)
            return Counter(all_r)

        all_roles = sorted(predictor.target_encoder.classes_)
        counts = get_historical_counts(df)
        
        valid_role_options = []
        for r in all_roles:
            # Check constraints match filter
            const = predictor.constraints.get(r, {})
            c_ranks = const.get('ranks', [])
            c_branches = const.get('branches', [])
            
            # Apply Filters
            if f_rank != "All" and c_ranks and f_rank not in c_ranks: continue
            if f_branch != "All" and c_branches and f_branch not in c_branches: continue
            
            # Count incumbents
            n = counts.get(r, 0)
            if n < f_min_inc: continue # Filter by Min Incumbents
            
            valid_role_options.append((r, f"{r} ({n} incumbents)"))
            
        if not valid_role_options:
            st.warning("No roles match the selected filters.")
            target_role = None
        else:
            # Selectbox
            sel_val = st.selectbox(
                "Target Appointment", 
                options=[x[0] for x in valid_role_options],
                format_func=lambda x: next((v[1] for v in valid_role_options if v[0] == x), x)
            )
            target_role = sel_val
        
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
                            score_color = "🟢 [HIGH]" if score > 0.5 else "🟡 [MED]" if score > 0.1 else "🔴 [LOW]"
                            
                            with st.expander(f"{score_color} {score:.1%} | {row['Employee_ID']} - {row['Rank']} {row['Name']} ({row['Branch']}) - Currently {row.get('CurrentRole', 'Unknown')}"):
                                # 1. Breakdown Chart
                                st.markdown("#### Why this recommendation?")
                                feats = row.get('_Feats', {})
                                if feats:
                                    
                                    # Extract XAI Contribs
                                    contribs = row.get('_Contribs', None)
                                    
                                    metrics = explainer.format_feature_explanation(feats, score=score, constraints=predictor.constraints, contribs=contribs, mode='billet_lookup')
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
                                    
                                    # Deep Dive XAI - NEW IMPLEMENTATION
                                    if contribs:
                                        display_xai_section(
                                            predictor=predictor,
                                            explainer=explainer,
                                            officer_data=row,
                                            target_role=target_role,
                                            score=score,
                                            contribs=contribs,
                                            base_value=row.get('_BaseVal', 0.0),
                                            feats=feats,
                                            mode="billet"
                                        )
                                
                                # 2. Relevant Prior Experience (Incumbent Overlap)
                                st.markdown("####  Relevant Prior Experience")
                                st.caption("Based on the careers of previous incumbents, these roles in the candidate's history are strong qualifiers:")
                                
                                rel_exp = explainer.get_relevant_experience(target_role, row.get('Appointment_history', ''))
                                
                                if rel_exp:
                                    for r in rel_exp:
                                        st.info(f"**{r['role']}**: {r['desc']}")
                    st.warning("No candidates met the confidence threshold for this specific role. Try adjusting filters or the 'Min Incumbents' setting.")

    # =========================================================================
    # MODE: SEMANTIC AI SEARCH
    # =========================================================================
    elif mode == "🔍 Semantic AI Search":
        st.header("🧠 Semantic AI Search")
        
        st.info("""
        **True AI-Powered Search**: Ask questions in natural language and the AI understands your intent.
        
        Supports: Complex queries, negation (NOT, except), temporal (recently, longest), comparisons (more than), and more!
        """)
        
        # Initialize Semantic AI Engine with progress
        @st.cache_resource
        def get_semantic_engine(_df, _predictor):
            try:
                from semantic_engine import SemanticAIEngine
                return SemanticAIEngine(_df, _predictor)
            except Exception as e:
                return None
        
        # Show loading progress for first-time initialization
        if 'semantic_engine_loaded' not in st.session_state:
            with st.status("🔄 Initializing Semantic AI Engine...", expanded=True) as status:
                st.write("📚 Loading query parser...")
                semantic_engine = get_semantic_engine(df, predictor)
                st.write("🔍 Building search indices...")
                st.write("✅ Ready!")
                status.update(label="✅ Semantic AI Engine Ready!", state="complete", expanded=False)
            st.session_state['semantic_engine_loaded'] = True
            st.session_state['semantic_engine'] = semantic_engine
        else:
            semantic_engine = st.session_state.get('semantic_engine') or get_semantic_engine(df, predictor)
        
        if not semantic_engine:
            st.error("Semantic engine not available. Please install sentence-transformers.")
            st.stop()
        
        # Three tabs for different use cases
        tab1, tab2, tab3 = st.tabs(["🎯 Career Match", "🔍 Billet Search", "👥 Similar Officer"])
        
        # =====================================================
        # TAB 1: SEMANTIC CAREER MATCH
        # =====================================================
        with tab1:
            st.markdown("### Find Officers by Experience & Skills")
            st.caption("Use the INCLUDE field for required skills/experience and EXCLUDE for things to avoid")
            
            # Separate Include/Exclude fields
            col_inc, col_exc = st.columns(2)
            
            with col_inc:
                include_terms = st.text_area(
                    "✅ INCLUDE (required)",
                    placeholder="e.g., masters, head of department, instructor, phd, tactical...",
                    height=80,
                    key="career_include",
                    help="Comma-separated terms. Officers MUST have these in their history."
                )
            
            with col_exc:
                exclude_terms = st.text_area(
                    "❌ EXCLUDE (avoid)",
                    placeholder="e.g., staff, engineering, cadet...",
                    height=80,
                    key="career_exclude",
                    help="Comma-separated terms. Officers with these will be filtered OUT."
                )
            
            # Additional filters
            st.markdown("#### 🎚️ Optional Filters")
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                all_ranks = ["Any"] + sorted(df['Rank'].unique())
                filter_rank_career = st.selectbox("Rank", all_ranks, key="career_rank")
            
            with col_f2:
                all_branches = ["Any"] + sorted(df['Branch'].unique())
                filter_branch_career = st.selectbox("Branch", all_branches, key="career_branch")
            
            with col_f3:
                all_pools = ["Any"] + sorted(df['Pool'].unique())
                filter_pool_career = st.selectbox("Pool", all_pools, key="career_pool")
            
            if st.button("🔍 Search Officers", type="primary", key="search_officers"):
                if include_terms.strip() or filter_rank_career != "Any" or filter_branch_career != "Any":
                    progress_bar = st.progress(0, text="Starting search...")
                    
                    # Parse include/exclude terms
                    include_list = [t.strip().lower() for t in include_terms.split(',') if t.strip()]
                    exclude_list = [t.strip().lower() for t in exclude_terms.split(',') if t.strip()]
                    
                    progress_bar.progress(0.2, text="🔍 Filtering officers...")
                    
                    matched_officers = []
                    
                    for idx, row in df.iterrows():
                        score = 0
                        reasons = []
                        
                        # Build searchable text
                        appt_hist = str(row.get('Appointment_history', '')).lower()
                        train_hist = str(row.get('Training_history', '')).lower()
                        curr_role = str(row.get('current_appointment', '')).lower()
                        combined_text = f"{appt_hist} {train_hist} {curr_role}"
                        
                        # Apply rank filter
                        if filter_rank_career != "Any":
                            if row['Rank'] != filter_rank_career:
                                continue
                            reasons.append(f"✓ Rank: {filter_rank_career}")
                            score += 2
                        
                        # Apply branch filter
                        if filter_branch_career != "Any":
                            if row['Branch'] != filter_branch_career:
                                continue
                            reasons.append(f"✓ Branch: {filter_branch_career}")
                            score += 2
                        
                        # Apply pool filter
                        if filter_pool_career != "Any":
                            if row['Pool'] != filter_pool_career:
                                continue
                            reasons.append(f"✓ Pool: {filter_pool_career}")
                            score += 1
                        
                        # Check EXCLUDE terms (must NOT match)
                        excluded = False
                        for excl in exclude_list:
                            if excl in combined_text:
                                excluded = True
                                break
                        if excluded:
                            continue  # Skip this officer
                        
                        # Check INCLUDE terms (must match)
                        include_matches = 0
                        for incl in include_list:
                            if incl in combined_text:
                                include_matches += 1
                                score += 3
                                if incl in train_hist:
                                    reasons.append(f"🎓 Training: '{incl}'")
                                elif incl in appt_hist:
                                    reasons.append(f"📋 Experience: '{incl}'")
                                else:
                                    reasons.append(f"✓ Contains: '{incl}'")
                        
                        # Require at least some includes to match (if provided)
                        if include_list and include_matches == 0:
                            continue
                        
                        # Add to results if score > 0
                        if score > 0 or (not include_list and not exclude_list):
                            matched_officers.append({
                                'row': row,
                                'score': score,
                                'reasons': reasons
                            })
                    
                    progress_bar.progress(0.7, text="📊 Ranking results...")
                    
                    # Sort by score
                    matched_officers.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Convert to MatchResult format for consistency
                    from dataclasses import dataclass
                    results = []
                    for m in matched_officers[:50]:
                        results.append(type('MatchResult', (), {
                            'entity_data': m['row'].to_dict(),
                            'final_score': m['score'] / 10.0,  # Normalize
                            'lgbm_score': None,
                            'match_reasons': m['reasons']
                        })())
                    
                    st.session_state['career_results'] = results
                    st.session_state['career_xai'] = {
                        'include': include_list,
                        'exclude': exclude_list,
                        'filters': {
                            'rank': filter_rank_career,
                            'branch': filter_branch_career,
                            'pool': filter_pool_career
                        }
                    }
                    
                    progress_bar.progress(1.0, text="✅ Complete!")
                    progress_bar.empty()
                else:
                    st.warning("Please enter at least one search term or filter")
            
            # Display Results
            if st.session_state.get('career_results'):
                results = st.session_state['career_results']
                xai = st.session_state.get('career_xai')
                
                # XAI Breakdown
                with st.expander("🧠 Search Criteria", expanded=True):
                    if xai and isinstance(xai, dict):
                        # New format with include/exclude
                        if xai.get('include'):
                            st.write(f"**✅ Include terms:** {', '.join(xai['include'])}")
                        if xai.get('exclude'):
                            st.write(f"**❌ Exclude terms:** {', '.join(xai['exclude'])}")
                        
                        filters = xai.get('filters', {})
                        filter_strs = []
                        if filters.get('rank') != 'Any':
                            filter_strs.append(f"Rank = {filters['rank']}")
                        if filters.get('branch') != 'Any':
                            filter_strs.append(f"Branch = {filters['branch']}")
                        if filters.get('pool') != 'Any':
                            filter_strs.append(f"Pool = {filters['pool']}")
                        if filter_strs:
                            st.write(f"**🎚️ Filters:** {', '.join(filter_strs)}")
                        
                        st.write(f"**📊 Found:** {len(results)} matching officers")
                
                # Results
                st.markdown(f"### Results ({len(results)} officers found)")
                
                for i, result in enumerate(results[:15]):
                    data = result.entity_data
                    score_pct = f"{result.final_score:.0%}"
                    lgbm_badge = f" | LightGBM: {result.lgbm_score:.0%}" if result.lgbm_score else ""
                    
                    with st.expander(f"#{i+1} | {score_pct} Match | {data.get('Employee_ID')} - {data.get('Rank')} {data.get('Name')}{lgbm_badge}"):
                        # Basic Info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**🏷️ Rank:** {data.get('Rank')}")
                            st.write(f"**🌿 Branch:** {data.get('Branch')}")
                        with col2:
                            st.write(f"**🌊 Pool:** {data.get('Pool')}")
                            st.write(f"**📅 Entry:** {data.get('Entry_type')}")
                        with col3:
                            st.write(f"**📊 Eval Avg:** {data.get('8_yr_avg_eval', 'N/A')}")
                            st.write(f"**🎖️ Current:** {data.get('current_appointment', 'N/A')[:40]}...")
                        
                        st.divider()
                        
                        # Match Reasons (XAI)
                        if result.match_reasons:
                            st.markdown("**🎯 Why Matched:**")
                            for reason in result.match_reasons[:5]:
                                st.write(f"  {reason}")
                            st.divider()
                        
                        # Detailed History
                        col_appt, col_train = st.columns(2)
                        
                        officer_id = data.get('Employee_ID', i)
                        
                        with col_appt:
                            st.markdown("**📋 Appointment History:**")
                            appt_hist = data.get('Appointment_history', 'N/A')
                            if appt_hist and appt_hist != 'N/A':
                                # Format nicely
                                appt_display = str(appt_hist)[:500]
                                if len(str(appt_hist)) > 500:
                                    appt_display += "..."
                                st.text_area("", appt_display, height=120, disabled=True, key=f"career_appt_{officer_id}", label_visibility="collapsed")
                            else:
                                st.caption("No appointment history available")
                        
                        with col_train:
                            st.markdown("**🎓 Training History:**")
                            train_hist = data.get('Training_history', 'N/A')
                            if train_hist and train_hist != 'N/A':
                                train_display = str(train_hist)[:500]
                                if len(str(train_hist)) > 500:
                                    train_display += "..."
                                st.text_area("", train_display, height=120, disabled=True, key=f"career_train_{officer_id}", label_visibility="collapsed")
                            else:
                                st.caption("No training history available")
                        
                        # Promotion History
                        promo_hist = data.get('Promotion_history', '')
                        if promo_hist:
                            st.markdown("**🚀 Promotion History:**")
                            st.caption(str(promo_hist)[:300])
        
        # =====================================================
        # TAB 2: BILLET SEARCH (Constraint-Based)
        # =====================================================
        with tab2:
            st.markdown("### Find Billets Using Natural Language")
            st.caption("Search billets by rank, branch, role type, or keywords")
            
            # Quick Filters
            st.markdown("#### 🎚️ Filters")
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                all_ranks = sorted(df['Rank'].unique())
                filter_rank = st.selectbox("Rank", ["Any"] + all_ranks, key="billet_rank_filter")
            
            with col_f2:
                all_branches = sorted(df['Branch'].unique())
                filter_branch = st.selectbox("Branch", ["Any"] + all_branches, key="billet_branch_filter")
            
            with col_f3:
                role_keywords = st.text_input(
                    "Role Keywords",
                    placeholder="e.g., instructor, head of, fleet...",
                    key="billet_keywords"
                )
            
            if st.button("🔍 Search Billets", type="primary", key="search_billets"):
                progress = st.progress(0, text="Searching billets...")
                
                # Get all roles from constraints
                all_roles = predictor.constraints.keys()
                matched_billets = []
                
                progress.progress(0.2, text="Filtering by constraints...")
                
                for role in all_roles:
                    const = predictor.constraints.get(role, {})
                    role_ranks = const.get('ranks', [])
                    role_branches = const.get('branches', [])
                    
                    match_reasons = []
                    score = 0
                    
                    # Filter by rank
                    if filter_rank != "Any":
                        if role_ranks and filter_rank not in role_ranks:
                            continue  # Skip - rank doesn't match
                        elif filter_rank in role_ranks:
                            match_reasons.append(f"✓ Rank: {filter_rank}")
                            score += 5
                    
                    # Filter by branch
                    if filter_branch != "Any":
                        if role_branches and filter_branch not in role_branches:
                            continue  # Skip - branch doesn't match
                        elif filter_branch in role_branches:
                            match_reasons.append(f"✓ Branch: {filter_branch}")
                            score += 5
                    
                    # Filter by keywords
                    if role_keywords:
                        keywords = [k.strip().lower() for k in role_keywords.split(',')]
                        role_lower = role.lower()
                        keyword_matches = [kw for kw in keywords if kw in role_lower]
                        if keywords and not keyword_matches:
                            continue  # Skip - no keyword match
                        for kw in keyword_matches:
                            match_reasons.append(f"✓ Contains: '{kw}'")
                            score += 3
                    
                    # If no filters applied, include all with base score
                    if filter_rank == "Any" and filter_branch == "Any" and not role_keywords:
                        score = 1
                    
                    if score > 0:
                        matched_billets.append({
                            'role': role,
                            'ranks': role_ranks,
                            'branches': role_branches,
                            'pools': const.get('pools', []),
                            'score': score,
                            'reasons': match_reasons
                        })
                
                progress.progress(0.8, text="Sorting results...")
                
                # Sort by score
                matched_billets.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['billet_results'] = matched_billets
                
                progress.progress(1.0, text="Complete!")
                progress.empty()
            
            # Display results
            if st.session_state.get('billet_results'):
                billets = st.session_state['billet_results']
                
                # XAI Summary
                with st.expander("🧠 Search Summary", expanded=True):
                    st.write(f"**Found:** {len(billets)} matching billets")
                    filters_used = []
                    if filter_rank != "Any":
                        filters_used.append(f"Rank = {filter_rank}")
                    if filter_branch != "Any":
                        filters_used.append(f"Branch = {filter_branch}")
                    if role_keywords:
                        filters_used.append(f"Keywords = {role_keywords}")
                    if filters_used:
                        st.write(f"**Filters:** {', '.join(filters_used)}")
                    else:
                        st.write("**Filters:** None (showing all billets)")
                
                st.markdown(f"### Results ({len(billets)} billets)")
                
                for i, billet in enumerate(billets[:20]):
                    with st.expander(f"#{i+1} | {billet['role']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**🎖️ Required Ranks:** {', '.join(billet['ranks']) if billet['ranks'] else 'Any'}")
                            st.write(f"**🌿 Required Branches:** {', '.join(billet['branches']) if billet['branches'] else 'Any'}")
                        with col2:
                            st.write(f"**🌊 Pools:** {', '.join(billet['pools']) if billet['pools'] else 'Any'}")
                        
                        if billet['reasons']:
                            st.markdown("**🎯 Why Matched:**")
                            for r in billet['reasons']:
                                st.success(r)
        
        # =====================================================
        # TAB 3: SIMILAR OFFICER FINDER
        # =====================================================
        with tab3:
            st.markdown("### Find Officers with Similar Careers")
            st.caption("Select an officer to find others with similar career trajectories")
            
            # Officer Selection
            col_sel1, col_sel2 = st.columns([2, 1])
            
            with col_sel1:
                # Filter options
                filter_branch_sim = st.selectbox("Filter by Branch", ["All"] + sorted(df['Branch'].unique()), key="sim_branch")
                
                filtered_df = df if filter_branch_sim == "All" else df[df['Branch'] == filter_branch_sim]
                
                officer_options = filtered_df.apply(
                    lambda r: f"{r['Employee_ID']} - {r['Rank']} {r['Name']} ({r['Branch']})", 
                    axis=1
                ).tolist()
                
                selected_officer = st.selectbox("Select Reference Officer", officer_options, key="sim_officer")
            
            with col_sel2:
                focus = st.text_input(
                    "Focus Area (optional)",
                    placeholder="e.g., instructor, HoD, fleet...",
                    key="sim_focus"
                )
            
            if st.button("🔍 Find Similar Officers", type="primary", key="find_similar"):
                if selected_officer:
                    progress = st.progress(0, text="Finding similar officers...")
                    
                    ref_id = int(selected_officer.split(' - ')[0])
                    ref_officer = df[df['Employee_ID'] == ref_id].iloc[0]
                    
                    progress.progress(0.2, text="Analyzing reference officer...")
                    
                    # Extract meaningful features from reference officer
                    ref_branch = ref_officer['Branch']
                    ref_rank = ref_officer['Rank']
                    ref_pool = ref_officer['Pool']
                    ref_entry = ref_officer['Entry_type']
                    ref_appt = str(ref_officer.get('Appointment_history', '')).lower()
                    ref_train = str(ref_officer.get('Training_history', '')).lower()
                    
                    # Identify key patterns
                    ref_has_hod = 'head of' in ref_appt or 'hod' in ref_appt
                    ref_has_instructor = 'instructor' in ref_appt
                    ref_has_staff = 'staff' in ref_appt
                    ref_has_ship = 'iss' in ref_appt or 'uss' in ref_appt or 'ship' in ref_appt
                    ref_has_masters = 'masters' in ref_train or 'master' in ref_train
                    ref_has_advanced = 'advanced' in ref_train or 'specialization' in ref_train
                    
                    progress.progress(0.4, text="Comparing officers...")
                    
                    # Find similar officers
                    similarities = []
                    for idx, row in df.iterrows():
                        if row['Employee_ID'] == ref_id:
                            continue
                        
                        sim_reasons = []
                        sim_score = 0
                        
                        # Same branch
                        if row['Branch'] == ref_branch:
                            sim_score += 3
                            sim_reasons.append(f"🌿 Same branch: {ref_branch}")
                        
                        # Same or adjacent rank
                        if row['Rank'] == ref_rank:
                            sim_score += 2
                            sim_reasons.append(f"🎖️ Same rank: {ref_rank}")
                        
                        # Same pool
                        if row['Pool'] == ref_pool:
                            sim_score += 1
                            sim_reasons.append(f"🌊 Same pool: {ref_pool}")
                        
                        # Same entry type
                        if row['Entry_type'] == ref_entry:
                            sim_score += 1
                            sim_reasons.append(f"📅 Same entry: {ref_entry}")
                        
                        # Career pattern matches
                        cand_appt = str(row.get('Appointment_history', '')).lower()
                        cand_train = str(row.get('Training_history', '')).lower()
                        
                        cand_has_hod = 'head of' in cand_appt or 'hod' in cand_appt
                        cand_has_instructor = 'instructor' in cand_appt
                        cand_has_staff = 'staff' in cand_appt
                        cand_has_ship = 'iss' in cand_appt or 'uss' in cand_appt or 'ship' in cand_appt
                        cand_has_masters = 'masters' in cand_train or 'master' in cand_train
                        cand_has_advanced = 'advanced' in cand_train or 'specialization' in cand_train
                        
                        if ref_has_hod and cand_has_hod:
                            sim_score += 2
                            sim_reasons.append("👔 Both served as Head of Department")
                        
                        if ref_has_instructor and cand_has_instructor:
                            sim_score += 2
                            sim_reasons.append("📚 Both served as Instructor")
                        
                        if ref_has_staff and cand_has_staff:
                            sim_score += 1
                            sim_reasons.append("🏢 Both had Staff appointments")
                        
                        if ref_has_ship and cand_has_ship:
                            sim_score += 1
                            sim_reasons.append("🚀 Both served on ships")
                        
                        if ref_has_masters and cand_has_masters:
                            sim_score += 2
                            sim_reasons.append("🎓 Both have Masters degree")
                        
                        if ref_has_advanced and cand_has_advanced:
                            sim_score += 1
                            sim_reasons.append("📜 Both have advanced training")
                        
                        # Focus area match
                        if focus:
                            if focus.lower() in cand_appt or focus.lower() in cand_train:
                                sim_score += 3
                                sim_reasons.append(f"🎯 Focus match: '{focus}'")
                        
                        if sim_score > 0:
                            similarities.append({
                                'row': row,
                                'score': sim_score,
                                'reasons': sim_reasons
                            })
                    
                    progress.progress(0.8, text="Ranking results...")
                    
                    similarities.sort(key=lambda x: x['score'], reverse=True)
                    st.session_state['similar_results'] = similarities[:20]
                    st.session_state['similar_ref'] = ref_officer
                    
                    progress.progress(1.0, text="Complete!")
                    progress.empty()
            
            # Display similar officers
            if st.session_state.get('similar_results'):
                ref = st.session_state.get('similar_ref')
                st.markdown(f"### Officers Similar to {ref['Name']}")
                
                with st.expander("📊 Reference Officer Profile", expanded=True):
                    col_ref1, col_ref2 = st.columns(2)
                    with col_ref1:
                        st.write(f"**{ref['Rank']} {ref['Name']}**")
                        st.write(f"**Branch:** {ref['Branch']}")
                        st.write(f"**Pool:** {ref['Pool']}")
                    with col_ref2:
                        st.write(f"**Current:** {ref['current_appointment']}")
                        st.write(f"**Entry:** {ref['Entry_type']}")
                
                for i, sim in enumerate(st.session_state['similar_results'][:10]):
                    row = sim['row']
                    score = sim['score']
                    
                    with st.expander(f"#{i+1} | Score: {score} | {row['Rank']} {row['Name']} ({row['Branch']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Current:** {row['current_appointment']}")
                            st.write(f"**Pool:** {row['Pool']}")
                        with col2:
                            st.write(f"**Entry:** {row['Entry_type']}")
                        
                        # Show meaningful similarity reasons
                        st.markdown("**🔍 Why Similar:**")
                        for reason in sim['reasons']:
                            st.success(reason)


    # =========================================================================
if __name__ == "__main__":
    main()
