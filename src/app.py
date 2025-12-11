import streamlit as st
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

@st.cache_resource(ttl=3600)
def load_predictor():
    return Predictor()

# --- CACHE CLEARING UTILITY ---
def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)

def main():
    st.title("🚀 AI Appointer Assist")
    st.markdown("AI-powered system for recommending next-best assignments based on career history.")
    
    # Load Resources
    with st.spinner("Loading AI Models..."):
        predictor = load_predictor()
        df = load_data()
        
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

                st.dataframe(
                    results.style.format({"Confidence": "{:.1%}"})
                           .background_gradient(subset=["Confidence"], cmap="Greens"),
                    use_container_width=True,
                    column_config={
                        "Prediction": st.column_config.TextColumn("Role", width="medium"),
                        "Explanation": st.column_config.TextColumn("Reasoning", width="large")
                    }
                )
                
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
            st.plotly_chart(fig, use_container_width=True)
            
    elif mode == "Simulation":
        st.header("🎮 Career Simulation Playground")
        st.markdown("Explore hypothetical career scenarios with intelligent constraints.")
        
        col1, col2 = st.columns(2)
        with col1:
            rank = st.selectbox("Rank", sorted(df['Rank'].unique()))
            branch = st.selectbox("Branch", sorted(df['Branch'].unique()))
            pool = st.selectbox("Pool", sorted(df['Pool'].unique()))
            
        with col2:
            entry = st.selectbox("Entry Type", sorted(df['Entry_type'].unique()))
            
            # Smart role selection: Filter roles by selected rank and branch
            # Get roles from constraints that match the selected rank and branch
            import json
            with open('models/all_constraints.json') as f:
                constraints = json.load(f)
            
            # Find roles that allow this rank and branch
            valid_roles = []
            for role_name, role_const in constraints.items():
                allowed_ranks = role_const.get('ranks', [])
                allowed_branches = role_const.get('branches', [])
                
                # Check if role is valid for selected rank and branch
                if rank in allowed_ranks and branch in allowed_branches:
                    valid_roles.append(role_name)
            
            if valid_roles:
                valid_roles = sorted(valid_roles)
                last_role = st.selectbox(
                    "Current Role",
                    valid_roles,
                    help=f"Showing roles valid for {rank} in {branch} branch"
                )
            else:
                st.warning(f"No roles found for {rank} in {branch} branch. Using fallback.")
                last_role = "Generic Officer Role"
        
        # Auto-calculate years of service based on rank
        rank_to_years = {
            'Ensign': 2,
            'Lieutenant (jg)': 4,
            'Lieutenant': 7,
            'Lieutenant Commander': 11,
            'Commander': 16,
            'Captain': 22,
            'Commodore': 28,
            'Rear Admiral': 32
        }
        years_service = rank_to_years.get(rank, 5)
        
        st.info(f"📊 Auto-calculated: ~{years_service} years of service for {rank}")
        
        if st.button("🔮 Run Simulation"):
            # Create realistic dummy data
            dummy_data = {
                'Employee_ID': [999999],
                'Rank': [rank],
                'Branch': [branch],
                'Pool': [pool],
                'Entry_type': [entry],
                'Appointment_history': [f"{last_role} (01 JAN 2300 - )"],
                'Training_history': [f"Basic Training (01 JAN 2290 - 01 FEB 2290), Advanced {branch} Course (01 JAN 2295 - 01 JUN 2295)"], 
                'Promotion_history': [f"{rank} (01 JAN 2300 - )"],
                'current_appointment': [last_role],
                'appointed_since': ["01/01/2300"]
            }
            
            dummy_df = pd.DataFrame(dummy_data)
            
            # Predict with current rank_flexibility setting
            with st.spinner("Simulating career trajectory..."):
                try:
                    results = predictor.predict(dummy_df, rank_flex_up=rank_flex_up, rank_flex_down=rank_flex_down)
                    
                    st.subheader("🎯 Simulation Results")
                    st.dataframe(
                        results.style.format({"Confidence": "{:.1%}"})
                               .background_gradient(subset=["Confidence"], cmap="Greens", vmin=0, vmax=1.0),
                        width="stretch",
                        hide_index=True
                    )
                    
                    best_role = results.iloc[0]['Prediction']
                    confidence = results.iloc[0]['Confidence']
                    
                    if confidence < 0.15:
                        st.warning("⚠️ Low confidence. Try adjusting Rank Flexibility slider.")
                    else:
                        st.success(f"✨ Top Prediction: **{best_role}** ({confidence:.1%} confidence)")
                        
                except Exception as e:
                    st.error(f"Simulation Error: {e}")
                    st.info("Try adjusting the Rank Flexibility slider or selecting a different role.")
            
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
                        
                        st.success(f"Found {len(match_df)} recommended candidates.")
                        st.dataframe(
                            match_df.style.format({"Confidence": "{:.1%}"})
                                    .background_gradient(subset=["Confidence"], cmap="Greens"),
                            width="stretch",
                            column_config={
                                "Employee_ID": st.column_config.NumberColumn("Employee ID", format="%d"),
                                "Name": st.column_config.TextColumn("Name", width="medium"),
                                "Rank": st.column_config.TextColumn("Rank", width="small"),
                                "Branch": st.column_config.TextColumn("Branch", width="medium"),
                                "Current_Role": st.column_config.TextColumn("Current Role", width="medium"),
                                "Explanation": st.column_config.TextColumn("Reasoning", width="large")
                            }
                        )
                    else:
                        st.warning("No suitable candidates found for this role.")

if __name__ == "__main__":
    main()
