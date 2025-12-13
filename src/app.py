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
    mode = st.sidebar.radio("Mode", ["Employee Lookup", "Simulation", "Billet Lookup", "Branch Analytics", "Dataset Explorer", "Admin Console"])
    
    # ... (Cache control) ...
    
    # Rank Flexibility Control (Global)
    with st.expander("⚙️ Advanced Settings (Rank Flexibility)"):
        rank_flex_up = st.slider("Rank Flexibility (Up)", 0, 2, 0, help="Allow promotion of N levels from predicted rank")
        rank_flex_down = st.slider("Rank Flexibility (Down)", 0, 2, 0, help="Allow demotion of N levels from predicted rank")
    
    if st.sidebar.button("🔄 Reload Models & Cache", help="Click if recent updates are not showing"):
        clear_cache()
        st.rerun()
    
    # =========================================================================
    # MODE 5: DATASET EXPLORER
    # =========================================================================
    if mode == "Dataset Explorer":
        st.header("💾 Dataset HR Explorer")
        st.markdown("Interactive exploration of the workforce hierarchy, organizational structure, and career paths.")

        tab_hier, tab_org, tab_flow, tab_stats = st.tabs([
            "👑 HR Hierarchy", 
            "🏢 Org Structure", 
            "🔀 Career Path Flow",
            "📊 Statistics"
        ])
        
        # --- TAB 1: HR HIERARCHY ---
        with tab_hier:
            st.markdown("### Multi-Level Hierarchy Analysis")
            import plotly.express as px
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                hier_path = st.multiselect("Hierarchy Path", ['Branch', 'Rank', 'Pool', 'Entry_type'], default=['Branch', 'Rank', 'Pool'])
            with h_col2:
                color_by = st.selectbox("Color By", ['Count', 'Avg Service (Years)', 'Avg Eval Score'])
                
            if hier_path:
                # Aggregate
                agg_cols = hier_path
                # We need metrics. Join with specific columns.
                # Eval Score? Not in sample dataset rows explicitly as number, mainly textual history? 
                # Actually, check DataProcessor features. 
                # Let's count and sum service years.
                
                # Check metrics available in df
                # 'years_service' isn't in df unless calculated by FeatureEngineer. 
                # We should run FE on full df if not present, OR do quick calc.
                # Predictor loads df but maybe doesn't save FE result to main 'df' var?
                # 'df' comes from load_data(). It is RAW.
                
                # Quick enrichment if needed
                if 'years_service' not in df.columns:
                     # Simple approx
                     # Parse Date? Expensive for UI.
                     # Let's check if we can reuse predictor's logic or just skip complex metrics for now.
                     pass
                
                # GroupBy
                # We need a 'count' column
                df_viz = df.copy()
                df_viz['count'] = 1
                
            if hier_path:
                # Aggregate
                # We need a 'count' column
                df_viz = df.copy()
                df_viz['count'] = 1
                
                if color_by == 'Count':
                    fig = px.sunburst(df_viz, path=hier_path, values='count', 
                                      color='count', color_continuous_scale='Viridis',
                                      title=f"Workforce by {' > '.join(hier_path)}")
                else:
                    st.info("Metric-based coloring requires pre-calculated metrics. Using Count for now.")
                    fig = px.sunburst(df_viz, path=hier_path, values='count',
                                      title=f"Workforce by {' > '.join(hier_path)}")
                
                # SIMPLIFY: Clean text, detail in tooltip
                fig.update_traces(textinfo='label', hoverinfo='label+value+percent entry')
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Select at least one level for hierarchy.")

        # --- TAB 2: ORG STRUCTURE ---
        with tab_org:
            st.markdown("### 🏢 Organizational Unit Browser")
            st.info("Drill down from high-level Units to individual Posts.")
            
            # 1. Parse Units Heuristically
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

            # Check if cached
            if 'Unit' not in df.columns:
                with st.spinner("Parsing Organizational Structure..."):
                    df[['Org_Unit', 'Org_Post']] = df['current_appointment'].apply(lambda x: pd.Series(extract_unit(x)))
            
            # Tree Map: Unit -> Post
            org_view = df.groupby(['Org_Unit', 'Org_Post']).size().reset_index(name='count')
            
            # Search Filter
            search_unit = st.text_input("Search Unit", "")
            if search_unit:
                org_view = org_view[org_view['Org_Unit'].str.contains(search_unit, case=False)]
            
            if len(org_view) > 0:
                fig_tree = px.treemap(org_view, path=['Org_Unit', 'Org_Post'], values='count',
                                      color='count', color_continuous_scale='Blues',
                                      title="Organizational Command Structure")
                fig_tree.update_layout(height=800) # Taller for more data
                fig_tree.update_traces(textinfo="label+value", root_color="lightgrey")
                st.plotly_chart(fig_tree, use_container_width=True)
                
                # Leaf Detail Picker
                st.divider()
                st.markdown("#### 👤 Unit Roster")
                sel_unit = st.selectbox("Select Unit to Inspect", sorted(org_view['Org_Unit'].unique()))
                
                if sel_unit:
                    roster = df[df['Org_Unit'] == sel_unit]
                    st.dataframe(roster[['Employee_ID', 'Rank', 'Name', 'current_appointment', 'Branch']], hide_index=True)
            else:
                st.warning("No units match search.")

        # --- TAB 3: CAREER PATH FLOW (SANKEY) ---
        with tab_flow:
            st.markdown("### 🔀 Interactive Career Path Explorer")
            st.caption("Trace how officers move between specific roles. Use filters to focus the diagram.")
            import plotly.graph_objects as go
            
            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_branch = st.multiselect("Filter Branch", sorted(df['Branch'].unique()), default=[])
            with fc2:
                f_rank = st.multiselect("Filter Rank", sorted(df['Rank'].unique()), default=[])
            with fc3:
                use_generic = st.checkbox("Group by Generic Titles", value=True, help="Merges 'Div Officer USS A' and 'Div Officer USS B' into 'Div Officer'")

            # Data Prep
            sankey_df = df.copy()
            if f_branch: sankey_df = sankey_df[sankey_df['Branch'].isin(f_branch)]
            if f_rank: sankey_df = sankey_df[sankey_df['Rank'].isin(f_rank)]
            
            # Helper to generate transitions
            transitions = []
            
            # Regex for generic cleaning
            def clean_title(t):
                if not use_generic: return t.strip()
                t = re.sub(r'\s*\/.*', '', t) # Remove post
                t = re.sub(r'\s*USS .*', '', t) # Remove Ship
                t = re.sub(r'\s*Starbase .*', '', t) # Remove Starbase
                return t.strip()

            import re
            
            for _, row in sankey_df.iterrows():
                # Parse History
                hist_str = row['Appointment_history']
                if not isinstance(hist_str, str): continue
                
                # Clean and Split
                raw_items = hist_str.split(',')
                roles = []
                for item in raw_items:
                    clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                    if clean:
                        roles.append(clean_title(clean))
                
                # Chain
                if len(roles) >= 2:
                    for i in range(len(roles)-1):
                        transitions.append((roles[i], roles[i+1]))
            
            # --- NETWORK GRAPH IMPLEMENTATION ---
            import networkx as nx
            import numpy as np
            
            # 1. Build Transitions & Identify Ranks
            transitions = []
            node_rank_evidence = {} # Map Role -> List of Ranks seen
            
            # Helper to clean titles
            def clean_title_network(t):
                if not use_generic: return t.strip()
                t = re.sub(r'\s*\/.*', '', t) # Remove post
                t = re.sub(r'\s*USS .*', '', t) # Remove Ship
                t = re.sub(r'\s*Starbase .*', '', t) # Remove Starbase
                return t.strip()
            
            # We process rows to get Transitions AND Role->Rank mapping
            for _, row in sankey_df.iterrows():
                row_rank = row['Rank']
                hist_str = row['Appointment_history']
                
                if not isinstance(hist_str, str): continue
                
                # Split history
                raw_items = hist_str.split(',')
                roles = []
                for item in raw_items:
                    clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                    if clean:
                        final_role = clean_title_network(clean)
                        roles.append(final_role)
                        
                        # Collect Rank Evidence
                        # We assume the user's CURRENT rank applies to their LAST role? 
                        # Or do we rely on the row['Rank'] for the *current* role?
                        # Heuristic: The *last* role in the history list tends to be the Current Appointment title.
                        # So we associate row['Rank'] with the LAST role in the list.
                        # For previous roles, we don't know the rank for sure, but we can infer from other rows where that role IS the current one.
                        
                if roles:
                    # Current Role (Last item) gets the Current Rank
                    curr_role = roles[-1]
                    if curr_role not in node_rank_evidence: node_rank_evidence[curr_role] = []
                    node_rank_evidence[curr_role].append(row_rank)
                    
                    # Transitions
                    if len(roles) >= 2:
                        for i in range(len(roles)-1):
                            transitions.append((roles[i], roles[i+1]))

            if transitions:
                # 2. Define Rank Hierarchy (Master List)
                # We use this ONLY for sorting. We only display ranks actually present in data.
                master_rank_order = [
                    'Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 
                    'Commander', 'Captain', 'Commodore', 'Rear Admiral', 'Vice Admiral', 'Admiral'
                ]
                # Fallback for unknown ranks: put them at bottom or top
                
                # Determine "Actual" Ranks present in this filtered dataset
                present_ranks = sankey_df['Rank'].unique()
                
                # Sort present ranks based on master order
                sorted_ranks = []
                for mr in master_rank_order:
                    if mr in present_ranks:
                        sorted_ranks.append(mr)
                
                # Add any unknown ranks that weren't in master list (e.g. 'Cadet')
                for pr in present_ranks:
                    if pr not in sorted_ranks:
                        sorted_ranks.insert(0, pr) # Put unknowns at start (junior)
                        
                # 3. Resolve Node Ranks
                # Map each Role to its Mode Rank (most frequent)
                node_assigned_rank = {}
                for role, ranks in node_rank_evidence.items():
                    if not ranks: continue
                    from collections import Counter
                    common = Counter(ranks).most_common(1)
                    node_assigned_rank[role] = common[0][0]
                
                # Also include nodes involved in transitions but maybe not as "current" (last) role
                # These might lack direct rank evidence. We assign them to 'Ensign' or infer?
                # For now, if no evidence, skip or assign lowest.
                
                # Build Graph
                G = nx.DiGraph()
                
                # Add edges (weighted)
                t_counts = Counter(transitions)
                for (u, v), w in t_counts.items():
                    G.add_edge(u, v, weight=w)
                
                # Define all_nodes for usage in other charts
                all_nodes = list(G.nodes())
                
                # 4. Metrics for Visuals (Centrality, Communities)
                # "Visualize entire appointment space" -> Clusters of related roles.
                # k parameter controls spacing.
                pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
                
                # Metrics for Visuals
                # Centrality -> Node Size
                centrality = nx.degree_centrality(G)
                # Communities -> Node Color
                # Use greedy_modularity_communities (undirected for community)
                from networkx.algorithms import community
                try:
                    communities = community.greedy_modularity_communities(G.to_undirected())
                    # Map node -> community ID
                    comm_map = {}
                    for i, comm in enumerate(communities):
                        for n in comm:
                            comm_map[n] = i
                except:
                    comm_map = {n: 0 for n in G.nodes()}

                # 5. Advanced ECharts Visualizations
                from streamlit_echarts import st_echarts
                import random
                
                # Selector
                viz_type = st.radio(
                    "Visualization Mode", 
                    ["🕸️ Network Graph (Relationship Map)", "🌊 Sankey Diagram (Flow)", "🗺️ Career Lattice (Rank vs Branch)"], 
                    horizontal=True
                )
                
                # --- A. NETWORK GRAPH (FIXED LES MIS STYLE) ---
                if "Network" in viz_type:
                    # 1. Cluster Naming Heuristic
                    cluster_names = {}
                    unique_ids = sorted(list(set(comm_map.values())))
                    
                    for cid in unique_ids:
                        nodes_in_c = [n for n, c in comm_map.items() if c == cid]
                        words = []
                        for n in nodes_in_c:
                            parts = re.split(r'[\s\(\)\-\/]+', n)
                            for p in parts:
                                if len(p) > 3 and p.lower() not in ['officer', 'chief', 'asst', 'vice', 'staff']:
                                    words.append(p)
                        
                        from collections import Counter
                        if words:
                            top_word = Counter(words).most_common(1)[0][0]
                            cluster_time = f"{top_word} Group"
                        else:
                            ranks = [node_assigned_rank.get(n, "") for n in nodes_in_c]
                            if ranks:
                                top_rank = Counter(ranks).most_common(1)[0][0]
                                cluster_time = f"{top_rank} Group"
                            else:
                                cluster_time = f"Cluster {cid}"
                            
                        cluster_names[cid] = cluster_time

                    # 2. Caching Layout (Prevent Resets)
                    layout_key = f"layout_{hash(tuple(sorted(list(G.nodes()))))}"
                    
                    if layout_key not in st.session_state:
                         # Calculate optimal layout once
                         st.session_state[layout_key] = nx.spring_layout(G, k=0.15, iterations=100, seed=42)
                    
                    pos_screen = st.session_state[layout_key]
                    
                    # Prepare Nodes with Fixed X, Y
                    echarts_nodes = []
                    
                    vals = np.array(list(pos_screen.values()))
                    min_v = vals.min(axis=0)
                    max_v = vals.max(axis=0)
                    
                    node_list = list(G.nodes())
                    categories = [{"name": cluster_names[cid]} for cid in unique_ids]
                    
                    for i, node in enumerate(node_list):
                        # Safe normalization
                        denom_x = (max_v[0] - min_v[0]) if (max_v[0] - min_v[0]) != 0 else 1
                        denom_y = (max_v[1] - min_v[1]) if (max_v[1] - min_v[1]) != 0 else 1
                        
                        x_norm = (pos_screen[node][0] - min_v[0]) / denom_x
                        y_norm = (pos_screen[node][1] - min_v[1]) / denom_y
                        
                        # Scale
                        x_fixed = x_norm * 1000 - 500
                        y_fixed = y_norm * 800 - 400
                        
                        deg = G.degree(node)
                        sz = min(10 + deg*3, 60)
                        
                        cid = comm_map.get(node, 0)
                        cat_name = cluster_names[cid]
                        
                        try:
                            cat_idx = next(i for i, c in enumerate(categories) if c["name"] == cat_name)
                        except:
                            cat_idx = 0
                        
                        echarts_nodes.append({
                            "name": node,
                            "x": x_fixed,
                            "y": y_fixed,
                            "symbolSize": sz,
                            "category": cat_idx,
                            "value": deg,
                            "label": {"show": deg > 5}
                        })

                    # Links
                    echarts_links = [{"source": u, "target": v} for u, v in G.edges()]
                    
                    option = {
                        "title": {
                            "text": "Appointment Relationship Map",
                            "subtext": "Fixed Layout (Cached)",
                            "top": "bottom",
                            "left": "right"
                        },
                        "tooltip": {},
                        "legend": [{
                            "data": [c["name"] for c in categories],
                            "type": "scroll", 
                            "orient": "horizontal",
                            "top": 10
                        }],
                        "series": [{
                            "name": "Roles",
                            "type": "graph",
                            "layout": "none", # FIXED POSITIONS
                            "data": echarts_nodes,
                            "links": echarts_links,
                            "categories": categories,
                            "roam": True,
                            "label": {
                                "position": "right",
                                "formatter": "{b}"
                            },
                            "lineStyle": {
                                "color": "source",
                                "curveness": 0.3
                            },
                            "emphasis": {
                                "focus": "adjacency",
                                "lineStyle": {"width": 5}
                            }
                        }]
                    }
                    st_echarts(option, height="700px")
                    st.caption("ℹ️ **Stable Layout**: Graph positions are cached. Changing filters will regenerate the layout.")
                    
                # --- B. SANKEY (DAG ENFORCED) ---
                elif "Sankey" in viz_type:
                    # 1. Build Temporary Graph
                    G_sankey = nx.DiGraph()
                    t_counts = Counter(transitions)
                    for (u, v), w in t_counts.items():
                        if w > 0: G_sankey.add_edge(u, v, weight=w)
                    
                    # 2. Strict DAG Enforcement (The "Rigorous" Fix)
                    # Sankey fails on cycles. Remove self-loops first.
                    G_sankey.remove_edges_from(nx.selfloop_edges(G_sankey))
                    
                    # Iteratively remove cycles
                    # Limit iterations to avoid infinite loops in worst case
                    max_breaks = 100
                    breaks = 0
                    while not nx.is_directed_acyclic_graph(G_sankey) and breaks < max_breaks:
                        try:
                            cycle = nx.find_cycle(G_sankey)
                            # Remove the edge with lowest weight in the cycle to preserve main flows
                            # (Simple heuristic: just remove the first edge found in cycle for speed)
                            G_sankey.remove_edge(*cycle[0][:2]) 
                            breaks += 1
                        except:
                            break
                            
                    # 3. Build Safe Data
                    slinks = []
                    found_nodes = set()
                    
                    for u, v, data in G_sankey.edges(data=True):
                        w = data.get('weight', 1)
                        slinks.append({"source": str(u), "target": str(v), "value": int(w)})
                        found_nodes.add(str(u))
                        found_nodes.add(str(v))

                    if not slinks:
                        st.warning("⚠️ No valid acyclic career paths found. Try expanding your filters.")
                    else:
                        snodes = [{"name": n} for n in sorted(list(found_nodes))]
                        
                        option = {
                            "title": {"text": "Career Transition Flow (Acyclic)", "left": "center"},
                            "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
                            "series": [{
                                "type": "sankey",
                                "data": snodes,
                                "links": slinks,
                                "orient": "horizontal",
                                "label": {"position": "right"},
                                "lineStyle": {"color": "source", "curveness": 0.5},
                                "layoutIterations": 32 
                            }]
                        }
                        st_echarts(option, height="800px")
                        if breaks > 0:
                            st.caption(f"🌊 **Sankey Diagram**: Visualizing linear flows. Note: {breaks} cyclic loops were hidden to ensure rendering.")
                        else:
                            st.caption("🌊 **Sankey Diagram**: Shows volume of movement between roles.")
                    
                # --- C. CAREER LATTICE (JITTERED) ---
                elif "Lattice" in viz_type:
                    # 1. Setup Axes
                    if 'sorted_ranks' not in locals():
                        sorted_ranks = ["Ensign", "Lieutenant (jg)", "Lieutenant", "Lieutenant Commander", "Commander", "Captain", "Commodore", "Rear Admiral", "Vice Admiral", "Admiral"]
                    
                    # Map Nodes to Branches
                    role_branch_groups = sankey_df.groupby('current_appointment')['Branch'].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
                    node_branch_map = role_branch_groups.to_dict()
                    present_branches = sorted(list(set([node_branch_map.get(n, "Unknown") for n in G.nodes()])))
                    
                    # 2. Build Nodes with JITTER and NO LABELS (by default)
                    lattice_nodes = []
                    
                    for node in G.nodes():
                        # Y: Branch
                        b = node_branch_map.get(node, "Unknown")
                        try: y_idx = present_branches.index(b)
                        except: y_idx = 0
                            
                        # X: Rank
                        r = node_assigned_rank.get(node, "Unknown")
                        try: x_idx = sorted_ranks.index(r)
                        except: x_idx = 0
                        
                        # Add Jitter to prevent overlap
                        rng = random.Random(str(node)) # Deterministic jitter based on name
                        x_jit = x_idx + rng.uniform(-0.35, 0.35)
                        y_jit = y_idx + rng.uniform(-0.35, 0.35)
                        
                        lattice_nodes.append({
                            "name": node,
                            "value": [x_jit, y_jit], # JITTERED COORDINATES
                            "symbolSize": 10 + (G.degree(node)*1.5),
                            "itemStyle": {"color": "#5470c6"},
                            # HIDE LABELS to reduce clutter
                            "label": {"show": False, "formatter": "{b}"},
                            "emphasis": {
                                "label": {"show": True, "position": "top", "fontWeight": "bold"}
                            }
                        })
                        
                    # 3. Links
                    lattice_links = [{"source": u, "target": v} for u, v in G.edges()]
                    
                    option = {
                        "title": {"text": "Career Lattice (Rank vs Branch)", "left": "center"},
                        "tooltip": {"trigger": "item"},
                        "grid": {
                            "left": "15%", "right": "15%", "top": "10%", "bottom": "10%"
                        },
                        "xAxis": {
                            "type": "category",
                            "data": sorted_ranks,
                            "name": "Rank Progression",
                            "nameLocation": "middle",
                            "nameGap": 30
                        },
                        "yAxis": {
                            "type": "category",
                            "data": present_branches,
                            "name": "Branch / Pool",
                            "nameLocation": "end"
                        },
                        "series": [{
                            "type": "graph",
                            "layout": "none",
                            "coordinateSystem": "cartesian2d",
                            "data": lattice_nodes,
                            "links": lattice_links,
                            "edgeSymbol": ['none', 'arrow'],
                            "edgeSymbolSize": [4, 8],
                            "lineStyle": {
                                "color": "#ccc",
                                "curveness": 0.1,
                                "width": 1,
                                "opacity": 0.6
                            },
                            "roam": True
                        }]
                    }
                    
                    st_echarts(option, height="800px")
                    st.caption("🗺️ **Career Lattice**: Roles are plotted by Rank (X) and Branch (Y). **Nodes are jittered** to show overlaps. Hover to see names.")
            
            else:
                st.info("No transition data for selected.")


        # --- TAB 4: STATISTICS ---
        with tab_stats:
            st.markdown("### 📊 Dataset Demographics")
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Officers", len(df))
            k2.metric("Unique Roles", df['current_appointment'].nunique())
            k3.metric("Branches", df['Branch'].nunique())
            k4.metric("Avg Service", "N/A") # Placeholder
            
            # Charts
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rank Distribution**")
                st.bar_chart(df['Rank'].value_counts())
            with c2:
                st.markdown("**Branch Distribution**")
                st.bar_chart(df['Branch'].value_counts())

    # =========================================================================
    # MODE 6: ADMIN CONSOLE (RETRAINING HUD)
    # =========================================================================
    if mode == "Admin Console":
        st.header("🛠️ Model Management Console")
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
                st.success("✅ Dataset Structure Validated")
                
                # 2. Training Trigger
                st.subheader("2. Retrain Model")
                st.info("Training will generate a new 'Staging' model. Production is not affected until you deploy.")
                
                if st.button("🚀 Start Training Pipeline"):
                    with st.status("Running Training Pipeline...", expanded=True) as status:
                        st.write("Initializing...")
                        
                        # Run Training
                        session_id, result = manager.train_staging_model(path)
                        
                        if session_id:
                            st.write("✅ Data Processing Complete")
                            st.write("✅ LTR Model Trained")
                            st.write("✅ Artifacts Generated")
                            status.update(label="Training Complete!", state="complete", expanded=False)
                            
                            st.session_state.training_session = session_id
                            st.session_state.training_metrics = result # Save metrics
                            st.success(f"Training Success! Session ID: {session_id}")
                        else:
                            status.update(label="Training Failed", state="error")
                            st.error(f"Pipeline Error: {result}")
            else:
                st.error(f"❌ Validation Failed: {msg}")
                
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
                if st.button("📢 Deploy to Production", type="primary"):
                    success, msg = manager.commit_model(st.session_state.training_session)
                    if success:
                        st.success(f"DEPLOYED! {msg}")
                        st.cache_resource.clear()
                        st.balloons()
                    else:
                        st.error(f"Deployment Failed: {msg}")
            
            with col_d2:
                if st.button("🗑️ Discard Staging Model"):
                    st.session_state.training_session = None
                    st.rerun()

        # 4. Rollback
        st.divider()
        with st.expander("⚠️ Danger Zone: Rollback"):
            st.warning("Restore the previous model version if the current one is unstable.")
            if st.button("⏪ Rollback to Last Backup"):
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
                                
