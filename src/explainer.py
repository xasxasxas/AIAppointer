import pandas as pd
import re
import difflib
import plotly.graph_objects as go

class Explainer:
    def __init__(self, df=None, known_titles=None):
        self.transition_index = {} # (from, to) -> [(id, name, rank, branch)]
        self.known_titles = known_titles if known_titles else []
        if df is not None:
            self.index_transitions(df)

    def _normalize_title(self, raw_title):
        """
        Maps raw history title to canonical known title using Fuzzy logic.
        Matches Predictor logic to ensure Index keys align with Probability keys.
        """
        if not self.known_titles:
            return raw_title.strip()
            
        t = raw_title.strip()
        # 1. Exact
        if t in self.known_titles:
            return t
        
        # 2. Substring (Case Insensitive)
        t_lower = t.lower()
        for k in self.known_titles:
            if k.lower() in t_lower: # Role name inside history string? "Ensign (jg)" contains "Ensign"?
                return k
            if t_lower in k.lower(): # Reverse? "Ensign" inside "Ensign Role"?
                return k
                
        # 3. Fuzzy
        matches = difflib.get_close_matches(t, self.known_titles, n=1, cutoff=0.4)
        if matches:
            return matches[0]
            
        return t

    def index_transitions(self, df):
        """
        Builds a map of (FromTitle, ToTitle) -> List of Officers who made this move.
        Also builds Role -> Training Profile map (Consensus Training).
        """
        self.role_training_counts = {} # role -> Counter(course -> count)
        self.role_incumbent_counts = {} # role -> total_unique_officers
        
        # Iterate over all officers and parse their history
        for idx, row in df.iterrows():
            emp_id = row.get('Employee_ID', 'Unknown')
            name = row.get('Name', f"Officer {emp_id}")
            rank = row.get('Rank', 'Unknown')
            branch = row.get('Branch', 'Unknown')
            
            # Parse Training
            train_raw = row.get('Training_history', '')
            courses = set()
            if isinstance(train_raw, str) and train_raw:
                # Split by comma
                for t in train_raw.split(','):
                    # Clean dates: "Course (Date)" -> "Course"
                    # Regex: Remove parenthesis and everything inside it
                    clean_course = re.sub(r'\s*\(.*?\)', '', t).strip()
                    if clean_course: courses.add(clean_course)
            
            hist_raw = row.get('Appointment_history', '')
            if not isinstance(hist_raw, str):
                continue
                
            # Parse history (Same logic as Predictor)
            items = str(hist_raw).split(',')
            titles = []
            for item in items:
                clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                if clean: 
                    norm = self._normalize_title(clean)
                    titles.append(norm)
            
            unique_roles_held = set(titles)
            for role in unique_roles_held:
                if role not in self.role_training_counts:
                    self.role_training_counts[role] = {}
                    self.role_incumbent_counts[role] = 0
                
                self.role_incumbent_counts[role] += 1
                
                for course in courses:
                    self.role_training_counts[role][course] = self.role_training_counts[role].get(course, 0) + 1
            
            if len(titles) < 2:
                continue
            
            for i in range(len(titles) - 1):
                t_from = titles[i]
                t_to = titles[i+1]
                
                key = (t_from, t_to)
                if key not in self.transition_index:
                    self.transition_index[key] = []
                
                existing = self.transition_index[key]
                if existing and existing[-1]['Employee_ID'] == emp_id:
                    continue
                    
                self.transition_index[key].append({
                    'Employee_ID': emp_id,
                    'Name': name,
                    'Rank': rank,
                    'Branch': branch,
                    'Pool': row.get('Pool', 'Unknown'),
                    'Entry_type': row.get('Entry_type', 'Unknown'),
                    'Appointment_history': hist_raw,
                    'Training_history': train_raw,
                    'From': t_from,
                    'To': t_to
                })
                
    def get_precedents(self, from_title, to_title, limit=5, fuzzy=False):
        """
        Finds officers who moved From -> To.
        Auto-normalizes inputs to match Index keys.
        """
        # Normalize inputs to match how we indexed them
        k_from = self._normalize_title(from_title)
        k_to = self._normalize_title(to_title)
        
        key = (k_from, k_to)
        
        matches = self.transition_index.get(key, [])
        
        if not matches and fuzzy:
            # Fuzzy Fallback
            # Check for keys that look like (from, to)
            import difflib
            
            # Map known titles
            close_from = difflib.get_close_matches(k_from, self.known_titles, n=1, cutoff=0.6)
            close_to = difflib.get_close_matches(k_to, self.known_titles, n=1, cutoff=0.6)
            
            if close_from and close_to:
                fuzzy_key = (close_from[0], close_to[0])
                matches = self.transition_index.get(fuzzy_key, [])
            
        return matches[:limit]

    def get_relevant_experience(self, target_role, candidate_history, limit=3):
        """
        Analyzes candidate's past roles to see if any are known feeders for the target role.
        """
        # 1. Identify all Feeders for Target Role
        # Scan transition index: (Any, Target) -> Count
        norm_target = self._normalize_title(target_role)
        feeders = {} # role -> frequency
        
        for (t_from, t_to), officers in self.transition_index.items():
            if t_to == norm_target:
                feeders[t_from] = feeders.get(t_from, 0) + len(officers)
                
        if not feeders:
            return []
            
        # 2. Check Candidate History
        # Parse history string 'Role A > Role B > ...'
        cand_roles = []
        if isinstance(candidate_history, str):
             # Try clean splits first
             import re
             # Split by ">" if we pre-processed it, or "," from raw
             if ">" in candidate_history:
                 parts = candidate_history.split('>') 
             else:
                 parts = candidate_history.split(',')
                 
             for p in parts:
                 clean = re.sub(r'\s*\(.*?\)', '', p).strip()
                 if clean: cand_roles.append(self._normalize_title(clean))
        
        # 3. Find Intersection
        hits = []
        seen = set()
        for role in cand_roles:
            if role in feeders and role not in seen:
                hits.append({
                    'role': role,
                    'frequency': feeders[role],
                    'desc': f"Historically, {feeders[role]} officers promoted to '{target_role}' from type '{role}'."
                })
                seen.add(role)
                
        # Sort by frequency (Relevance)
        hits.sort(key=lambda x: x['frequency'], reverse=True)
        return hits[:limit]

    def format_feature_explanation(self, feats, score=0.0, constraints=None, contribs=None, mode='employee_lookup'):
        """
        Converts raw feature dict into rich human-readable explanation objects.
        Merges related metrics to save space and provide clearer context.
        
        Args:
            mode: 'employee_lookup' or 'billet_lookup' - changes feature emphasis/labels
        """
        ctx = feats.get('_Context', {})
        t_from = ctx.get('From_Title', 'Unknown')
        t_to = ctx.get('To_Title', 'Unknown')
        p_from = ctx.get('From_Pool', 'Unknown')
        p_to = ctx.get('To_Pool', 'Unknown')
        
        # --- CONTEXT-AWARE FEATURE LABELS ---
        # Different emphasis for Billet Lookup (candidate selection) vs Employee Lookup (role finding)
        if mode == 'billet_lookup':
            FEATURE_LABELS = {
                'rank_match_exact': 'Meets Rank Requirement',
                'branch_match': 'Branch Qualification',
                'prior_title_prob': 'Historical Precedent for Role',
                'years_service': 'Experience Level',
                'days_in_current_rank': 'Available for Reassignment',
                'title_similarity': 'Role Relevance',
                'training_explicit_match': 'Required Training Met',
                'markov_avg_prob': 'Career Path Alignment'
            }
        else:
            FEATURE_LABELS = {
                'rank_match_exact': 'Promotion Ready?',
                'branch_match': 'Branch Fit',
                'prior_title_prob': 'Career Path Match',
                'years_service': 'Seniority',
                'days_in_current_rank': 'Time Since Promotion',
                'title_similarity': 'Title Similarity',
                'training_explicit_match': 'Training Match',
                'markov_avg_prob': 'Sequence Probability'
            }
        
        # Store for use in waterfall chart
        self._current_feature_labels = FEATURE_LABELS
        
        # Get constraints for expectations
        role_const = constraints.get(t_to, {}) if constraints else {}
        expected_ranks = role_const.get('ranks', [])
        expected_branches = role_const.get('branches', [])
        
        ex = {}
        
        # 0. AI Probability Score (NEW)
        # Dynamic Driver Analysis
        if contribs:
            # XAI-Based Description
            # Contribs is approx {feature: value}
            # Identify top 3 positive and top 1 negative
            pos = {k:v for k,v in contribs.items() if v > 0}
            neg = {k:v for k,v in contribs.items() if v < 0}
            
            top_pos = sorted(pos.items(), key=lambda x: x[1], reverse=True)[:3]
            top_neg = sorted(neg.items(), key=lambda x: x[1])[:1]
            
            driver_lines = []
            for k, v in top_pos:
                # Clean feature name
                name = k.replace('_', ' ').title()
                driver_lines.append(f"+ **{name}** (+{v:.2f})")
            
            for k, v in top_neg:
                name = k.replace('_', ' ').title()
                driver_lines.append(f"- **{name}** ({v:.2f})")
                
            driver_text = "\n".join(driver_lines)
            
            desc = (
                f"**Score Drivers (Calculated)**:\n{driver_text}\n\n"
                "This score is mathematically derived from the sum of these feature contributions.\n"
                "See 'Deep Dive Analysis' below for the full calculation."
            )
        else:
            # Heuristic Fallback
            drivers = []
            if feats.get('prior_title_prob', 0) > 0.05:
                drivers.append(f"**Strong History Signal ({feats['prior_title_prob']:.1%})**")
            if feats.get('rank_match_exact', 0):
                drivers.append("**Rank Eligibility**")
            if feats.get('branch_match', 0):
                drivers.append("**Branch Fit**")
            
            driver_text = ", ".join(drivers) if drivers else "**Baseline Structural Fit**"
            
            # Determine Status for text
            rank_status = 'Met' if feats.get('rank_match_exact') else 'Miss'
            branch_status = 'Met' if feats.get('branch_match') else 'Miss'
            
            desc = (
                f"**Score Drivers**: {driver_text}\n\n"
                f"The AI calculates this probability by comparing this candidate's profile against:\n"
                f"1.  **Past Patterns**: How often officers moved from '{t_from}' to '{t_to}'.\n"
                f"2.  **Hard Constraints**: Rank ({rank_status}) and Branch ({branch_status}).\n"
                f"3.  **Training Overlap**: Shared skills with previous incumbents."
            )
        
        ex['AI Score'] = {
            'value': f"{score:.1%}",
            'desc': desc
        }
        
        # 1. Historical Precedent
        # Ensure we use normalized lookup here too for consistency
        norm_from = self._normalize_title(t_from)
        norm_to = self._normalize_title(t_to)
        
        prob = feats.get('prior_title_prob', 0)
        # Look up exact count if possible using normalized keys
        count = len(self.transition_index.get((norm_from, norm_to), []))
        
        suffix = "officer" if count == 1 else "officers"
        
        ex['History Strength'] = {
            'value': f"{prob:.1%}",
            'desc': f"**{prob:.1%} ({count} {suffix})** of those who held the role **'{t_from}'** immediately moved to **'{t_to}'**. This strong historical pattern suggests this is a standard career progression." if prob > 0 else f"No direct historical precedent (N=0) found for moving strictly from **'{t_from}'** to **'{t_to}'**."
        }
        
        # 2. Branch & Pool Fit (MERGED)
        pool_prob = feats.get('prior_pool_prob', 0)
        branch_match = feats.get('branch_match', 0)
        
        # Format expectations
        exp_branch_str = f"Expected Branches: {', '.join(expected_branches)}" if expected_branches else "Expected Branches: Any"
        # Pool constraints not strictly defined in JSON usually, imply 'Any' or context based
        exp_pool_str = "Expected Pool: Open (No strict constraint)"
        
        if branch_match and pool_prob > 0:
            bp_val = "Strong Fit"
            bp_desc = f"**Branch Match**: Candidate is in the correct branch.\n\n{exp_branch_str}\n{exp_pool_str}\n\n**Pool Flow**: {pool_prob:.1%} historical flow from {p_from} to {p_to}."
        elif branch_match:
            bp_val = "Branch Match"
            bp_desc = f"Candidate matches Branch requirement, but Pool transition ({p_from} -> {p_to}) is uncommon ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
        elif pool_prob > 0.1:
            bp_val = "Flow Match"
            bp_desc = f"Cross-Branch candidate, but valid Pool transition flow ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
        else:
            bp_val = "Weak Fit"
            bp_desc = f"Cross-Branch candidate with low historical pool flow ({pool_prob:.1%}).\n\n{exp_branch_str}\n{exp_pool_str}"
            
        ex['Branch & Pool Fit'] = {
            'value': bp_val,
            'desc': bp_desc
        }
        
        # 3. Rank & Entry Fit (MERGED)
        rank_match = feats.get('rank_match_exact', 0)
        entry = ctx.get('Entry_Type', 'Unknown')
        
        # Format expectations
        exp_rank_str = f"Expected Ranks: {', '.join(expected_ranks)}" if expected_ranks else "Expected Ranks: Any"
        exp_entry_str = "Expected Entry: Any (No strict constraint)"
        
        re_val = "Rank Met" if rank_match else "Rank Mismatch"
        re_desc = "Rank requirement is met." if rank_match else "Rank does not match standard constraints."
        re_desc += f"\n\n{exp_rank_str}"
        re_desc += f"\n{exp_entry_str}"
        re_desc += f"\n\nCandidate Entry Protocol: **{entry}**"
        
        ex['Rank & Entry Fit'] = {
            'value': re_val,
            'desc': re_desc
        }
        
        # 4. Training Match (Consensus-Based)
        cand_training_str = ctx.get('Officer_Training', '')
        cand_courses = set()
        if isinstance(cand_training_str, str) and cand_training_str:
            for t in cand_training_str.split(','):
                # Clean dates here too
                clean_t = re.sub(r'\s*\(.*?\)', '', t).strip()
                if clean_t: cand_courses.add(clean_t)
        
        # Get profile for Target Title
        norm_target = self._normalize_title(t_to)
        
        relevant_courses = getattr(self, 'role_training_counts', {}).get(norm_target, {})
        total_incumbents = getattr(self, 'role_incumbent_counts', {}).get(norm_target, 1)
        
        # Find overlaps
        hits = []
        for course, count in relevant_courses.items():
            if course in cand_courses:
                 # Calculate popularity among incumbents
                 # "What was the most popular training?"
                prevalence = count
                total = total_incumbents
                hits.append({'course': course, 'n': count, 'pct': count/max(1,total)})
        
        # Sort by popularity (count desc)
        hits.sort(key=lambda x: x['n'], reverse=True)
        
        if hits:
            # Format top matches
            # "Most popular: X (used by 5), Y (used by 3)"
            details = ", ".join([f"{h['course']} (used by {h['n']})" for h in hits[:3]])
            
            ex['Training Match'] = {
                'value': f"Matched ({len(hits)})",
                'desc': f"The officer possesses **{len(hits)}** qualifications commonly held by previous incumbents. Most popular overlaps: **{details}**."
            }
        else:
            # If no match, show what ARE the popular ones they miss?
            # Find top popular courses for role
            popular_all = sorted(relevant_courses.items(), key=lambda x: x[1], reverse=True)[:3]
            if popular_all:
                missing = ", ".join([f"{c} ({n})" for c, n in popular_all])
                ex['Training Match'] = {
                    'value': "None",
                    'desc': f"Officer lacks common qualifications for this role. Previous incumbents typically held: **{missing}**."
                }
            else:
                ex['Training Match'] = {
                    'value': "N/A",
                    'desc': f"No historical training data for role **'{t_to}'**."
                }
            
        return ex

    def _generate_dynamic_tooltip(self, name, shap_val, feat_val):
        """
        Generates a context-aware explanation string based on the feature value and its impact.
        """
        # Formatter helpers
        is_pos = shap_val > 0
        impact = "Boosts Score" if is_pos else "Polishes Score" if abs(shap_val) < 0.1 else "Reduces Score"
        
        # Specific Logic
        if name == 'rank_match_exact':
            status = "Matches" if feat_val == 1 else "Mismatch"
            return f"<b>Rank Alignment: {status}</b><br>Value: {feat_val}<br>Impact: {shap_val:+.2f}<br><br>Because the rank {status.lower()} the requirement, this {impact.lower()}."
            
        elif name == 'branch_match':
            status = "Matches" if feat_val == 1 else "Mismatch"
            return f"<b>Branch Alignment: {status}</b><br>Value: {feat_val}<br>Impact: {shap_val:+.2f}<br><br>Candidate {status.lower()} the target branch."
            
        elif name == 'prior_title_prob':
            if feat_val == 0:
                return f"<b>History Pattern: None</b><br>Value: 0%<br>Impact: {shap_val:+.2f}<br><br>We have never seen a direct transition from the candidate's current role to this target in the dataset. This lack of precedent penalizes the score."
            return f"<b>History Pattern: {feat_val:.1%}</b><br>Impact: {shap_val:+.2f}<br><br>This transition happens {feat_val:.1%} of the time historically, providing a strong signal."
            
        elif name == 'years_service':
            return f"<b>Tenure: {feat_val} Years</b><br>Impact: {shap_val:+.2f}<br><br>Candidate's total service length."
            
        elif name == 'title_similarity':
            return f"<b>Keyword Match: {feat_val:.2f}</b><br>Impact: {shap_val:+.2f}<br><br>Similarity between current title and target title (0-1). Higher overlap suggests functional relevance."
         
        # Default fallback
        return f"<b>{name.replace('_', ' ').title()}</b><br>Value: {feat_val}<br>Impact: {shap_val:+.2f}"

    def create_shap_waterfall(self, contribs, base_value=0.0, feats=None):
        """
        Creates a Plotly Waterfall chart from SHAP contributions.
        Supports Base Value display and rich Tooltips.
        """
        # Dictionary Map for Readable Labels
        # Use context-aware labels if set by format_feature_explanation, else fallback
        FEATURE_MAP = getattr(self, '_current_feature_labels', None) or {
            'rank_diff': 'Rank Proximity',
            'branch_match': 'Branch Match',
            'pool_match': 'Pool Match',
            'years_in_current_rank': 'Time in Rank',
            'days_in_current_rank': 'Time in Current Role',
            'num_prior_roles': 'Career Depth',
            'history_word_overlap': 'Title Match',
            'title_similarity': 'Title Similarity',
            'training_match': 'Training Fit',
            'prior_title_prob': 'History Pattern',
            'is_same_branch': 'Strict Branch',
            'rank_match_exact': 'Rank Eligibility',
            'years_service': 'Total Service'
        }

        # Prepare Data
        # Sort by magnitude descending, take top 8
        sorted_items = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
        top_k = sorted_items[:8]
        
        # Rest grouping
        remainder = sum([v for k,v in sorted_items[8:]])
        
        names = []
        hover_texts = []
        values = []
        
        # 0. Add Base Value (Start)
        names.append("Baseline")
        values.append(base_value)
        hover_texts.append(f"<b>Base Score: {base_value:.2f}</b><br>The starting assumption (average fit) before looking at specific candidate details.")
        
        # Features
        for k, v in top_k:
            # Get Readable Name
            label = FEATURE_MAP.get(k, k.replace('_', ' ').title())
            names.append(label)
            
            # Get Value for Tooltip
            f_val = feats.get(k, 'N/A') if feats else 'N/A'
            
            # Generate Rich Tooltip
            tt = self._generate_dynamic_tooltip(k, v, f_val)
            hover_texts.append(tt)
            
            values.append(v)
            
        # Remainder
        if abs(remainder) > 0.001:
            names.append("Other Features")
            values.append(remainder)
            hover_texts.append("Sum of all other small feature contributions.")
            
        # Calculate Final for Label
        final_score = sum(values)

        # Plot
        fig = go.Figure(go.Waterfall(
            name = "Prediction",
            orientation = "v",
            measure = ["absolute"] + ["relative"] * (len(values)-1) + ["total"],
            x = names + ["Raw Score"],
            textposition = "outside",
            # Fix: Show numeric total instead of 'Total' text
            text = [f"{v:+.2f}" for v in values] + [f"{final_score:.2f}"],
            y = values + [0], # Dummy for total
            hovertext = hover_texts + [f"<b>Final Raw Link Score: {final_score:.2f}</b>"],
            hoverinfo = "text+name", 
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
             title = dict(text=f"Decision Path (Final Raw Score: {final_score:.2f})", font=dict(size=14)),
             showlegend = False,
             height=450,
             margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig

    def create_global_beeswarm_plot(self, shap_explanation):
        """
        Renders a SHAP Beeswarm plot using Matplotlib.
        Returns the Figure object.
        """
        import shap
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot beeswarm
        # max_display=12 to show top features
        shap.plots.beeswarm(shap_explanation, max_display=12, show=False)
        
        # Adjust layout
        plt.tight_layout()
        
        # Return figure
        return fig

    def create_global_bar_plot(self, shap_explanation):
        """
        Renders a SHAP Bar plot (Global Feature Importance).
        """
        import shap
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        shap.plots.bar(shap_explanation, max_display=12, show=False)
        plt.tight_layout()
        return fig

    def create_force_plot_html(self, base_value, shap_values, features):
        """
        Generates the HTML for a SHAP Force Plot (JavaScript).
        Use st.components.v1.html to render.
        Note: Force plot expects single prediction data.
        """
        import shap
        
        # shap.force_plot returns HTML string if matplotlib=False (default)
        # We need to call shap.initjs() in the app, or inject it.
        # However, force_plot return is complex (Visualizer object).
        # We use .html() method.
        
        try:
             plot = shap.force_plot(
                 base_value, 
                 shap_values, 
                 features, 
                 matplotlib=False
             )
             return f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        except Exception as e:
             return f"<div>Error generating force plot: {e}</div>"

