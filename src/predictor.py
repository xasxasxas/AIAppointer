
import pandas as pd
import numpy as np
import joblib
import os
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.features_ltr import LTRFeatureEngineer
from src.markov_engine import MarkovSequenceEngine
from src.xai import ModelExplainer  # Import XAI

# RANK ORDER for Heuristics
RANK_ORDER = [
    'Ensign', 'Lieutenant (jg)', 'Lieutenant', 'Lieutenant Commander', 
    'Commander', 'Captain', 'Rear Admiral (Lower Half)', 
    'Rear Admiral', 'Vice Admiral', 'Admiral'
]

class Predictor:
    def __init__(self, models_dir='models/ltr'):
        print(f"Loading LTR models from {models_dir}...")
        
        # Paths
        if not os.path.exists(models_dir):
            models_dir = os.path.join(os.getcwd(), 'models', 'ltr')
            
        try:
            self.model = joblib.load(os.path.join(models_dir, 'lgbm_ranker.pkl'))
            self.role_meta = joblib.load(os.path.join(models_dir, 'role_meta.pkl'))
            self.feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
            
            # Feature Engineer
            self.ltr_fe = LTRFeatureEngineer()
            self.ltr_fe.load(os.path.join(models_dir, 'ltr_fe.pkl'))
            
            # Standard FE for Heuristics
            self.std_fe = FeatureEngineer()

            # Initialize Explainer (Heavy)
            try:
                self.xai = ModelExplainer(self.model)
                print("✓ SHAP Explainer Initialized.")
            except Exception as e:
                print(f"⚠️ SHAP Init Failed: {e}")
                self.xai = None
            
            # Load Transition Stats
            try:
                self.transition_stats = joblib.load(os.path.join(models_dir, 'transition_stats.pkl'))
                print("✓ Transition Stats Loaded.")
            except:
                print("⚠️ Transition Stats NOT Found. LTR will run without priors.")
                self.transition_stats = None
            
            # Load Markov Engine
            try:
                self.markov_engine = MarkovSequenceEngine()
                self.markov_engine.load(os.path.join(models_dir, 'markov_stats.pkl'))
                print("✓ Markov Engine Loaded.")
            except:
                print("⚠️ Markov Engine NOT Found. LTR will run without Markov features.")
                self.markov_engine = None
            
            # Load Strict Constraints (Ground Truth)
            import json
            with open('models/all_constraints.json', 'r') as f:
                self.strict_constraints = json.load(f)
                
            # STRICT WHITELIST: Only these roles exist.
            self.valid_roles = set(self.strict_constraints.keys())
            
            # UI COMPATIBILITY
            # The UI expects 'target_encoder.classes_' to list all roles for Dropdowns
            # and 'constraints' dict.
            self.constraints = self.strict_constraints
            self.target_encoder = type('E', (), {'classes_': sorted(list(self.valid_roles))})
                
            self.ready = True
            print("✓ LTR System Loaded.")
        except Exception as e:
            print(f"Error loading LTR system: {e}")
            self.ready = False

    def get_global_context(self, n=500, branch_filter=None, pool_filter=None, entry_filter=None):
        """
        Loads a random sample of the dataset and generates features.
        Used for Global SHAP plots (Beeswarm).
        Cached to avoid reloading.
        """
        # Cache Init
        if not hasattr(self, '_xai_cache'):
            self._xai_cache = {}
            
        cache_key = (n, branch_filter, pool_filter, entry_filter)
        if cache_key in self._xai_cache:
            return self._xai_cache[cache_key]
            
        print(f"Loading global context for XAI (Filter={branch_filter}, {pool_filter}, {entry_filter})...")
        try:
            # 1. Load Raw Data
            df = pd.read_csv('data/hr_star_trek_v4c_modernized_clean_modified_v4.csv')
            
            # Filters
            if branch_filter and branch_filter != "All":
                df = df[df['Branch'] == branch_filter]
            
            if pool_filter and pool_filter != "All":
                df = df[df['Pool'] == pool_filter]
                
            if entry_filter and entry_filter != "All":
                df = df[df['Entry_type'] == entry_filter]
                
            if len(df) < 5:
                print("Warning: Filters resulted in too few samples.")
                return None
            
            # 2. Sample
            if len(df) > n:
                df = df.sample(n, random_state=42)
                
            # 3. Process Features (DataProcessor -> Standard + LTR)
            # Must parse history strings first
            dp = DataProcessor()
            df = dp.get_current_features(df)
            df = self.std_fe.extract_features(df)
            
            # Prepare pairs
            pairs = []
            for _, row in df.iterrows():
                # Target: Current Appointment
                target = row.get('current_appointment', 'Unknown')
                
                row_dict = row.to_dict()
                row_dict['Role'] = target 
                
                # Enrich with Role Meta (Safely, preserving Officer attributes)
                if self.role_meta and target in self.role_meta:
                     meta = self.role_meta[target]
                     for k, v in meta.items():
                         if k not in row_dict:
                             row_dict[k] = v

                pairs.append(row_dict)
            
            # Transform
            if not pairs:
                 return None
                 
            X_df = self.ltr_fe.transform(pd.DataFrame(pairs), 
                                         transition_stats=self.transition_stats,
                                         markov_engine=self.markov_engine)
            
            # Ensure columns match model
            # Reindex to match self.feature_cols
            X_final = X_df.reindex(columns=self.feature_cols, fill_value=0)
            
            self._xai_cache[cache_key] = X_final
            return X_final
            
        except Exception as e:
            print(f"Global Context Load Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _prepare_input_context(self, input_data):
        """Prepare Officer Context Dictionary"""
        if isinstance(input_data, pd.DataFrame):
            row = input_data.iloc[0].to_dict()
        else:
            row = input_data
            
        # Normalize Rank
        if 'Rank' in row and isinstance(row['Rank'], str):
             row['Rank'] = row['Rank'].strip()  # Don't lower(), our keys are Case Sensitive "Lieutenant (jg)"

        # CRITICAL FIX: Clean History String
        # Input might be raw "Role (Date - Date)" strings which fail text matching
        # We need to extract just the titles.
        
        hist_raw = row.get('Appointment_history', '')
        if isinstance(hist_raw, str):
            # Regex to clean "Title (Date...)" -> "Title"
            # Split by comma or newline if multiple?
            # Usually dataset is "Role A (Date), Role B (Date)"
            import re
            # Split by comma, then clean each
            items = str(hist_raw).split(',')
            titles = []
            for item in items:
                # Remove dates (...)
                clean = re.sub(r'\s*\(.*?\)', '', item).strip()
                if clean: titles.append(clean)
            
            row['history_str'] = " > ".join(titles)
            # Update last_role_title for Priors
            if titles:
                row['last_role_title'] = titles[0] # Most recent usually first or last? 
                # Check dataset convention. Usually index 0 is latest? 
                # Let's assume text order implies sequence.
                # Actually, dataset 'Appointment_history' format varies.
                # Safer: Use 'current_appointment' if available
        
        if 'current_appointment' in row:
             row['last_role_title'] = str(row['current_appointment']).strip()
             
        # FUZZY MATCHING for Last Role Title
        # To ensure we hit the Transition Stats keys
        if 'last_role_title' in row and self.transition_stats:
            known_titles = list(self.transition_stats['title_trans'].keys())
            curr = row['last_role_title']
            
            # 1. Exact Match
            if curr in self.transition_stats['title_trans']:
                pass
            else:
                # 2. Substring Match (Case Insensitive)
                # If 'Ensign' is a key, and title is 'Ensign Role', matched!
                found_sub = None
                curr_lower = curr.lower()
                for key in known_titles:
                     if key.lower() in curr_lower or curr_lower in key.lower():
                         found_sub = key
                         break
                
                if found_sub:
                    row['last_role_title'] = found_sub
                    # print(f"Confident Fix: Substring Mapped '{curr}' -> '{found_sub}'")
                else:
                    # 3. Fuzzy Match (Lower Threshold)
                    import difflib
                    matches = difflib.get_close_matches(curr, known_titles, n=1, cutoff=0.4)
                    if matches:
                        # print(f"Confidence Fix: Fuzzy Mapped '{curr}' -> '{matches[0]}'")
                        row['last_role_title'] = matches[0]
                    else:
                        # 4. NEW: TF-IDF Similarity Fallback
                        # Uses existing TF-IDF vectorizer to find semantically similar title
                        if hasattr(self, 'ltr_fe') and self.ltr_fe.tfidf:
                            try:
                                from sklearn.metrics.pairwise import cosine_similarity
                                curr_vec = self.ltr_fe.tfidf.transform([curr])
                                all_vecs = self.ltr_fe.tfidf.transform(known_titles)
                                sims = cosine_similarity(curr_vec, all_vecs)[0]
                                best_idx = np.argmax(sims)
                                if sims[best_idx] > 0.3:  # Threshold for semantic match
                                    row['last_role_title'] = known_titles[best_idx]
                                    # print(f"TF-IDF Matched '{curr}' -> '{known_titles[best_idx]}' (sim={sims[best_idx]:.2f})")
                            except Exception as e:
                                pass  # Fallback to original if TF-IDF fails

        # Enrich if needed (snapshot_history is from simulation)
        if 'snapshot_history' in row:
             hist = row['snapshot_history']
             if isinstance(hist, list):
                 row['history_str'] = " > ".join([h.get('title', '') for h in hist])
                 if hist: row['last_role_title'] = hist[-1].get('title', '')
                 
        return row

    def predict(self, input_data, rank_flex_up=0, rank_flex_down=0):
        """
        Learning-to-Rank Prediction
        """
        if not self.ready:
            return pd.DataFrame([{'Prediction': 'System Not Ready', 'Confidence': 0, 'Explanation': 'Model Error'}])
            
        officer = self._prepare_input_context(input_data)
        
        # 0. Enrich Officer Data (Calculate Years in Rank, etc.)
        try:
            # Convert to DF for FeatureEngineer
            temp_df = pd.DataFrame([officer])
            temp_df = self.std_fe.extract_features(temp_df)
            officer = temp_df.iloc[0].to_dict()
        except Exception as e:
            print(f"Warning: Could not enrich officer features: {e}")

        context = officer
        
        # 1. Filter Candidates by Constraints
        candidates_to_score = []
        
        current_rank = officer.get('Rank')
        current_branch = officer.get('Branch')
        
        # Multi-Pass Candidate Selection
        # Pass 1: Strict Constraints
        # Pass 2: Relaxed Constraints (if Strict yields 0) - Ignore Branch/Entry
        
        candidates_to_score = []
        
        for attempt in ['strict', 'relaxed']:
            temp_candidates = []
            if attempt == 'relaxed':
                 print("  > Strict constraints yielded 0 candidates. Retrying with RELAXED constraints.")
            
            for role_name in self.valid_roles:
                # Meta retrieval
                meta = self.role_meta.get(role_name)
                if not meta:
                    cons = self.strict_constraints.get(role_name, {})
                    meta = {
                        'Role': role_name,
                        'Branch': cons.get('branches', ['Unknown'])[0],
                        'Pool': 'Unknown',
                        'REQ_Ranks': cons.get('ranks', []),
                        'freq': 1
                    }
                
                # Validation Logic
                if self.strict_constraints:
                     role_cons = self.strict_constraints.get(role_name, {})
                     
                     # 1. RANK CHECK (Always Enforced, unless we add Pass 3)
                     ranks = role_cons.get('ranks', [])
                     if ranks and current_rank not in ranks:
                         continue
                         
                     if attempt == 'strict':
                        # 2. MATCH CHECK (Branch) - STRICT ENFORCEMENT
                        branches = role_cons.get('branches', [])
                        if branches and current_branch:
                            # Strict: Officer branch must be in allowed branches
                            if current_branch not in branches:
                                continue
                        
                        # 3. ENTRY CHECK
                        entries = role_cons.get('entries', [])
                        current_entry = context.get('Entry_type')
                        if entries and current_entry and current_entry not in entries:
                            continue
                
                # If we got here, it passed
                if role_name in self.role_meta:
                    temp_candidates.append(self.role_meta[role_name])
            
            # End of Role Loop
            if temp_candidates:
                candidates_to_score = temp_candidates
                break # Found valid candidates
                
        # If still empty after relaxed pass, effectively return empty

            
        # If strict filter removes everything, relax
        if not candidates_to_score:
             # Fallback? Or just return empty?
             # User wants strict. Return empty.
             pass

        # Special Handling: Removed Soft Mapping hack as constraints now natively support Lt search logic.

        if not candidates_to_score:
             return pd.DataFrame([{'Prediction': 'No valid roles found for Rank/Branch/Entry', 'Confidence': 0, 'Explanation': 'Strict Constraints blocked all options.'}])
             
        X_rows = []
        meta_list = []
        feats_list = []
        
        for cand in candidates_to_score:
            feats = self.ltr_fe.generate_pair_features(officer, cand, self.transition_stats, self.markov_engine)
            
            # Inject Context for Explainability
            # Calculate simple Entry Match for display (using strict constraints fallback logic)
            # We don't have easy access to role_cons here in the loop without looking up again.
            # But we can pass it if we refactor. For now, let's just pass officer Entry Type.
            
            entry_type = officer.get('Entry_type', 'Unknown')
            # Look up constraint for this role?
            # We are iterating `candidates_to_score` which came from `valid_roles` loop.
            # We don't have the constraints handy here in `predict` loop easily.
            # But LTR feats usually don't use it.
            # Let's just pass the officer's type. Explainer can display "Candidate is [Type]".
            
            feats['_Context'] = {
                'From_Title': officer.get('last_role_title', 'Unknown'),
                'To_Title': cand.get('Role', 'Unknown'),
                'From_Pool': officer.get('Pool', 'Unknown'),
                'To_Pool': cand.get('Pool', 'Unknown'),
                'Officer_Training': officer.get('Training_history', ''),
                'Entry_Type': entry_type
            }
            
            # Ensure correct col order
            vector = [feats.get(c, 0) for c in self.feature_cols]
            X_rows.append(vector)
            meta_list.append(cand)
            feats_list.append(feats)
            
        if not X_rows:
            return pd.DataFrame()
            
        # Batch Predict
        raw_scores = self.model.predict(X_rows)
        
        # --- XAI-DRIVEN SCORING ---
        # User requested using XAI Score to improve quality/explainability.
        # We calculate SHAP values for all candidates and use them to better differentiate.
        if self.xai:
            try:
                # Convert to DF for SHAP Batch
                X_df = pd.DataFrame(X_rows, columns=self.feature_cols)
                
                # Fast Batch SHAP
                shap_matrix = self.xai.explainer.shap_values(X_df)
                
                # Handle Output Shape (List for Classifier, Array for Ranker)
                if isinstance(shap_matrix, list):
                    shap_matrix = shap_matrix[1] # Positive Class
                    
                # Calculate SHAP Sums (Proxy for explained score)
                # Base Value is needed to reconstruction full score
                base_val = self.xai.explainer.expected_value
                if isinstance(base_val, list): base_val = base_val[1]
                
                shap_scores = np.sum(shap_matrix, axis=1) + base_val
                
                # Use SHAP scores as the primary signal
                # They are usually consistent with raw_scores but allow us to introspect if needed.
                raw_scores = shap_scores
                
            except Exception as e:
                print(f"⚠️ Batch XAI Score Failed: {e}")
                # Fallback to raw_scores
        
        # --- SIGMOID CALIBRATION ---
        # Sigmoid converts log-odds to probabilities naturally
        # This provides meaningful scores reflecting true model confidence
        def sigmoid(x):
            # Clip to prevent overflow
            x_clipped = np.clip(x, -20, 20)
            return 1 / (1 + np.exp(-x_clipped))
        
        # Apply sigmoid to raw_scores (log-odds from LightGBM)
        scores = sigmoid(raw_scores)
        
        # Scale to 5%-95% range to avoid extreme 0% or 100%
        scores = 0.05 + 0.9 * scores
        
        
        # 3. Rank and Format
        scored_candidates = []
        for i, score in enumerate(scores):
            scored_candidates.append({
                'role': meta_list[i]['Role'],
                'score': score,
                '_Feats': feats_list[i],
                '_Vector': X_rows[i] # Keep vector for XAI
            })
            
        # Sort desc
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Format Results
        results = []
        for item in scored_candidates[:10]:
            prob = item['score']
            role = item['role']
            
            # XAI Calculation (On Demand for Top K)
            contribs = {}
            base_val = 0.0
            if self.xai:
                try:
                    # Create DF for SHAP
                    row_df = pd.DataFrame([item['_Vector']], columns=self.feature_cols)
                    contribs, base_val = self.xai.get_contributions(row_df)
                except Exception as e:
                    print(f"XAI Error: {e}")
            
            reason = "High compatibility score."
            if prob > 0.8: reason = "Excellent Fit: Rank, Branch, and Skills match."
            elif prob > 0.5: reason = "Good Fit: Likely transition target."
            else: reason = "Moderate Fit."
            
            results.append({
                'Prediction': role,
                'Confidence': prob,
                'Explanation': reason,
                '_Feats': item['_Feats'],
                '_Contribs': contribs, # Attach SHAP values
                '_BaseVal': base_val,  # Attach Base Value
                '_RawScore': item.get('score', 0) # This is normalized score though? No, item['score'] is normalized.
                # We need raw score. In Step 1210 we lost raw score in `scored_candidates`.
                # Wait, 'scored_candidates' list in Step 1197 line 280 uses `scores`.
                # We should store raw_scores in scored_candidates too if we want it.
            })
            
        return pd.DataFrame(results)

    def predict_for_role(self, candidates_df, target_role, rank_flex_up=0, rank_flex_down=0):
        """
        Reverse Lookup: Find best officers for a role.
        """
        if target_role not in self.role_meta:
             # Try to find closest? or Generic
             cand_meta = {'Role': target_role, 'Branch': 'Unknown', 'REQ_Ranks': [], 'freq': 1}
        else:
             cand_meta = self.role_meta[target_role]
             
        results = []
        
        # Iterate officers
        # Optimization: Filter officers by Rank first?
        # Yes, filter candidates_df by role requirements
        
        # valid_officers = candidates_df[candidates_df['Rank'].isin(cand_meta['REQ_Ranks'])]
        # Taking all for now to show scores
        # Iterate officers
        X_rows = []
        feats_list = []
        indices = []
        
        # Load constraints for target role
        target_cons = self.strict_constraints.get(target_role, {})
        allowed_ranks = target_cons.get('ranks', [])
        allowed_branches = target_cons.get('branches', [])
        allowed_entries = target_cons.get('entries', []) # New
        
        for idx, row in candidates_df.iterrows():
            # STRICT FILTERING
            # 1. Rank
            # (Handled by flex sliders? Or strict?)
            current_rank = row['Rank']
            if allowed_ranks:
                # If strict, must be in allowed keys
                # BUT we need to handle flexibility logic here if we really want to support "Promotable" officers
                # Simplified: If rank strictly in allowed, keep.
                # If rank_flex > 0, we need a "rank map" to know if adjacent.
                # Since we don't have rank map easily accessible here, let's skip STRICT rank filter if flexibility is enabling.
                # But if flexibility is 0, we can filter.
                if rank_flex_up == 0 and rank_flex_down == 0:
                     if current_rank not in allowed_ranks:
                         continue
            
            # 2. Entry Type Check
            if allowed_entries and row['Entry_type'] not in allowed_entries:
                continue
                
            # 3. Branch Check (Optional/Strict?)
            # Usually role implies branch, but cross-branch happens.
            # If strict constraints exist, respect them.
            if allowed_branches and row['Branch'] not in allowed_branches:
                continue

            off_dict = row.to_dict()
            try:
                self._prepare_input_context(off_dict) # In-place enrich
            except:
                pass # Use raw if fails
            feats = self.ltr_fe.generate_pair_features(off_dict, cand_meta, self.transition_stats, self.markov_engine)
            
            # Inject Context for Explainability
            feats['_Context'] = {
                'From_Title': off_dict.get('last_role_title', 'Unknown'),
                'To_Title': cand_meta.get('Role', 'Unknown'),
                'From_Pool': off_dict.get('Pool', 'Unknown'),
                'To_Pool': cand_meta.get('Pool', 'Unknown'),
                'Officer_Training': off_dict.get('Training_history', '')
            }
            
            # Store feats for later retrieval
            # We can't put Dict in X_rows.
            # We must map index to feats.
            # Hack: Store in a temp list aligned with X_rows?
            # Or simplified: X_rows is list of lists
            
            vector = [feats.get(c, 0) for c in self.feature_cols]
            
            X_rows.append(vector)
            feats_list.append(feats) # Needs initialization
            indices.append(idx)
            
        if not X_rows: return pd.DataFrame()
        
        # Predict Raw Scores
        raw_scores = self.model.predict(X_rows)
        
        # --- XAI-DRIVEN SCORING (Consistency with Predict) ---
        if self.xai:
            try:
                # Convert to DF for SHAP Batch
                X_df = pd.DataFrame(X_rows, columns=self.feature_cols)
                
                # Fast Batch SHAP
                shap_matrix = self.xai.explainer.shap_values(X_df)
                
                if isinstance(shap_matrix, list):
                    shap_matrix = shap_matrix[1] # Positive Class
                    
                base_val = self.xai.explainer.expected_value
                if isinstance(base_val, list): base_val = base_val[1]
                
                shap_scores = np.sum(shap_matrix, axis=1) + base_val
                
                # Use SHAP scores
                raw_scores = shap_scores
            except Exception as e:
                print(f"⚠️ Batch XAI Score Failed in Billet Lookup: {e}")

        # --- SIGMOID CALIBRATION ---
        # Sigmoid converts log-odds to probabilities naturally
        def sigmoid(x):
            x_clipped = np.clip(x, -20, 20)
            return 1 / (1 + np.exp(-x_clipped))
        
        # Apply sigmoid to raw_scores
        scores = sigmoid(raw_scores)
        
        # Scale to 5%-95% range
        scores = 0.05 + 0.9 * scores
        
        # --- FRESHNESS BOOST (Billet Lookup Specific) ---
        # Officers who have been in current role longer are "ready to move"
        # Apply up to 15% boost based on tenure
        for i in range(len(scores)):
            days_in_role = feats_list[i].get('days_in_current_rank', 0)
            # Boost curve: 0-1 year = 0%, 1-2 years = 5%, 2-3 years = 10%, 3+ years = 15%
            years_in_role = days_in_role / 365.0
            freshness_boost = min(0.15, max(0, (years_in_role - 1) * 0.05))
            scores[i] = min(0.95, scores[i] * (1 + freshness_boost))
        
        out = []
        for i, score in enumerate(scores):
            orig_idx = indices[i]
            row = candidates_df.loc[orig_idx]
            feat_dict = feats_list[i]
            
            # REMOVED THRESHOLD FILTER to ensure we always return candidates.
            # Extract current role from enriched context or raw row
            curr_role = off_dict.get('current_appointment', off_dict.get('last_role_title', 'Unknown'))
            
            out.append({
                'Employee_ID': row['Employee_ID'],
                'Name': f"Officer {row['Employee_ID']}",
                'Rank': row['Rank'],
                'Branch': row['Branch'],
                'CurrentRole': curr_role,
                'Confidence': score,
                'Explanation': f"Match Probability: {score:.1%}",
                '_Feats': feat_dict 
            })
        
        if not out:
             return pd.DataFrame(columns=['Employee_ID', 'Name', 'Rank', 'Branch', 'CurrentRole', 'Confidence', 'Explanation'])
             
        # Convert to DF and Sort
        res_df = pd.DataFrame(out).sort_values('Confidence', ascending=False)
        
        # SHAP Calculation for Top N (Performance Optimization)
        # Only explain top 20 to keep it fast
        top_n_df = res_df.head(20)
        
        if hasattr(self, 'xai') and self.xai:
            # Reconstruct X_rows for these top candidates
            # We need to map back to original feature vectors or reconstruct them
            # Efficient way: We stored '_Feats' in the row. Reconstruct vector from '_Feats'.
            
            X_explain = []
            for _, r in top_n_df.iterrows():
                f_dict = r['_Feats']
                vec = [f_dict.get(c, 0) for c in self.feature_cols]
                X_explain.append(vec)
                
            if X_explain:
                 # Convert list of lists to DataFrame for column names
                 X_explain_df = pd.DataFrame(X_explain, columns=self.feature_cols)
                 
                 # Use global explanation object logic for batch processing
                 expl_obj = self.xai.get_explanation_object(X_explain_df)
                 
                 shap_vals = expl_obj.values
                 base_vals = expl_obj.base_values
                 
                 # Handle base_values being scalar or array
                 if np.isscalar(base_vals):
                     base_vals = [base_vals] * len(X_explain)
                 
                 # Assign back to DF
                 contribs_list = []
                 base_list = []
                 for i in range(len(X_explain)):
                     # Zip feature names with values for this row
                     row_contribs = dict(zip(self.feature_cols, shap_vals[i]))
                     # Filter zeros to save space? Optional.
                     contribs_list.append(row_contribs)
                     base_list.append(base_vals[i])
                     
                 # Add to DF (using index assignment to match properly)
                 res_df.loc[top_n_df.index, '_Contribs'] = contribs_list
                 res_df.loc[top_n_df.index, '_BaseVal'] = base_list

        return res_df
