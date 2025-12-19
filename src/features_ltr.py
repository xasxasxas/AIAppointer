
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class LTRFeatureEngineer:
    def __init__(self):
        self.tfidf = None
        self.stop_words = 'english'
        
    def fit(self, all_role_titles):
        """
        Fit TF-IDF on all known role titles to understand the vocabulary.
        """
        print("Fitting TF-IDF on role vocabulary...")
        self.tfidf = TfidfVectorizer(stop_words=self.stop_words, ngram_range=(1, 2))
        self.tfidf.fit(all_role_titles.unique().astype(str))
        
    def compute_similarity(self, officer_history_str, target_role_title):
        """
        Compute similarity between officer's past roles and variable target role.
        """
        if not officer_history_str or not target_role_title:
            return 0.0
            
        try:
            # Vectorize
            vecs = self.tfidf.transform([officer_history_str, target_role_title])
            # Cosine Sim
            sim = cosine_similarity(vecs[0], vecs[1])[0][0]
            return sim
        except:
            return 0.0
            
    def generate_pair_features(self, officer_row, candidate_row, transition_stats=None, markov_engine=None):
        """
        Generate a feature dictionary for a single (Officer, Candidate) pair.
        
        Args:
            officer_row: Dict/Series of officer state
            candidate_row: Dict/Series of candidate role attributes
            constraints_map: Optional dict of role requirements
        """
        feats = {}
        
        # 1. Basic Identifiers (for debugging)
        feats['officer_rank'] = officer_row.get('Rank', 'Unknown')
        feats['target_role'] = candidate_row.get('Role', 'Unknown')
        
        # 2. MATCH FEATURES (The "AI" logic)
        
        # A. Rank Compatibility
        # Does the officer have the rank required by the role?
        # We need the Role's "Required Rank" distribution.
        # If candidate_row comes from the constraint map or stats
        role_req_ranks = candidate_row.get('REQ_Ranks', [])
        curr_rank = officer_row.get('Rank')
        
        feats['rank_match_exact'] = 1 if curr_rank in role_req_ranks else 0
        
        # B. Branch Match
        off_branch = officer_row.get('Branch', 'Unknown')
        role_branch = candidate_row.get('Branch', 'Unknown') # Target Branch
        feats['branch_match'] = 1 if off_branch == role_branch else 0
        
        # C. Text Similarity (Context)
        # Compare Officer History vs Target Role
        # 'history_str' should be pre-computed in officer_row
        hist_str = officer_row.get('history_str', '')
        tgt_title = str(candidate_row.get('Role', ''))
        
        if self.tfidf:
            feats['title_similarity'] = self.compute_similarity(hist_str, tgt_title)
        else:
            feats['title_similarity'] = 0.0
            
        # D. Training Overlap (Implicit Prereqs)
        # If the role title contains "Instructor", does officer have "Education" training?
        role_lower = tgt_title.lower()
        training_str = str(officer_row.get('Training_history', '')).lower()
        
        has_req_training = 0
        if "instructor" in role_lower and "education" in training_str: has_req_training = 1
        if "command" in role_lower and "command" in training_str: has_req_training = 1
        if "intelligence" in role_lower and "intelligence" in training_str: has_req_training = 1
         # ... general overlap
        feats['training_explicit_match'] = has_req_training
        
        # 3. Officer Context
        # Use numericals directly
        feats['years_service'] = float(officer_row.get('years_service', 0))
        feats['days_in_current_rank'] = float(officer_row.get('years_in_current_rank', 0)) * 365
        
        # 4. Role Frequency (Popularity)
        feats['role_popularity'] = float(candidate_row.get('freq', 1))
        
        # 5. Transition Priors (Soft Signals)
        # transition_stats = {'pool_trans': {curr_pool: {tgt_pool: prob}}, 
        #                     'title_trans': {curr_title: {tgt_title: prob}}}
        
        feats['prior_pool_prob'] = 0.0
        feats['prior_title_prob'] = 0.0
        
        if transition_stats:
            # A. Pool Transition
            curr_pool = officer_row.get('Pool', 'Unknown')
            # Candidate row might be from metadata, need to ensure it has Pool. 
            # If not, try to look it up or default.
            tgt_pool = candidate_row.get('Pool', 'Unknown')
            
            if curr_pool != 'Unknown' and tgt_pool != 'Unknown':
                 pool_matrix = transition_stats.get('pool_trans', {})
                 if curr_pool in pool_matrix:
                     feats['prior_pool_prob'] = pool_matrix[curr_pool].get(tgt_pool, 0.0)
                     
            # B. Title Transition
            # Officer row "last_role_title" usually holds the role BEFORE the target.
            # In 'predict' context, 'current_appointment' is the last role.
            # Let's standardize on "last_role_title" being present or falling back to "current_appointment"
            curr_title = officer_row.get('last_role_title')
            if not curr_title or curr_title == 'Unknown':
                curr_title = officer_row.get('current_appointment', 'Unknown')
                
            tgt_title = candidate_row.get('Role', 'Unknown')
            
            if curr_title != 'Unknown':
                title_matrix = transition_stats.get('title_trans', {})
                if curr_title in title_matrix:
                    feats['prior_title_prob'] = title_matrix[curr_title].get(tgt_title, 0.0)
        
        # 6. Markov Sequential Features (NEW)
        feats['markov_2nd_order_prob'] = 0.0
        feats['markov_3rd_order_prob'] = 0.0
        feats['markov_avg_prob'] = 0.0
        
        if markov_engine:
            # Extract career history from officer data
            career_history = self._extract_career_history(officer_row)
            
            if len(career_history) >= 1:
                # Get Markov probabilities for the target role
                markov_probs = markov_engine.predict_proba(
                    career_history, 
                    candidate_roles=[tgt_title]
                )
                
                # Get diagnostic info to determine which order was used
                info = markov_engine.get_transition_info(career_history)
                order_used = info['order_used']
                
                # Assign probabilities based on order used
                if order_used >= 2:
                    feats['markov_2nd_order_prob'] = markov_probs.get(tgt_title, 0.0)
                
                if order_used >= 3:
                    feats['markov_3rd_order_prob'] = markov_probs.get(tgt_title, 0.0)
                
                # Average probability (always available)
                feats['markov_avg_prob'] = markov_probs.get(tgt_title, 0.0)
        
        return feats
    
    def _extract_career_history(self, officer_row):
        """
        Extract ordered list of past roles from officer data.
        
        Returns:
            List of role titles in chronological order
        """
        # Try to get from snapshot_history first (most complete)
        if 'snapshot_history' in officer_row and officer_row['snapshot_history']:
            history = [h.get('title', '') for h in officer_row['snapshot_history'] if h.get('title')]
            if history:
                return history
        
        # Fallback: construct from last_role_title and current_appointment
        history = []
        if 'last_role_title' in officer_row and officer_row['last_role_title']:
            last_role = officer_row['last_role_title']
            if last_role and last_role != 'Unknown':
                history.append(last_role)
        
        if 'current_appointment' in officer_row:
            curr_role = officer_row['current_appointment']
            if curr_role and curr_role not in history:
                history.append(curr_role)
        
        return history

    def save(self, path):
        joblib.dump(self.tfidf, path)
        
    def load(self, path):
        self.tfidf = joblib.load(path)

    def transform(self, pairs_df, transition_stats=None, markov_engine=None):
        """
        Batch transform a DataFrame of items into features.
        Used for Global SHAP context generation.
        Assumes each row contains both Officer attributes and Target Role attributes (merged).
        """
        features_list = []
        for idx, row in pairs_df.iterrows():
            # In Global Context loading, 'row' is a merged dict of User + Target Role Metadata
            officer_row = row.to_dict()
            candidate_row = row.to_dict() # Same row carries the target info
            
            feats = self.generate_pair_features(officer_row, candidate_row, 
                                               transition_stats=transition_stats,
                                               markov_engine=markov_engine)
            features_list.append(feats)
            
        return pd.DataFrame(features_list)
