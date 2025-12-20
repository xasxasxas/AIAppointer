from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import streamlit as st

class SemanticAIEngine:
    def __init__(self, df, predictor, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the Semantic AI Engine.
        Args:
            df: The main dataframe containing officer data.
            predictor: The trained Predictor object (used for constraints/roles).
            model_name: HuggingFace model name for embeddings.
        """
        self.df = df
        self.predictor = predictor
        self.model_name = model_name
        self.model = None
        self.embeddings = {}
        
        # Lazy load model
        self.load_model()
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            with st.spinner(f"Loading Semantic Model ({self.model_name})..."):
                self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            st.error(f"Failed to load semantic model: {e}")
            self.model = None

    def search_officers(self, query, top_k=10):
        """
        Search for officers based on a natural language query.
        """
        if not self.model:
            return pd.DataFrame()
            
        # 1. Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # 2. Prepare corpus (if not already cached)
        if 'officer_history' not in self.embeddings:
            # Combine history and training for rich context
            corpus = self.df.apply(lambda x: f"{x.get('Appointment_history', '')} {x.get('Training_history', '')} {x.get('current_appointment', '')}", axis=1).tolist()
            self.embeddings['officer_history'] = self.model.encode(corpus, convert_to_tensor=True)
            
        # 3. Search
        hits = util.semantic_search(query_embedding, self.embeddings['officer_history'], top_k=top_k)
        
        # 4. Format results
        results = []
        hit_list = hits[0] # Get first query results
        for hit in hit_list:
            idx = hit['corpus_id']
            score = hit['score']
            row = self.df.iloc[idx].copy()
            row['semantic_score'] = score
            results.append(row)
            
        return pd.DataFrame(results)

    def search_billets(self, query, top_k=10):
        """
        Search for billets (roles) based on a description.
        """
        if not self.model or not self.predictor:
            return []
            
        # Get all valid roles
        roles = list(self.predictor.constraints.keys())
        
        # Encode roles
        if 'roles' not in self.embeddings:
            self.embeddings['roles'] = self.model.encode(roles, convert_to_tensor=True)
            
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings['roles'], top_k=top_k)
        
        results = []
        for hit in hits[0]:
            idx = hit['corpus_id']
            results.append({
                'role': roles[idx],
                'score': hit['score']
            })
            
        return results

    def find_similar_officers(self, officer_id, top_k=5):
        """
        Find officers semantically similar to a given officer ID.
        """
        # Find the source officer index
        try:
            idx = self.df[self.df['Employee_ID'] == officer_id].index[0]
        except IndexError:
            return pd.DataFrame()
            
        if 'officer_history' not in self.embeddings:
            # Trigger build
            self.search_officers("dummy") 
            
        # Get embedding for this officer
        source_embedding = self.embeddings['officer_history'][idx]
        
        # Search
        hits = util.semantic_search(source_embedding, self.embeddings['officer_history'], top_k=top_k+1)
        
        results = []
        for hit in hits[0]:
            res_idx = hit['corpus_id']
            if res_idx == idx: continue # Skip self
            
            row = self.df.iloc[res_idx].copy()
            row['similarity_score'] = hit['score']
            results.append(row)
            
        return pd.DataFrame(results)
