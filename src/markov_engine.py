"""
Markov Sequence Engine for Career Path Prediction

This module implements higher-order Markov chains to capture sequential patterns
in career progressions. It extends the existing first-order transition probabilities
to 2nd and 3rd-order Markov models.

Key Features:
- 2nd-order Markov: P(next_role | current_role, previous_role)
- 3rd-order Markov: P(next_role | current_role, prev_role, prev_prev_role)
- Automatic fallback to lower orders when data is sparse
- Smoothing for unseen transitions
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import joblib
import os
from typing import List, Dict, Tuple, Optional


class MarkovSequenceEngine:
    """
    Enhanced Markov Chain engine for sequential career path modeling.
    
    Builds and uses higher-order Markov transition matrices to predict
    next career moves based on historical sequences.
    """
    
    def __init__(self, max_order=3, smoothing_alpha=0.01):
        """
        Initialize Markov engine.
        
        Args:
            max_order: Maximum Markov order to compute (default: 3)
            smoothing_alpha: Laplace smoothing parameter for unseen transitions
        """
        self.max_order = max_order
        self.smoothing_alpha = smoothing_alpha
        self.transition_matrices = {}  # {order: {context: {next_role: prob}}}
        self.role_counts = Counter()  # For smoothing
        self.all_roles = set()
        
    def fit(self, career_sequences: List[List[str]]):
        """
        Build transition matrices from historical career sequences.
        
        Args:
            career_sequences: List of career sequences, where each sequence is
                            a list of role titles in chronological order
                            Example: [['Role A', 'Role B', 'Role C'], ...]
        """
        print(f"Building Markov transition matrices (order 1-{self.max_order})...")
        
        # Collect all roles for smoothing
        for seq in career_sequences:
            self.all_roles.update(seq)
            self.role_counts.update(seq)
        
        print(f"Found {len(self.all_roles)} unique roles across {len(career_sequences)} sequences")
        
        # Build matrices for each order
        for order in range(1, self.max_order + 1):
            print(f"  Building {order}-order transitions...")
            self.transition_matrices[order] = self._build_transition_matrix(
                career_sequences, order
            )
            
        # Print statistics
        for order in range(1, self.max_order + 1):
            n_contexts = len(self.transition_matrices[order])
            print(f"  Order {order}: {n_contexts} unique contexts")
    
    def _build_transition_matrix(
        self, 
        sequences: List[List[str]], 
        order: int
    ) -> Dict[Tuple[str, ...], Dict[str, float]]:
        """
        Build transition matrix for a specific Markov order.
        
        Args:
            sequences: List of career sequences
            order: Markov order (1, 2, or 3)
            
        Returns:
            Dictionary mapping context tuples to next-role probability distributions
        """
        # Count transitions
        transition_counts = defaultdict(Counter)
        
        for seq in sequences:
            # Need at least (order + 1) roles to create a transition
            if len(seq) < order + 1:
                continue
                
            # Slide window over sequence
            for i in range(len(seq) - order):
                # Context: previous 'order' roles
                context = tuple(seq[i:i+order])
                # Next role
                next_role = seq[i+order]
                
                transition_counts[context][next_role] += 1
        
        # Convert counts to probabilities with smoothing
        transition_probs = {}
        
        for context, next_role_counts in transition_counts.items():
            total_count = sum(next_role_counts.values())
            
            # Laplace smoothing
            vocab_size = len(self.all_roles)
            probs = {}
            
            for role in self.all_roles:
                count = next_role_counts.get(role, 0)
                # Smoothed probability
                probs[role] = (count + self.smoothing_alpha) / (
                    total_count + self.smoothing_alpha * vocab_size
                )
            
            transition_probs[context] = probs
        
        return transition_probs
    
    def predict_proba(
        self, 
        career_history: List[str], 
        candidate_roles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Predict probability distribution over next roles given career history.
        
        Uses highest-order Markov chain possible based on history length,
        with automatic fallback to lower orders.
        
        Args:
            career_history: List of past roles in chronological order
                          Example: ['Role A', 'Role B', 'Role C']
            candidate_roles: Optional list of specific roles to score.
                           If None, scores all known roles.
        
        Returns:
            Dictionary mapping role names to probabilities
        """
        if not career_history:
            # No history - return uniform distribution
            return self._uniform_distribution(candidate_roles)
        
        # Try highest order possible given history length
        history_len = len(career_history)
        
        for order in range(min(self.max_order, history_len), 0, -1):
            # Extract context (last 'order' roles)
            context = tuple(career_history[-order:])
            
            # Check if we have this context
            if context in self.transition_matrices[order]:
                probs = self.transition_matrices[order][context]
                
                # Filter to candidate roles if specified
                if candidate_roles:
                    return {role: probs.get(role, self.smoothing_alpha / len(self.all_roles)) 
                           for role in candidate_roles}
                else:
                    return probs
        
        # Fallback: No matching context found - return uniform
        return self._uniform_distribution(candidate_roles)
    
    def _uniform_distribution(self, candidate_roles: Optional[List[str]] = None) -> Dict[str, float]:
        """Return uniform probability distribution."""
        roles = candidate_roles if candidate_roles else list(self.all_roles)
        if not roles:
            return {}
        prob = 1.0 / len(roles)
        return {role: prob for role in roles}
    
    def get_top_k_predictions(
        self, 
        career_history: List[str], 
        k: int = 5,
        candidate_roles: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-K most likely next roles.
        
        Args:
            career_history: List of past roles
            k: Number of top predictions to return
            candidate_roles: Optional list of roles to consider
            
        Returns:
            List of (role, probability) tuples, sorted by probability descending
        """
        probs = self.predict_proba(career_history, candidate_roles)
        
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_probs[:k]
    
    def get_transition_info(self, career_history: List[str]) -> Dict:
        """
        Get diagnostic information about which Markov order was used.
        
        Useful for debugging and understanding model behavior.
        """
        if not career_history:
            return {
                'order_used': 0,
                'context': None,
                'context_seen': False,
                'fallback_reason': 'No history'
            }
        
        history_len = len(career_history)
        
        for order in range(min(self.max_order, history_len), 0, -1):
            context = tuple(career_history[-order:])
            
            if context in self.transition_matrices[order]:
                return {
                    'order_used': order,
                    'context': context,
                    'context_seen': True,
                    'n_transitions': len(self.transition_matrices[order][context])
                }
        
        return {
            'order_used': 0,
            'context': tuple(career_history[-min(self.max_order, history_len):]),
            'context_seen': False,
            'fallback_reason': 'Context not in training data'
        }
    
    def save(self, filepath: str):
        """Save Markov engine to disk."""
        data = {
            'max_order': self.max_order,
            'smoothing_alpha': self.smoothing_alpha,
            'transition_matrices': self.transition_matrices,
            'role_counts': dict(self.role_counts),
            'all_roles': list(self.all_roles)
        }
        joblib.dump(data, filepath)
        print(f"Markov engine saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Markov engine from disk."""
        data = joblib.load(filepath)
        self.max_order = data['max_order']
        self.smoothing_alpha = data['smoothing_alpha']
        self.transition_matrices = data['transition_matrices']
        self.role_counts = Counter(data['role_counts'])
        self.all_roles = set(data['all_roles'])
        print(f"Markov engine loaded from {filepath}")
        print(f"  Max order: {self.max_order}")
        print(f"  Total roles: {len(self.all_roles)}")


def extract_career_sequences(df: pd.DataFrame) -> List[List[str]]:
    """
    Extract career sequences from the processed DataFrame.
    
    Args:
        df: DataFrame with 'Appointment_history' column containing career sequences
        
    Returns:
        List of career sequences (each sequence is a list of role titles)
    """
    from data_processor import DataProcessor
    
    sequences = []
    dp = DataProcessor()
    
    for idx, row in df.iterrows():
        # Parse appointment history using the correct method
        appointments = dp.parse_history_column(row['Appointment_history'])
        
        if appointments:
            # Extract role titles in chronological order
            sequence = [appt['title'] for appt in appointments if appt.get('title')]
            if len(sequence) >= 2:  # Need at least 2 roles for a transition
                sequences.append(sequence)
    
    return sequences



if __name__ == "__main__":
    """
    Test the Markov engine with sample data.
    """
    # Sample career sequences for testing
    test_sequences = [
        ['Div Officer SS Nova', 'Instructor - Warp Technology Institute', 'Head of Deptt USS Phoenix'],
        ['Div Officer CS Atlantis', 'Instructor - Fleet Technical Training Center', 'Head of Deptt USS Phoenix'],
        ['Div Officer SS Nova', 'Instructor - Advanced Systems College', 'Head of Deptt ISS Atlantis'],
        ['Asst Manager Fleet Maintenance Wing Alpha', 'Dy Manager Fleet Maintenance Wing Alpha', 'Manager Fleet Maintenance Wing Alpha'],
        ['Div Officer RS Horizon', 'Instructor - Starfleet Science Academy', 'Head of Deptt CS Gateway'],
    ]
    
    # Initialize and fit
    engine = MarkovSequenceEngine(max_order=3)
    engine.fit(test_sequences)
    
    # Test predictions
    print("\n" + "="*60)
    print("Testing Predictions")
    print("="*60)
    
    test_history = ['Div Officer SS Nova', 'Instructor - Warp Technology Institute']
    print(f"\nCareer History: {' → '.join(test_history)}")
    
    # Get predictions
    top_5 = engine.get_top_k_predictions(test_history, k=5)
    print("\nTop 5 Predictions:")
    for i, (role, prob) in enumerate(top_5, 1):
        print(f"  {i}. {role}: {prob:.4f}")
    
    # Get diagnostic info
    info = engine.get_transition_info(test_history)
    print(f"\nDiagnostic Info:")
    print(f"  Order used: {info['order_used']}")
    print(f"  Context: {info.get('context')}")
    print(f"  Context seen in training: {info['context_seen']}")
    
    # Test save/load
    print("\n" + "="*60)
    print("Testing Save/Load")
    print("="*60)
    engine.save('test_markov_engine.pkl')
    
    engine2 = MarkovSequenceEngine()
    engine2.load('test_markov_engine.pkl')
    
    # Verify loaded engine works
    top_5_loaded = engine2.get_top_k_predictions(test_history, k=5)
    print("\nPredictions from loaded engine match:", top_5 == top_5_loaded)
