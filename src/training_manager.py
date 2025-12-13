import os
import shutil
import pandas as pd
import glob
from datetime import datetime
import joblib

# Import pipeline components
# We import them inside methods or here if no circular deps
from src.build_ltr_dataset import build_dataset
from src.train_ltr import train_ltr
from src.data_processor import DataProcessor

class TrainingManager:
    def __init__(self, base_dir='.'):
        self.base_dir = base_dir
        self.staging_dir = os.path.join(base_dir, 'models', 'staging')
        self.prod_dir = os.path.join(base_dir, 'models', 'ltr')
        self.upload_dir = os.path.join(base_dir, 'data_uploads')
        self.backup_dir = os.path.join(base_dir, 'models', 'backups')
        
        # Ensure dirs exist
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def save_uploaded_dataset(self, file_obj):
        """Saves uploaded Streamlit file object to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.csv"
        path = os.path.join(self.upload_dir, filename)
        
        # Write bytes
        with open(path, "wb") as f:
            f.write(file_obj.getbuffer())
            
        return path
        
    def validate_dataset(self, csv_path):
        """Checks if dataset has minimum required columns."""
        required = ['Employee_ID', 'Rank', 'Branch', 'Pool', 'current_appointment', 'Appointment_history']
        try:
            df = pd.read_csv(csv_path)
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                return False, f"Missing columns: {missing}"
            
            if len(df) < 50:
                return False, f"Dataset too small ({len(df)} rows). Need at least 50."
                
            return True, f"Valid dataset ({len(df)} rows)."
        except Exception as e:
            return False, f"Error reading CSV: {e}"
            
    def train_staging_model(self, csv_path):
        """Runs the full pipeline in a staging isolation."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(self.staging_dir, session_id)
        os.makedirs(session_path, exist_ok=True)
        
        try:
            # 1. Build Dataset (Generates pairs + artifacts)
            print(f"Manager: Building dataset from {csv_path} to {session_path}")
            build_dataset(csv_path, session_path)
            
            # 2. Train Model (Uses pairs in session_path, saves model to session_path)
            pairs_path = os.path.join(session_path, 'train_pairs.csv')
            print(f"Manager: Training model on {pairs_path}")
            train_ltr(pairs_path, session_path)
            
            # 3. Verify Artifacts
            # We expect: lgbm_ranker.pkl, role_meta.pkl, transition_stats.pkl, feature_cols.pkl, ltr_fe.pkl, metrics.json
            expected = ['lgbm_ranker.pkl', 'role_meta.pkl', 'transition_stats.pkl', 'feature_cols.pkl', 'ltr_fe.pkl']
            missing = [f for f in expected if not os.path.exists(os.path.join(session_path, f))]
            
            if missing:
                return None, f"Training failed to generate artifacts: {missing}"
            
            # Load Metrics
            import json
            metrics = {}
            m_path = os.path.join(session_path, 'metrics.json')
            if os.path.exists(m_path):
                with open(m_path, 'r') as f:
                    metrics = json.load(f)
                
            return session_id, metrics
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Pipeline Error: {e}"
            
    def commit_model(self, session_id):
        """Promotes staging model to production after backing up current."""
        staging_path = os.path.join(self.staging_dir, session_id)
        if not os.path.exists(staging_path):
            return False, "Staging session not found."
            
        # 1. Backup Current
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, backup_id)
        
        try:
            # Copy specific LTR files from prod to backup
            # Or just copy entire directory?
            # shutil.copytree is safest but prod_dir might have other stuff?
            # Let's copytree the prod_dir contents to backup
            # If prod dir doesn't exist, skip backup
            if os.path.exists(self.prod_dir):
                shutil.copytree(self.prod_dir, backup_path, dirs_exist_ok=True)
        except Exception as e:
            return False, f"Backup failed: {e}"
            
        # 2. Promote New (Overwrite)
        # Copy files from staging to prod
        try:
            # We only copy the critical artifacts
            files = glob.glob(os.path.join(staging_path, "*.pkl"))
            for f in files:
                shutil.copy2(f, self.prod_dir)
                
            return True, f"Model Deployed. Backup saved to {backup_id}."
        except Exception as e:
            return False, f"Deployment failed: {e}"
            
    def rollback(self):
        """Restores the most recent backup."""
        # Find latest backup
        backups = sorted(glob.glob(os.path.join(self.backup_dir, "*")), reverse=True)
        if not backups:
            return False, "No backups found."
            
        latest = backups[0]
        try:
            # Copy files from backup to prod
            files = glob.glob(os.path.join(latest, "*.pkl"))
            for f in files:
                shutil.copy2(f, self.prod_dir)
                
            return True, f"Rolled back to {os.path.basename(latest)}"
        except Exception as e:
            return False, f"Rollback error: {e}"
