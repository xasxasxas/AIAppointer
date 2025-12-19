"""
Enhanced Gantt Chart Visualization for Complete HR Temporal Analysis
Shows appointments, training, and promotions in comprehensive timeline
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re


def parse_appointment_with_dates(appointment_history_str):
    """Parse appointment history string to extract roles with dates."""
    if not isinstance(appointment_history_str, str) or not appointment_history_str:
        return []
    
    appointments = []
    items = appointment_history_str.split(',')
    
    for i, item in enumerate(items):
        item = item.strip()
        if not item:
            continue
        
        match = re.match(r'(.+?)\s*\(([^)]+)\)', item)
        
        if match:
            title = match.group(1).strip()
            date_str = match.group(2).strip()
            start_date = parse_date_flexible(date_str)
            
            # End date is the start of next appointment
            if i < len(items) - 1:
                next_item = items[i + 1].strip()
                next_match = re.match(r'(.+?)\s*\(([^)]+)\)', next_item)
                if next_match:
                    end_date = parse_date_flexible(next_match.group(2).strip())
                else:
                    end_date = None
            else:
                end_date = None
            
            appointments.append({
                'title': title,
                'start_date': start_date,
                'end_date': end_date
            })
    
    return appointments


def parse_training_with_dates(training_history_str):
    """Parse training history to extract courses with dates."""
    if not isinstance(training_history_str, str) or not training_history_str:
        return []
    
    trainings = []
    items = training_history_str.split(',')
    
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        match = re.match(r'(.+?)\s*\(([^)]+)\)', item)
        if match:
            course = match.group(1).strip()
            date_str = match.group(2).strip()
            date = parse_date_flexible(date_str)
            
            if date:
                trainings.append({'course': course, 'date': date})
    
    return trainings


def detect_promotions(appointment_history):
    """Detect promotions by analyzing rank changes."""
    promotions = []
    rank_order = ['Ensign', 'Lieutenant', 'Lt. Commander', 'Commander', 'Captain', 'Admiral']
    
    for i in range(len(appointment_history) - 1):
        curr_title = appointment_history[i]['title']
        next_title = appointment_history[i + 1]['title']
        
        curr_rank = None
        next_rank = None
        
        for rank in rank_order:
            if rank.lower() in curr_title.lower():
                curr_rank = rank
            if rank.lower() in next_title.lower():
                next_rank = rank
        
        if curr_rank and next_rank and rank_order.index(next_rank) > rank_order.index(curr_rank):
            promotions.append({
                'from_rank': curr_rank,
                'to_rank': next_rank,
                'date': appointment_history[i + 1]['start_date'],
                'title': next_title
            })
    
    return promotions


def parse_date_flexible(date_str):
    """Parse date string in various formats."""
    if not date_str:
        return None
    
    formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%d/%m/%Y']
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    year_match = re.search(r'(\d{4})', date_str)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 1, 1)
        except:
            pass
    
    return None


def create_comprehensive_officer_timeline(df, filter_branch=None, start_date=None, end_date=None):
    """
    Create comprehensive Gantt chart showing appointments, training, and promotions for each officer.
    """
    timeline_data = []
    training_markers = []
    promotion_markers = []
    
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    for _, officer in filtered_df.iterrows():
        officer_name = f"{officer.get('Name', 'Unknown')} ({officer.get('Employee_ID', 'N/A')})"
        
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        trainings = parse_training_with_dates(officer.get('Training_history', ''))
        promotions = detect_promotions(appointments)
        
        for appt in appointments:
            if not appt['start_date']:
                continue
            
            if start_date and appt['start_date'] < start_date:
                continue
            if end_date and appt['start_date'] > end_date:
                continue
            
            end_date_val = appt['end_date'] if appt['end_date'] else datetime.now()
            
            timeline_data.append({
                'Officer': officer_name,
                'Task': appt['title'],
                'Start': appt['start_date'],
                'End': end_date_val,
                'Type': 'Appointment',
                'Branch': officer.get('Branch', 'Unknown'),
                'Rank': officer.get('Rank', 'Unknown')
            })
        
        for training in trainings:
            if start_date and training['date'] < start_date:
                continue
            if end_date and training['date'] > end_date:
                continue
            
            training_markers.append({
                'Officer': officer_name,
                'Date': training['date'],
                'Course': training['course']
            })
        
        for promo in promotions:
            if not promo['date']:
                continue
            if start_date and promo['date'] < start_date:
                continue
            if end_date and promo['date'] > end_date:
                continue
            
            promotion_markers.append({
                'Officer': officer_name,
                'Date': promo['date'],
                'From': promo['from_rank'],
                'To': promo['to_rank']
            })
    
    if not timeline_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No appointment data with valid dates found.<br>Ensure appointment history includes dates in format: Role (YYYY-MM-DD)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title="Officer Timeline", height=400)
        return fig
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        timeline_df,
        x_start='Start',
        x_end='End',
        y='Officer',
        color='Branch',
        hover_data=['Task', 'Rank'],
        title=f'Complete Officer Timeline ({len(timeline_df)} appointments)'
    )
    
    # Add training markers (green diamonds)
    for training in training_markers:
        fig.add_trace(go.Scatter(
            x=[training['Date']],
            y=[training['Officer']],
            mode='markers',
            marker=dict(symbol='diamond', size=12, color='green', line=dict(width=2, color='darkgreen')),
            name='Training',
            hovertext=f"Training: {training['Course']}",
            showlegend=False
        ))
    
    # Add promotion markers (gold stars)
    for promo in promotion_markers:
        fig.add_trace(go.Scatter(
            x=[promo['Date']],
            y=[promo['Officer']],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold', line=dict(width=2, color='orange')),
            name='Promotion',
            hovertext=f"Promotion: {promo['From']} → {promo['To']}",
            showlegend=False
        ))
    
    fig.update_yaxes(categoryorder='total ascending', title='Officers')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        height=max(600, len(timeline_df['Officer'].unique()) * 40),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def create_billet_occupancy_timeline(df, filter_branch=None, start_date=None, end_date=None):
    """
    Create Gantt chart showing which officers held each billet over time.
    """
    billet_data = []
    
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    for _, officer in filtered_df.iterrows():
        officer_name = f"{officer.get('Name', 'Unknown')} ({officer.get('Employee_ID', 'N/A')})"
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        
        for appt in appointments:
            if not appt['start_date']:
                continue
            
            if start_date and appt['start_date'] < start_date:
                continue
            if end_date and appt['start_date'] > end_date:
                continue
            
            end_date_val = appt['end_date'] if appt['end_date'] else datetime.now()
            
            billet_data.append({
                'Billet': appt['title'],
                'Officer': officer_name,
                'Start': appt['start_date'],
                'End': end_date_val,
                'Branch': officer.get('Branch', 'Unknown'),
                'Rank': officer.get('Rank', 'Unknown')
            })
    
    if not billet_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No billet occupancy data found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Billet Occupancy Timeline", height=400)
        return fig
    
    billet_df = pd.DataFrame(billet_data)
    billet_df = billet_df.sort_values(['Billet', 'Start'])
    
    fig = px.timeline(
        billet_df,
        x_start='Start',
        x_end='End',
        y='Billet',
        color='Branch',
        hover_data=['Officer', 'Rank'],
        title=f'Billet Occupancy Timeline ({len(billet_df)} assignments)',
        labels={'Billet': 'Position', 'Officer': 'Officer'}
    )
    
    fig.update_yaxes(categoryorder='total ascending', title='Billets/Positions')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        height=max(600, len(billet_df['Billet'].unique()) * 40),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig
