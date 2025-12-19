"""
Gantt Chart Visualization for Temporal Appointment Analysis
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re


def parse_appointment_with_dates(appointment_history_str):
    """
    Parse appointment history string to extract roles with dates.
    
    Format expected: "Role1 (Date1), Role2 (Date2), ..."
    
    Args:
        appointment_history_str: String containing appointment history
        
    Returns:
        List of dicts with 'title', 'start_date', 'end_date'
    """
    if not isinstance(appointment_history_str, str) or not appointment_history_str:
        return []
    
    appointments = []
    items = appointment_history_str.split(',')
    
    for i, item in enumerate(items):
        item = item.strip()
        if not item:
            continue
        
        # Extract title and date using regex
        # Pattern: "Title (Date)" or just "Title"
        match = re.match(r'(.+?)\s*\(([^)]+)\)', item)
        
        if match:
            title = match.group(1).strip()
            date_str = match.group(2).strip()
            
            # Try to parse date (various formats)
            start_date = parse_date_flexible(date_str)
            
            # End date is the start of next appointment, or "Present" if last
            if i < len(items) - 1:
                next_item = items[i + 1].strip()
                next_match = re.match(r'(.+?)\s*\(([^)]+)\)', next_item)
                if next_match:
                    end_date = parse_date_flexible(next_match.group(2).strip())
                else:
                    end_date = None
            else:
                end_date = None  # Current/last appointment
            
            appointments.append({
                'title': title,
                'start_date': start_date,
                'end_date': end_date
            })
    
    return appointments


def parse_date_flexible(date_str):
    """
    Parse date string in various formats.
    
    Supports:
    - YYYY-MM-DD
    - YYYY-MM
    - YYYY
    - Stardate formats
    
    Returns:
        datetime object or None
    """
    if not date_str:
        return None
    
    # Try standard formats
    formats = [
        '%Y-%m-%d',
        '%Y-%m',
        '%Y',
        '%m/%d/%Y',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract just year if present
    year_match = re.search(r'(\d{4})', date_str)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 1, 1)
        except:
            pass
    
    return None


def create_appointment_gantt(df, filter_branch=None, filter_role=None, max_officers=50):
    """
    Create temporal Gantt chart of appointments.
    
    Args:
        df: DataFrame with officer data
        filter_branch: Optional branch filter
        filter_role: Optional role filter
        max_officers: Maximum number of officers to display
        
    Returns:
        Plotly figure
    """
    gantt_data = []
    
    # Filter dataframe
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    # Limit to max officers for performance
    filtered_df = filtered_df.head(max_officers)
    
    for _, officer in filtered_df.iterrows():
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        
        for appt in appointments:
            if not appt['start_date']:
                continue
            
            # Filter by role if specified
            if filter_role and filter_role != "All" and appt['title'] != filter_role:
                continue
            
            # Use current date if no end date
            end_date = appt['end_date'] if appt['end_date'] else datetime.now()
            
            gantt_data.append({
                'Officer': f"{officer.get('Name', 'Unknown')} ({officer.get('Employee_ID', 'N/A')})",
                'Role': appt['title'],
                'Start': appt['start_date'],
                'End': end_date,
                'Branch': officer.get('Branch', 'Unknown'),
                'Rank': officer.get('Rank', 'Unknown'),
                'Employee_ID': officer.get('Employee_ID', 'N/A')
            })
    
    if not gantt_data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No appointment data with valid dates found.<br>Ensure appointment history includes dates in format: Role (YYYY-MM-DD)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Temporal Appointment Timeline",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    gantt_df = pd.DataFrame(gantt_data)
    
    # Sort by start date
    gantt_df = gantt_df.sort_values('Start')
    
    # Create Gantt chart
    fig = px.timeline(
        gantt_df,
        x_start='Start',
        x_end='End',
        y='Officer',
        color='Branch',
        hover_data=['Role', 'Rank', 'Employee_ID'],
        title=f'Temporal Appointment Timeline ({len(gantt_df)} appointments)',
        labels={'Officer': 'Officer', 'Start': 'Start Date', 'End': 'End Date'}
    )
    
    # Update layout
    fig.update_yaxes(categoryorder='total ascending', title='Officers')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        height=max(600, len(gantt_df) * 20),  # Dynamic height based on data
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def create_role_timeline(df, role_name):
    """
    Create timeline showing all officers who held a specific role over time.
    
    Args:
        df: DataFrame with officer data
        role_name: Name of the role to analyze
        
    Returns:
        Plotly figure
    """
    timeline_data = []
    
    for _, officer in df.iterrows():
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        
        for appt in appointments:
            if appt['title'] == role_name and appt['start_date']:
                end_date = appt['end_date'] if appt['end_date'] else datetime.now()
                
                timeline_data.append({
                    'Officer': f"{officer.get('Name', 'Unknown')}",
                    'Start': appt['start_date'],
                    'End': end_date,
                    'Branch': officer.get('Branch', 'Unknown'),
                    'Rank': officer.get('Rank', 'Unknown'),
                    'Employee_ID': officer.get('Employee_ID', 'N/A')
                })
    
    if not timeline_data:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No historical data found for role: {role_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df = timeline_df.sort_values('Start')
    
    fig = px.timeline(
        timeline_df,
        x_start='Start',
        x_end='End',
        y='Officer',
        color='Branch',
        hover_data=['Rank', 'Employee_ID'],
        title=f'Officers Who Held Role: {role_name}',
        labels={'Officer': 'Officer', 'Start': 'Start Date', 'End': 'End Date'}
    )
    
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(height=max(400, len(timeline_df) * 30))
    
    return fig
