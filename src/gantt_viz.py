"""
Enhanced Gantt Chart Visualization for Complete HR Temporal Analysis
Shows appointments, training, and promotions in comprehensive timeline
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    """Parse training history to extract courses with start and end dates."""
    if not isinstance(training_history_str, str) or not training_history_str:
        return []
    
    trainings = []
    items = training_history_str.split(',')
    
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        # Try to match "Course (StartDate - EndDate)" or "Course (Date)"
        match_range = re.match(r'(.+?)\s*\(([^)]+?)\s*-\s*([^)]+)\)', item)
        match_single = re.match(r'(.+?)\s*\(([^)]+)\)', item)
        
        if match_range:
            course = match_range.group(1).strip()
            start_date = parse_date_flexible(match_range.group(2).strip())
            end_date = parse_date_flexible(match_range.group(3).strip())
            
            if start_date:
                trainings.append({
                    'course': course,
                    'start_date': start_date,
                    'end_date': end_date if end_date else start_date + timedelta(days=30)
                })
        elif match_single:
            course = match_single.group(1).strip()
            date = parse_date_flexible(match_single.group(2).strip())
            
            if date:
                trainings.append({
                    'course': course,
                    'start_date': date,
                    'end_date': date + timedelta(days=30)  # Default 30 days duration
                })
    
    return trainings


def detect_promotions(appointment_history, officer_rank=None):
    """Detect promotions by analyzing rank changes."""
    promotions = []
    rank_order = ['Ensign', 'Lieutenant', 'Lt. Commander', 'Commander', 'Captain', 'Commodore', 'Admiral', 'Fleet Admiral']
    
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
        
        if curr_rank and next_rank:
            curr_idx = rank_order.index(curr_rank)
            next_idx = rank_order.index(next_rank)
            if next_idx > curr_idx:
                promotions.append({
                    'from_rank': curr_rank,
                    'to_rank': next_rank,
                    'date': appointment_history[i + 1]['start_date'],
                    'title': next_title
                })
    
    return promotions


def parse_promotion_history(promotion_history_str):
    """
    Parse promotion_history column to extract promotion dates and ranks.
    Format: "Lieutenant (jg)(02 JUL 2002 - 02 JUL 2007), Lieutenant(02 JUL 2007 - 02 JUL 2012), ..."
    """
    if not isinstance(promotion_history_str, str) or not promotion_history_str:
        return []
    
    promotions = []
    items = promotion_history_str.split(',')
    
    prev_rank = None
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        # Robust Regex: Capture everything before the last (...) as the Rank
        # This preserves "Lieutenant (jg)" instead of stripping "(jg)"
        # Matches: "Lieutenant (jg)(02 JUL 2002 - 02 JUL 2007)" -> Group 1: "Lieutenant (jg)", Group 2: "02 JUL 2002 - 02 JUL 2007"
        match = re.search(r'^(.*)\(([^)]+)\)$', item)
        
        if match:
            rank = match.group(1).strip()
            date_range = match.group(2).strip()
            
            # Extract start date from range
            date_parts = date_range.split('-')
            if date_parts:
                start_date_str = date_parts[0].strip()
                start_date = parse_date_flexible(start_date_str)
                
                # Verify date found
                if start_date:
                     # If we have a previous rank, track the transition
                    if prev_rank:
                        # Even if ranks are string-identical, if there's a new date entry, 
                        # it might be worth noting, but typically we want actual changes.
                        # With better regex, 'Lieutenant (jg)' != 'Lieutenant', so this works better.
                        if rank != prev_rank:
                            promotions.append({
                                'from_rank': prev_rank,
                                'to_rank': rank,
                                'date': start_date
                            })
                
                    prev_rank = rank
    
    return promotions



def parse_date_flexible(date_str):
    """Parse date string in various formats."""
    if not date_str:
        return None
    
    formats = [
        '%d %b %Y',  # 02 JUL 2002
        '%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'
    ]
    
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
    appointment_data = []
    training_data = []
    promotion_annotations = []
    
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    # Define rank order for sorting (senior to junior)
    rank_order = {
        'Fleet Admiral': 0, 'Admiral': 1, 'Commodore': 2, 'Captain': 3,
        'Commander': 4, 'Lt. Commander': 5, 'Lieutenant': 6, 'Ensign': 7
    }
    
    # Branch color palette (distinct from green training color)
    branch_colors = {
        'Tactical': '#e74c3c',      # Red
        'Engineering': '#3498db',    # Blue
        'Science': '#9b59b6',        # Purple
        'Medical': '#f39c12',        # Orange
        'Operations': '#1abc9c',     # Teal
        'Command': '#34495e'         # Dark gray
    }
    
    officers_with_data = []
    officer_info = {}  # Store rank and branch for sorting
    
    for _, officer in filtered_df.iterrows():
        emp_id = officer.get('Employee_ID', 'N/A')
        name = officer.get('Name', 'Unknown')
        rank = officer.get('Rank', 'Unknown')
        branch = officer.get('Branch', 'Unknown')
        
        # Create display label: "Rank Name (ID) - Branch"
        officer_label = f"{rank} {name} ({emp_id}) - {branch}"
        
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        trainings = parse_training_with_dates(officer.get('Training_history', ''))
        promotions = parse_promotion_history(officer.get('Promotion_history', ''))
        
        has_data = False
        
        # Add appointments with branch color
        for appt in appointments:
            if not appt['start_date']:
                continue
            
            if start_date and appt['start_date'] < start_date:
                continue
            if end_date and appt['start_date'] > end_date:
                continue
            
            end_date_val = appt['end_date'] if appt['end_date'] else datetime.now()
            
            appointment_data.append({
                'Officer': officer_label,
                'Task': appt['title'],
                'Start': appt['start_date'],
                'End': end_date_val,
                'Type': 'Appointment',
                'Branch': branch,
                'Rank': rank,
                'Color': branch_colors.get(branch, '#95a5a6')  # Default gray
            })
            has_data = True
        
        # Add training as duration bars (green)
        for training in trainings:
            if start_date and training['start_date'] < start_date:
                continue
            if end_date and training['start_date'] > end_date:
                continue
            
            training_data.append({
                'Officer': officer_label,
                'Task': f"[TRAINING] {training['course']}",
                'Start': training['start_date'],
                'End': training['end_date'],
                'Type': 'Training',
                'Branch': branch,
                'Rank': rank,
                'Color': '#2ca02c'  # Green for training
            })
            has_data = True
        
        # Track promotions for annotations
        for promo in promotions:
            if not promo['date']:
                continue
            if start_date and promo['date'] < start_date:
                continue
            if end_date and promo['date'] > end_date:
                continue
            
            promotion_annotations.append({
                'Officer': officer_label,
                'Date': promo['date'],
                'From': promo['from_rank'],
                'To': promo['to_rank']
            })
            has_data = True
        
        if has_data:
            officers_with_data.append(officer_label)
            officer_info[officer_label] = {
                'rank': rank,
                'rank_order': rank_order.get(rank, 99),
                'branch': branch
            }
    
    # DEBUG: Print promotion detection results
    print(f"DEBUG: Found {len(promotion_annotations)} promotions across {len(officers_with_data)} officers")
    if promotion_annotations:
        print(f"DEBUG: First promotion: {promotion_annotations[0]}")
    
    if not appointment_data and not training_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No appointment data with valid dates found.<br>Ensure appointment history includes dates in format: Role (YYYY-MM-DD)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title="Officer Timeline", height=400)
        return fig
    
    # Combine all data
    all_data = appointment_data + training_data
    timeline_df = pd.DataFrame(all_data)
    
    # Sort officers by rank (senior to junior)
    sorted_officers = sorted(officers_with_data, key=lambda x: officer_info[x]['rank_order'])
    
    # Altair Implementation
    import altair as alt
    
    # Ensure dataframes are ready
    timeline_df = pd.DataFrame(all_data)
    
    # Ensure 'Branch' column exists and proper string type
    if 'Branch' in timeline_df.columns:
        timeline_df['Branch'] = timeline_df['Branch'].fillna('Unknown').astype(str).str.strip()

    # Define High-Contrast Palette
    # Define High-Contrast Palette with UNIQUE colors for all known branches from dataset
    # 2. Define High-Contrast Palette with UNIQUE colors
    branch_palette = {
        'Tactical': '#D62728',      'Tactical Systems': '#D62728',
        'Engineering': '#1F77B4',   'Hull Systems': '#17BECF',
        'Science': '#9467BD',       'Medical': '#FF7F0E',
        'Command': '#E377C2',       'Operations': '#2CA02C',
        'Training': '#BCBD22',      'Promotion': '#FFD700', # GOLD for Promotion
        'Unknown': '#7F7F7F'
    }
    
    # 2b. Integrate Promotions into Main Dataframe
    # This ensures they share the exact same Axis, Scale, and Legend logic
    if promotion_annotations:
        for p in promotion_annotations:
            # Add to main list
            if p.get('Date'):
                 all_data.append({
                    'Officer': p['Officer'],
                    'Task': f"Promotion: {p['From']} -> {p['To']}",
                    'Start': p['Date'],
                    'End': p['Date'] + timedelta(days=7), # FIX: Add duration to ensure visibility
                    'Type': 'Promotion',
                    'Branch': 'Promotion', # Map to Palette
                    'Rank': p['To'],
                    'Color': '#FFD700'
                })

    # Re-create DataFrame with ALL data
    timeline_df = pd.DataFrame(all_data)
    
    # Ensure types
    if 'Branch' in timeline_df.columns:
        timeline_df['Branch'] = timeline_df['Branch'].fillna('Unknown').astype(str).str.strip()
    
    # RENDER FIX: Safe Domain Calculation from UNIFIED data
    if not timeline_df.empty:
        # Filter out NaT
        valid_starts = timeline_df['Start'].dropna()
        valid_ends = timeline_df['End'].dropna()
        if not valid_starts.empty:
            min_date = valid_starts.min()
            max_date = valid_ends.max()
        else:
             min_date, max_date = datetime(2360, 1, 1), datetime(2370, 1, 1)
    else:
        min_date, max_date = datetime(2360, 1, 1), datetime(2370, 1, 1)

    if pd.isna(min_date): min_date = datetime(2360, 1, 1)
    if pd.isna(max_date): max_date = datetime(2370, 1, 1)

    # Y-axis 
    y_axis = alt.Y('Officer', sort=sorted_officers, axis=alt.Axis(
        title=None, labels=True, labelLimit=300, labelFontSize=11
    ))
    
    # Base chart
    base = alt.Chart(timeline_df).encode(y=y_axis)
    
    # Common X-axis
    x_axis = alt.X('Start', title='Timeline', 
                   scale=alt.Scale(domain=[min_date, max_date], nice=True, padding=10))

    # Color Scale (Unified)
    data_branches = sorted(timeline_df['Branch'].unique())
    # Ensure Promotion is last if present for legend order? Or just sort specific ones.
    # Prioritize standard branches then Promotion
    priority_order = ['Tactical', 'Engineering', 'Science', 'Medical', 'Command', 'Operations', 'Promotion']
    final_domain = [b for b in priority_order if b in data_branches] + [b for b in data_branches if b not in priority_order]
    
    final_colors = [branch_palette.get(b, '#7F7F7F') for b in final_domain]
    
    color_scale = alt.Color('Branch', scale=alt.Scale(domain=final_domain, range=final_colors), 
                            legend=alt.Legend(title="Category", orient='top', symbolType='square', columns=min(7, len(final_domain))))

    # 1. Appointments
    appointments = base.transform_filter(
        alt.datum.Type == 'Appointment'
    ).mark_bar(
        height=12, stroke='white', strokeWidth=0.5, cornerRadius=1, opacity=0.9
    ).encode(
        x=x_axis,
        x2='End',
        color=color_scale,
        tooltip=['Officer', 'Task', 'Branch', 'Rank', alt.Tooltip('Start', format='%Y-%m-%d'), alt.Tooltip('End', format='%Y-%m-%d')]
    )
    
    # 2. Training
    training = base.transform_filter(
        alt.datum.Type == 'Training'
    ).mark_bar(
        height=6, color='#BCBD22', opacity=0.9
    ).encode(
        x=x_axis,
        x2='End',
        tooltip=['Officer', 'Task', alt.Tooltip('Start', format='%Y-%m-%d'), alt.Tooltip('End', format='%Y-%m-%d')]
    )
    
    # 3. Promotions (Now derived from BASE)
    promotions = base.transform_filter(
        alt.datum.Type == 'Promotion'
    ).mark_point(
        shape='star', size=200, filled=True, opacity=1
    ).encode(
        x=x_axis, # Same X axis
        color=color_scale, # Same Color Scale (Maps 'Promotion' -> Gold)
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.5),
        tooltip=['Officer', 'Task', alt.Tooltip('Start', format='%Y-%m-%d', title='Date')],
        order=alt.value(999) # Explicit Z-order
    )
    
    # Combine layers (No resolve scale needed!)
    chart = (appointments + training + promotions).properties(
        title=alt.TitleParams(
            text=f'Complete Officer Timeline ({len(appointment_data)} appts, {len(training_data)} training)',
            anchor='start',
            fontSize=16
        ),
        width='container'
    ).interactive() 
    
    # Force configuration for auto-scaling
    # CRITICAL FIX: Reduce step to 15 to keep total height well under 32k pixels (browser limit) for 1400 rows
    row_height = 15
    chart = chart.configure_view(
        step=row_height 
    ).configure_axis(
        grid=True,
        gridOpacity=0.2,
        domain=False
    )
    
    # Return chart and calculated height (redundant with step config but kept for app.py compatibility)
    return chart, max(500, len(officers_with_data) * row_height)




def create_billet_occupancy_timeline(df, filter_branch=None, start_date=None, end_date=None):
    """
    Create Gantt chart showing which officers held each billet over time.
    Uses simple branch colors for clarity.
    """
    billet_data = []
    
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    # Branch color palette
    branch_colors = {
        'Tactical': '#e74c3c',      # Red
        'Engineering': '#3498db',    # Blue
        'Science': '#9b59b6',        # Purple
        'Medical': '#f39c12'         # Orange
    }
    
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
    
    # Create figure with branch coloring (simple, no gradients)
    fig = px.timeline(
        billet_df,
        x_start='Start',
        x_end='End',
        y='Billet',
        color='Branch',  # Color by branch only
        hover_data=['Officer', 'Rank'],
        title=f'Billet Occupancy Timeline ({len(billet_df)} assignments)',
        labels={'Billet': 'Position', 'Officer': 'Officer'},
        color_discrete_map=branch_colors
    )
    
    # Add white borders for separation
    fig.update_traces(
        marker=dict(line=dict(color='white', width=2))
    )
    
    fig.update_yaxes(categoryorder='total ascending', title='Billets/Positions')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        height=max(400, len(billet_df['Billet'].unique()) * 50),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=250, r=50, t=80, b=50)
    )
    
    return fig

    billet_data = []
    
    filtered_df = df.copy()
    if filter_branch and filter_branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == filter_branch]
    
    # Define rank order for gradient intensity
    rank_order = {
        'Fleet Admiral': 1.0, 'Admiral': 0.9, 'Commodore': 0.8, 'Captain': 0.7,
        'Commander': 0.6, 'Lt. Commander': 0.5, 'Lieutenant': 0.4, 'Ensign': 0.3
    }
    
    # Base branch colors
    branch_base_colors = {
        'Tactical': (231, 76, 60),      # Red
        'Engineering': (52, 152, 219),   # Blue
        'Science': (155, 89, 182),       # Purple
        'Medical': (243, 156, 18)        # Orange
    }
    
    for _, officer in filtered_df.iterrows():
        officer_name = f"{officer.get('Name', 'Unknown')} ({officer.get('Employee_ID', 'N/A')})"
        rank = officer.get('Rank', 'Unknown')
        branch = officer.get('Branch', 'Unknown')
        appointments = parse_appointment_with_dates(officer.get('Appointment_history', ''))
        
        # Get rank intensity for gradient
        rank_intensity = rank_order.get(rank, 0.5)
        
        # Create color with gradient based on rank
        if branch in branch_base_colors:
            base_r, base_g, base_b = branch_base_colors[branch]
            # Lighter for junior ranks, darker for senior ranks
            r = int(base_r * rank_intensity + 255 * (1 - rank_intensity))
            g = int(base_g * rank_intensity + 255 * (1 - rank_intensity))
            b = int(base_b * rank_intensity + 255 * (1 - rank_intensity))
            color = f'rgb({r},{g},{b})'
        else:
            color = f'rgb(150,150,150)'  # Gray for unknown
        
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
                'Branch': branch,
                'Rank': rank,
                'Color': color,
                'RankIntensity': rank_intensity
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
    
    # Create figure manually to use custom colors
    fig = go.Figure()
    
    # Group by billet for y-axis positioning
    billets = billet_df['Billet'].unique()
    
    for _, row in billet_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['End'] - row['Start']],
            y=[row['Billet']],
            base=row['Start'],
            orientation='h',
            marker=dict(
                color=row['Color'],
                line=dict(color='white', width=2)
            ),
            hovertemplate=f"<b>{row['Billet']}</b><br>" +
                         f"Officer: {row['Officer']}<br>" +
                         f"Rank: {row['Rank']}<br>" +
                         f"Branch: {row['Branch']}<br>" +
                         f"Period: {row['Start'].strftime('%Y-%m-%d')} to {row['End'].strftime('%Y-%m-%d')}<br>" +
                         "<extra></extra>",
            showlegend=False,
            name=row['Branch']
        ))
    
    # Add legend manually for branches
    for branch, (r, g, b) in branch_base_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=f'rgb({r},{g},{b})'),
            name=f'{branch} (darker = senior)',
            showlegend=True
        ))
    
    fig.update_yaxes(categoryorder='total ascending', title='Billets/Positions')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        title=f'Billet Occupancy Timeline ({len(billet_df)} assignments)<br><sub>Color gradient: Darker shades = Senior ranks, Lighter shades = Junior ranks</sub>',
        height=max(400, len(billets) * 50),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=250, r=50, t=100, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Branch (Rank Gradient)"
        ),
        barmode='overlay'
    )
    
    return fig

    """
    Create Gantt chart showing which officers held each billet over time with clear separation.
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
    
    # Create figure with officer-based coloring for better separation
    fig = px.timeline(
        billet_df,
        x_start='Start',
        x_end='End',
        y='Billet',
        color='Officer',  # Color by officer for clear separation
        hover_data=['Branch', 'Rank'],
        title=f'Billet Occupancy Timeline ({len(billet_df)} assignments)',
        labels={'Billet': 'Position', 'Officer': 'Officer'}
    )
    
    # Add subtle borders between bars
    fig.update_traces(
        marker=dict(line=dict(color='white', width=2))
    )
    
    fig.update_yaxes(categoryorder='total ascending', title='Billets/Positions')
    fig.update_xaxes(title='Timeline')
    fig.update_layout(
        height=max(400, len(billet_df['Billet'].unique()) * 50),
        showlegend=True,
        hovermode='closest',
        margin=dict(l=250, r=50, t=80, b=50),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title="Officers"
        )
    )
    
    return fig
