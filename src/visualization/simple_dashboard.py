#!/usr/bin/env python3
"""
Simple Currency Analysis Dashboard

A clean, simple interface for analyzing currency seasonal patterns.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.schema import DataValidator
from src.data.preprocessing import DataPreprocessor

class SimpleCurrencyDashboard:
    """Simple dashboard for currency seasonal analysis"""
    
    def __init__(self, title="Simple Currency Analysis"):
        self.title = title
        self.app = dash.Dash(__name__)
        
        # Cache for loaded data
        self._currency_data = None
        self._processed_df = None
        self._last_load_time = None
        
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data_once(self):
        """Load data only once and cache it"""
        if self._currency_data is None:
            print("Loading currency data...")
            # Use the original data file with real values
            self._currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
            print("Using original data with real exchange rates")
            
            preprocessor = DataPreprocessor()
            self._processed_df = preprocessor.convert_to_dataframe(self._currency_data)
            # Remove duplicates to prevent reshape errors
            self._processed_df = self._processed_df.drop_duplicates(subset=['day', 'currency_code']).reset_index(drop=True)
            
            # Apply conversion factor for data before January 2, 2018
            self._processed_df = self.apply_currency_conversion(self._processed_df)
            
            self._last_load_time = pd.Timestamp.now()
            print("Data loaded successfully!")
        
        return self._currency_data, self._processed_df
    
    def apply_currency_conversion(self, df):
        """Apply conversion factor for data before January 2, 2018"""
        # Convert day column to datetime if it's not already
        df['day'] = pd.to_datetime(df['day'])
        
        # Define the cutoff date
        cutoff_date = pd.to_datetime('2018-01-02')
        
        # Apply conversion: divide by 10 for data before January 2, 2018
        mask = df['day'] < cutoff_date
        df.loc[mask, 'value'] = df.loc[mask, 'value'] / 10
        
        print(f"Applied conversion factor: {mask.sum()} data points before 2018-01-02 divided by 10")
        
        return df
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.H1(self.title, style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                # Currency Selector
                html.Div([
                    html.Label("Select Currency:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Dropdown(
                        id='currency-selector',
                        placeholder="Choose a currency...",
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': 20}),
                
                # Year Range Selector
                html.Div([
                    html.Label("Select Year Range:", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Dropdown(
                        id='year-range-selector',
                        options=[
                            {'label': 'All Years', 'value': 'all'},
                            {'label': 'Single Year', 'value': 'single'},
                            {'label': 'Year Interval', 'value': 'interval'}
                        ],
                        value='all',
                        style={'width': '100%'}
                    )
                ], style={'width': '20%', 'display': 'inline-block', 'marginRight': 20}),
                
                # Year Selector (conditional)
                html.Div([
                    html.Label("Select Year(s):", style={'fontWeight': 'bold', 'marginBottom': 5}),
                    dcc.Dropdown(
                        id='year-selector',
                        placeholder="Select year(s)...",
                        multi=True,
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'marginBottom': 30, 'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 10}),
            
            # Main Chart
            html.Div([
                dcc.Loading(
                    id="loading-chart",
                    type="default",
                    children=[
                        dcc.Graph(
                            id='currency-chart',
                            style={'height': 600}
                        )
                    ]
                )
            ], style={'marginBottom': 30}),
            
            # Report Section
            html.Div([
                html.H3("Analysis Report", style={'color': '#2c3e50', 'marginBottom': 20}),
                dcc.Loading(
                    id="loading-report",
                    type="default",
                    children=[
                        html.Div(id='analysis-report', style={
                            'backgroundColor': '#ffffff',
                            'padding': 20,
                            'borderRadius': 10,
                            'border': '1px solid #e1e5e9',
                            'lineHeight': 1.6
                        })
                    ]
                )
            ])
        ], style={'padding': 20, 'fontFamily': 'Arial, sans-serif'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('currency-selector', 'options'),
            Output('currency-selector', 'value'),
            Input('currency-selector', 'value')
        )
        def update_currency_options(value):
            """Update currency options from data"""
            try:
                currency_data, _ = self.load_data_once()
                options = []
                for curr in currency_data.data:
                    label = f"{curr.code} - {curr.name_fr}"
                    options.append({'label': label, 'value': curr.code})
                # Don't auto-select first currency, let user choose
                return options, value if value else None
            except Exception as e:
                return [], None
        
        @self.app.callback(
            Output('year-selector', 'options'),
            Output('year-selector', 'value'),
            Input('currency-selector', 'value'),
            Input('year-range-selector', 'value')
        )
        def update_year_options(currency, range_type):
            """Update year options based on selected currency and range type"""
            if not currency:
                return [], []
            
            try:
                _, df = self.load_data_once()
                
                # Filter by currency - use copy to avoid warnings
                df_filtered = df[df['currency_code'] == currency].copy()
                
                # Get available years
                df_filtered['year'] = pd.to_datetime(df_filtered['day']).dt.year
                available_years = sorted(df_filtered['year'].unique())
                
                options = [{'label': str(year), 'value': year} for year in available_years]
                
                if range_type == 'all':
                    return options, available_years
                elif range_type == 'single':
                    return options, [available_years[0]] if available_years else []
                else:  # interval
                    return options, available_years[:3] if len(available_years) >= 3 else available_years
                    
            except Exception as e:
                return [], []
        
        @self.app.callback(
            Output('currency-chart', 'figure'),
            Output('analysis-report', 'children'),
            Input('currency-selector', 'value'),
            Input('year-selector', 'value')
        )
        def update_chart_and_report(currency, years):
            """Update chart and report based on selections"""
            if not currency or not years:
                return self.create_empty_chart(), "Please select a currency and year(s)."
            
            try:
                # Use cached data
                _, df = self.load_data_once()
                
                # Filter by currency and years - use copy to avoid warnings
                df_filtered = df[df['currency_code'] == currency].copy()
                df_filtered['year'] = pd.to_datetime(df_filtered['day']).dt.year
                df_filtered = df_filtered[df_filtered['year'].isin(years)]
                
                # Create monthly averages
                df_filtered['month'] = pd.to_datetime(df_filtered['day']).dt.month
                monthly_data = df_filtered.groupby(['year', 'month'])['value'].mean().reset_index()
                
                # Create chart
                fig = self.create_monthly_chart(monthly_data, currency, years)
                
                # Create report
                report = self.create_analysis_report(monthly_data, currency, years)
                
                return fig, report
                
            except Exception as e:
                return self.create_empty_chart(), f"Error: {str(e)}"
    
    def create_monthly_chart(self, monthly_data, currency, years):
        """Create the main monthly movement chart"""
        fig = go.Figure()
        
        # Color palette for different years
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, year in enumerate(sorted(years)):
            year_data = monthly_data[monthly_data['year'] == year]
            
            # Sort by month
            year_data = year_data.sort_values('month')
            
            # Month names for x-axis
            month_names = [datetime(2024, month, 1).strftime('%b') for month in year_data['month']]
            
            fig.add_trace(go.Scatter(
                x=month_names,
                y=year_data['value'],
                mode='lines+markers',
                name=f'{year}',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{year}</b><br>Month: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Monthly Currency Movement - {currency} ({', '.join(map(str, sorted(years)))})",
            xaxis_title="Month",
            yaxis_title=f"Exchange Rate ({currency})",
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_analysis_report(self, monthly_data, currency, years):
        """Create detailed analysis report"""
        if monthly_data.empty:
            return "No data available for the selected criteria."
        
        # Calculate statistics
        stats = self.calculate_monthly_statistics(monthly_data)
        
        # Find intersection points
        intersections = self.find_intersection_points(monthly_data)
        
        # Find peak months
        peak_months = self.find_peak_months(monthly_data)
        
        # Create report
        report = []
        
        # Summary
        report.append(html.H4("📊 Summary", style={'color': '#2c3e50', 'marginTop': 0}))
        report.append(html.P(f"Analysis of {currency} exchange rates across {len(years)} year(s): {', '.join(map(str, sorted(years)))}"))
        
        # Add note about currency conversion
        if any(year < 2018 for year in years):
            report.append(html.P("📝 Note: Exchange rates before January 2, 2018 have been adjusted (divided by 10) to account for currency redenomination.", 
                                style={'fontStyle': 'italic', 'color': '#7f8c8d'}))
        
        # Peak Analysis
        report.append(html.H4("🏆 Peak Month Analysis", style={'color': '#2c3e50'}))
        for year in sorted(years):
            year_peaks = peak_months.get(year, [])
            if year_peaks:
                peak_month = year_peaks[0]
                month_name = datetime(2024, peak_month['month'], 1).strftime('%B')
                report.append(html.P(f"• {year}: {month_name} (Value: {peak_month['value']:.2f})"))
        
        # Monthly Rankings
        report.append(html.H4("📈 Monthly Performance Rankings", style={'color': '#2c3e50'}))
        for month in range(1, 13):
            month_name = datetime(2024, month, 1).strftime('%B')
            month_data = monthly_data[monthly_data['month'] == month]
            if not month_data.empty:
                avg_value = month_data['value'].mean()
                report.append(html.P(f"• {month_name}: Average {avg_value:.2f}"))
        
        # Intersection Points
        if intersections:
            report.append(html.H4("🎯 Intersection Points", style={'color': '#2c3e50'}))
            report.append(html.P("Months where different years show similar values:"))
            for intersection in intersections:
                month_name = datetime(2024, intersection['month'], 1).strftime('%B')
                report.append(html.P(f"• {month_name}: Values around {intersection['value']:.2f}"))
        
        # Pattern Analysis
        report.append(html.H4("🔍 Pattern Analysis", style={'color': '#2c3e50'}))
        
        # Find most consistent peak month
        all_peaks = []
        for year_peaks in peak_months.values():
            all_peaks.extend([peak['month'] for peak in year_peaks])
        
        if all_peaks:
            from collections import Counter
            peak_counts = Counter(all_peaks)
            most_common_peak = peak_counts.most_common(1)[0]
            month_name = datetime(2024, most_common_peak[0], 1).strftime('%B')
            report.append(html.P(f"• Most consistent peak month: {month_name} ({most_common_peak[1]} out of {len(years)} years)"))
        
        # Volatility analysis
        monthly_std = monthly_data.groupby('month')['value'].std()
        most_volatile_month = monthly_std.idxmax()
        least_volatile_month = monthly_std.idxmin()
        
        most_volatile_name = datetime(2024, most_volatile_month, 1).strftime('%B')
        least_volatile_name = datetime(2024, least_volatile_month, 1).strftime('%B')
        
        report.append(html.P(f"• Most volatile month: {most_volatile_name} (std: {monthly_std[most_volatile_month]:.2f})"))
        report.append(html.P(f"• Most stable month: {least_volatile_name} (std: {monthly_std[least_volatile_month]:.2f})"))
        
        return html.Div(report)
    
    def calculate_monthly_statistics(self, monthly_data):
        """Calculate monthly statistics"""
        stats = {}
        
        # Overall statistics
        stats['total_years'] = monthly_data['year'].nunique()
        stats['total_months'] = monthly_data['month'].nunique()
        stats['overall_mean'] = monthly_data['value'].mean()
        stats['overall_std'] = monthly_data['value'].std()
        
        # Monthly averages
        monthly_avg = monthly_data.groupby('month')['value'].mean()
        stats['monthly_averages'] = monthly_avg.to_dict()
        
        return stats
    
    def find_intersection_points(self, monthly_data):
        """Find months where different years have similar values"""
        intersections = []
        
        for month in range(1, 13):
            month_data = monthly_data[monthly_data['month'] == month]
            if len(month_data) > 1:
                values = month_data['value'].values
                # Check if values are close (within 2% of mean)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val / mean_val < 0.02:  # Less than 2% variation
                    intersections.append({
                        'month': month,
                        'value': mean_val,
                        'years': month_data['year'].tolist()
                    })
        
        return intersections
    
    def find_peak_months(self, monthly_data):
        """Find peak months for each year"""
        peak_months = {}
        
        for year in monthly_data['year'].unique():
            year_data = monthly_data[monthly_data['year'] == year]
            if not year_data.empty:
                # Find the month with highest value
                peak_idx = year_data['value'].idxmax()
                peak_month = year_data.loc[peak_idx]
                
                peak_months[year] = [{
                    'month': int(peak_month['month']),
                    'value': peak_month['value']
                }]
        
        return peak_months
    
    def create_empty_chart(self):
        """Create empty chart"""
        fig = go.Figure()
        fig.add_annotation(
            text="Select a currency and year(s) to view the analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="Currency Analysis Chart",
            xaxis_title="Month",
            yaxis_title="Exchange Rate",
            template='plotly_white'
        )
        return fig
    
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = SimpleCurrencyDashboard("Simple Currency Analysis Dashboard")
    dashboard.run()
