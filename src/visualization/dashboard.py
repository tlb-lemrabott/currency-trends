"""
Basic Visualization Dashboard

This module creates a simple dashboard to visualize currency trends analysis results.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CurrencyDashboard:
    """Basic dashboard for currency trends visualization"""
    
    def __init__(self, title: str = "Currency Trends Analysis Dashboard"):
        """
        Initialize the dashboard
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Controls", className="card-title"),
                            html.Label("Select Currency:"),
                            dcc.Dropdown(
                                id='currency-dropdown',
                                options=[
                                    {'label': 'USD', 'value': 'USD'},
                                    {'label': 'EUR', 'value': 'EUR'},
                                    {'label': 'All Currencies', 'value': 'all'}
                                ],
                                value='USD',
                                className="mb-3"
                            ),
                            html.Label("Analysis Type:"),
                            dcc.Dropdown(
                                id='analysis-dropdown',
                                options=[
                                    {'label': 'Price Trends', 'value': 'trends'},
                                    {'label': 'Moving Averages', 'value': 'ma'},
                                    {'label': 'Volatility Analysis', 'value': 'volatility'},
                                    {'label': 'Correlation Matrix', 'value': 'correlation'},
                                    {'label': 'Seasonal Predictions', 'value': 'seasonal_predictions'}
                                ],
                                value='trends',
                                className="mb-3"
                            ),
                            dbc.Button("Update Analysis", id="update-btn", color="primary", className="w-100")
                        ])
                    ])
                ], width=3),
                
                # Main content area
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='main-chart', style={'height': '500px'})
                        ])
                    ])
                ], width=9)
            ]),
            
            # Summary statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Summary Statistics", className="card-title"),
                            html.Div(id='summary-stats')
                        ])
                    ])
                ])
            ], className="mt-4"),
            
            # Additional charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Trend Analysis", className="card-title"),
                            dcc.Graph(id='trend-chart', style={'height': '400px'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Volatility Analysis", className="card-title"),
                            dcc.Graph(id='volatility-chart', style={'height': '400px'})
                        ])
                    ])
                ], width=6)
            ], className="mt-4"),
            
            # Hidden div for storing data
            html.Div(id='data-store', style={'display': 'none'})
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('main-chart', 'figure'),
             Output('trend-chart', 'figure'),
             Output('volatility-chart', 'figure'),
             Output('summary-stats', 'children')],
            [Input('update-btn', 'n_clicks'),
             Input('currency-dropdown', 'value'),
             Input('analysis-dropdown', 'value')]
        )
        def update_charts(n_clicks, currency, analysis_type):
            """Update charts based on user selection"""
            
            # Load sample data for demonstration
            from src.data.schema import DataValidator
            from src.data.preprocessing import DataPreprocessor
            from src.analysis.trends import TrendAnalyzer
            
            try:
                # Load and preprocess data
                currency_data = DataValidator.load_and_validate_from_file('data/sample_currency_data.json')
                preprocessor = DataPreprocessor()
                df = preprocessor.convert_to_dataframe(currency_data)
                
                # Remove duplicate entries to prevent reshape errors
                df = df.drop_duplicates(subset=['day', 'currency_code']).reset_index(drop=True)
                
                # Add some technical indicators
                df = preprocessor.add_technical_indicators(df)
                df = preprocessor.add_time_features(df)
                
                # For seasonal predictions, we need extended data
                if analysis_type == 'seasonal_predictions':
                    # Create extended sample data for demonstration
                    from datetime import datetime
                    start_date = datetime(2019, 1, 1)
                    end_date = datetime(2024, 12, 31)
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    extended_data = []
                    for date in dates:
                        month = date.month
                        year_factor = (date.year - 2019) * 0.5
                        seasonal_component = 10 * np.sin(2 * np.pi * month / 12) + 5 * np.sin(2 * np.pi * month / 6)
                        noise = np.random.normal(0, 2)
                        usd_value = 100 + year_factor + seasonal_component + noise
                        
                        extended_data.append({
                            'day': date,
                            'currency_code': 'USD',
                            'value': usd_value
                        })
                    
                    df = pd.DataFrame(extended_data)
                
                # Filter by currency if specified
                if currency and currency != 'all':
                    df = df[df['currency_code'] == currency]
                
                # Create main chart based on analysis type
                main_fig = self.create_main_chart(df, analysis_type, currency)
                trend_fig = self.create_trend_chart(df, currency)
                volatility_fig = self.create_volatility_chart(df, currency)
                summary_stats = self.create_summary_stats(df, currency)
                
                return main_fig, trend_fig, volatility_fig, summary_stats
                
            except Exception as e:
                logger.error(f"Error updating charts: {e}")
                # Return empty charts with error message
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Error loading data: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return error_fig, error_fig, error_fig, html.P(f"Error: {str(e)}")
    
    def create_main_chart(self, df: pd.DataFrame, analysis_type: str, currency: str) -> go.Figure:
        """Create main chart based on analysis type"""
        
        if analysis_type == 'trends':
            return self.create_trend_chart(df, currency)
        elif analysis_type == 'ma':
            return self.create_moving_averages_chart(df, currency)
        elif analysis_type == 'volatility':
            return self.create_volatility_chart(df, currency)
        elif analysis_type == 'correlation':
            return self.create_correlation_chart(df)
        elif analysis_type == 'seasonal_predictions':
            return self.create_seasonal_predictions_chart(df, currency)
        else:
            return self.create_price_chart(df, currency)
    
    def create_price_chart(self, df: pd.DataFrame, currency: str) -> go.Figure:
        """Create basic price chart"""
        fig = go.Figure()
        
        for curr in df['currency_code'].unique():
            curr_data = df[df['currency_code'] == curr]
            fig.add_trace(go.Scatter(
                x=curr_data['day'],
                y=curr_data['value'],
                mode='lines+markers',
                name=f'{curr} Price',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=f"Currency Exchange Rates {'(' + currency + ')' if currency != 'all' else ''}",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_trend_chart(self, df: pd.DataFrame, currency: str) -> go.Figure:
        """Create trend analysis chart"""
        fig = go.Figure()
        
        for curr in df['currency_code'].unique():
            curr_data = df[df['currency_code'] == curr].sort_values('day')
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=curr_data['day'],
                y=curr_data['value'],
                mode='lines+markers',
                name=f'{curr} Actual',
                line=dict(width=2)
            ))
            
            # Add moving averages if available
            if 'sma_7' in curr_data.columns:
                fig.add_trace(go.Scatter(
                    x=curr_data['day'],
                    y=curr_data['sma_7'],
                    mode='lines',
                    name=f'{curr} SMA-7',
                    line=dict(dash='dash', width=1)
                ))
            
            if 'ema_12' in curr_data.columns:
                fig.add_trace(go.Scatter(
                    x=curr_data['day'],
                    y=curr_data['ema_12'],
                    mode='lines',
                    name=f'{curr} EMA-12',
                    line=dict(dash='dot', width=1)
                ))
        
        fig.update_layout(
            title=f"Trend Analysis {'(' + currency + ')' if currency != 'all' else ''}",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_moving_averages_chart(self, df: pd.DataFrame, currency: str) -> go.Figure:
        """Create moving averages chart"""
        fig = go.Figure()
        
        for curr in df['currency_code'].unique():
            curr_data = df[df['currency_code'] == curr].sort_values('day')
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=curr_data['day'],
                y=curr_data['value'],
                mode='lines',
                name=f'{curr} Price',
                line=dict(width=2)
            ))
            
            # Add different moving averages
            if 'sma_7' in curr_data.columns:
                fig.add_trace(go.Scatter(
                    x=curr_data['day'],
                    y=curr_data['sma_7'],
                    mode='lines',
                    name=f'{curr} SMA-7',
                    line=dict(dash='dash')
                ))
            
            if 'sma_30' in curr_data.columns:
                fig.add_trace(go.Scatter(
                    x=curr_data['day'],
                    y=curr_data['sma_30'],
                    mode='lines',
                    name=f'{curr} SMA-30',
                    line=dict(dash='dot')
                ))
        
        fig.update_layout(
            title=f"Moving Averages {'(' + currency + ')' if currency != 'all' else ''}",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_volatility_chart(self, df: pd.DataFrame, currency: str) -> go.Figure:
        """Create volatility analysis chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Changes', 'Volatility'),
            vertical_spacing=0.1
        )
        
        for curr in df['currency_code'].unique():
            curr_data = df[df['currency_code'] == curr].sort_values('day')
            
            # Price changes
            if 'price_change_pct' in curr_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=curr_data['day'],
                        y=curr_data['price_change_pct'] * 100,
                        mode='lines+markers',
                        name=f'{curr} % Change',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
            
            # Volatility
            if 'volatility_7' in curr_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=curr_data['day'],
                        y=curr_data['volatility_7'],
                        mode='lines',
                        name=f'{curr} Volatility-7',
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            title=f"Volatility Analysis {'(' + currency + ')' if currency != 'all' else ''}",
            height=600,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="% Change", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_correlation_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix chart"""
        # Calculate correlation matrix
        pivot_df = df.pivot(index='day', columns='currency_code', values='value')
        corr_matrix = pivot_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Currency Correlation Matrix",
            template='plotly_white'
        )
        
        return fig
    
    def create_seasonal_predictions_chart(self, df: pd.DataFrame, currency: str) -> go.Figure:
        """Create seasonal predictions chart"""
        from src.analysis.seasonality import SeasonalityAnalyzer
        
        # Initialize seasonality analyzer
        analyzer = SeasonalityAnalyzer()
        
        # Get seasonal predictions
        predictions = analyzer.predict_seasonal_peaks(df, currency_code=currency, forecast_years=2)
        historical_peaks = analyzer.analyze_historical_seasonal_peaks(df, currency_code=currency)
        
        # Create subplot with historical data and predictions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Historical Seasonal Peaks', 'Monthly Performance', 
                          'Predicted Peaks', 'Confidence Factors'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Historical Seasonal Peaks
        if 'yearly_peaks' in historical_peaks:
            years = list(historical_peaks['yearly_peaks'].keys())
            peak_months = [historical_peaks['yearly_peaks'][year]['peak_month'] for year in years]
            peak_values = [historical_peaks['yearly_peaks'][year]['peak_value'] for year in years]
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=peak_months,
                    mode='lines+markers',
                    name='Peak Month',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=peak_values,
                    mode='lines+markers',
                    name='Peak Value',
                    line=dict(color='red', width=2),
                    marker=dict(size=6),
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # 2. Monthly Performance
        if 'monthly_performance' in historical_peaks:
            months = list(historical_peaks['monthly_performance'].keys())
            avg_values = [historical_peaks['monthly_performance'][month]['average_value'] for month in months]
            month_names = [datetime(2024, month, 1).strftime('%b') for month in months]
            
            fig.add_trace(
                go.Bar(
                    x=month_names,
                    y=avg_values,
                    name='Avg Monthly Value',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Predicted Peaks
        if 'predictions' in predictions:
            for year, year_data in predictions['predictions'].items():
                months = [peak['month'] for peak in year_data['predicted_peaks']]
                values = [peak['predicted_value'] for peak in year_data['predicted_peaks']]
                confidences = [peak['confidence'] for peak in year_data['predicted_peaks']]
                month_names = [datetime(2024, month, 1).strftime('%b') for month in months]
                
                fig.add_trace(
                    go.Scatter(
                        x=month_names,
                        y=values,
                        mode='lines+markers',
                        name=f'Predicted {year}',
                        line=dict(width=2),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
        
        # 4. Confidence Factors
        if 'confidence_factors' in predictions:
            confidence = predictions['confidence_factors']
            factors = list(confidence.keys())
            values = list(confidence.values())
            
            fig.add_trace(
                go.Bar(
                    x=factors,
                    y=values,
                    name='Confidence',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Seasonal Predictions Analysis {'(' + currency + ')' if currency != 'all' else ''}",
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Peak Month", row=1, col=1)
        fig.update_yaxes(title_text="Peak Value", row=1, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Average Value", row=1, col=2)
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Value", row=2, col=1)
        
        fig.update_xaxes(title_text="Factor", row=2, col=2)
        fig.update_yaxes(title_text="Confidence Score", row=2, col=2)
        
        return fig
    
    def create_summary_stats(self, df: pd.DataFrame, currency: str) -> html.Div:
        """Create summary statistics display"""
        
        stats = []
        
        for curr in df['currency_code'].unique():
            curr_data = df[df['currency_code'] == curr]
            
            curr_stats = {
                'currency': curr,
                'count': len(curr_data),
                'min': curr_data['value'].min(),
                'max': curr_data['value'].max(),
                'mean': curr_data['value'].mean(),
                'std': curr_data['value'].std(),
                'total_change': ((curr_data['value'].iloc[-1] - curr_data['value'].iloc[0]) / curr_data['value'].iloc[0]) * 100 if len(curr_data) > 1 else 0
            }
            stats.append(curr_stats)
        
        # Create summary cards
        cards = []
        for stat in stats:
            card = dbc.Card([
                dbc.CardBody([
                    html.H5(f"{stat['currency']}", className="card-title"),
                    html.P(f"Data Points: {stat['count']}"),
                    html.P(f"Min: {stat['min']:.2f}"),
                    html.P(f"Max: {stat['max']:.2f}"),
                    html.P(f"Mean: {stat['mean']:.2f}"),
                    html.P(f"Std: {stat['std']:.2f}"),
                    html.P(f"Total Change: {stat['total_change']:.2f}%", 
                           style={'color': 'green' if stat['total_change'] > 0 else 'red'})
                ])
            ], className="mb-3")
            cards.append(card)
        
        return html.Div(cards)
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard"""
        logger.info(f"Starting dashboard on port {port}")
        self.app.run(debug=debug, port=port)


def create_sample_dashboard():
    """Create and run a sample dashboard"""
    dashboard = CurrencyDashboard("Currency Trends Analysis Dashboard")
    return dashboard


if __name__ == "__main__":
    dashboard = create_sample_dashboard()
    dashboard.run()
