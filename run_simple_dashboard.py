#!/usr/bin/env python3
"""
Simple Dashboard Runner

This script runs the simplified currency analysis dashboard.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.visualization.simple_dashboard import SimpleCurrencyDashboard


def main():
    """Run the simple dashboard"""
    print("=== Simple Currency Analysis Dashboard ===")
    print("Starting simplified dashboard...")
    print("Dashboard will be available at: http://localhost:8051")
    print("Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        # Create and run dashboard
        dashboard = SimpleCurrencyDashboard("Simple Currency Analysis Dashboard")
        dashboard.run(debug=True, port=8051)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install dash dash-bootstrap-components plotly")


if __name__ == "__main__":
    main()
