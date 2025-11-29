#!/usr/bin/env python3
"""
Optuna Dashboard Launcher for EmptyDrops Optimization
====================================================

This script launches the Optuna Dashboard web interface to monitor and analyze
EmptyDrops hyperparameter optimization studies in real-time.

Features:
- Real-time monitoring of optimization progress
- Interactive visualizations and parameter analysis
- Multi-study comparison capabilities
- Study management (pause, resume, delete)
- Web-based interface accessible from any browser

Usage:
    python3 launch_optuna_dashboard.py [--port PORT] [--host HOST] [--auto-open]
"""

import argparse
import glob
import os
import sys
import webbrowser
import time
from pathlib import Path
from datetime import datetime

def find_study_databases():
    """Find all Optuna study database files."""
    study_dir = os.path.join("optuna_optimization", "studies")
    
    if not os.path.exists(study_dir):
        print(f"❌ Study directory not found: {study_dir}")
        return []
    
    # Find all .db files
    db_files = glob.glob(os.path.join(study_dir, "*.db"))
    
    if not db_files:
        print(f"❌ No study databases found in {study_dir}")
        return []

    # Sort by modification time (newest first)
    db_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"📊 Found {len(db_files)} study databases (sorted by date, newest first):")
    for i, db_file in enumerate(db_files):
        basename = os.path.basename(db_file)
        size_mb = os.path.getsize(db_file) / (1024 * 1024)
        
        # Get modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(db_file))
        date_str = mod_time.strftime("%Y-%m-%d %H:%M")
        
        # Try to get trial count from database
        try:
            import sqlite3
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trials")
            trial_count = cursor.fetchone()[0]
            conn.close()
            trial_info = f"{trial_count} trials"
        except:
            trial_info = "? trials"
        
        # Parse study type
        if 'multi-objective' in basename or 'multiobj' in basename:
            study_type = "Multi-obj"
        elif 'single-objective' in basename:
            study_type = "Single-obj"
        else:
            study_type = "Standard"
        
        print(f"  {i+1:2d}. {basename}")
        print(f"      📅 {date_str} | 🔬 {study_type} | 📊 {trial_info} | 💾 {size_mb:.1f} MB")
    print()

    return db_files

def create_storage_urls(db_files):
    """Create SQLite storage URLs for the dashboard."""
    storage_urls = []
    
    for db_file in db_files:
        # Convert to absolute path
        abs_path = os.path.abspath(db_file)
        storage_url = f"sqlite:///{abs_path}"
        storage_urls.append(storage_url)
    
    return storage_urls

def select_study(storage_urls, use_latest=False):
    """Allow user to select which study to view in dashboard."""
    if len(storage_urls) == 1:
        return storage_urls[0]  # Return only study
    
    if use_latest:
        return storage_urls[0]  # Return latest study (first in sorted list)
    
    print(f"\n📚 Multiple studies found. Choose one to view in dashboard:")
    print("   💡 Note: Optuna Dashboard can only show one study at a time")
    print("   🔄 You can restart the launcher to switch between studies")
    print()
    
    while True:
        try:
            choice = input(f"Select study (1-{len(storage_urls)}) or press Enter for latest: ").strip()
            
            if choice == '' or choice.lower() in ['latest', 'l']:
                return storage_urls[0]  # Most recent (first in list)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(storage_urls):
                return storage_urls[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(storage_urls)}")
        except ValueError:
            print("❌ Please enter a valid number or press Enter for latest")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def launch_dashboard(storage_urls, host="127.0.0.1", port=8080, auto_open=True, use_latest=False):
    """Launch the Optuna Dashboard."""
    
    print(f"\n🚀 Launching Optuna Dashboard...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Available studies: {len(storage_urls)}")
    
    try:
        import optuna_dashboard
        
        # Select which study to view
        selected_storage = select_study(storage_urls, use_latest)
        selected_name = os.path.basename(selected_storage.replace('sqlite:///', ''))
        
        # Create dashboard URL
        dashboard_url = f"http://{host}:{port}"
        
        print(f"\n📈 Dashboard will be available at: {dashboard_url}")
        print(f"   Viewing study: {selected_name}")
        print(f"   Press Ctrl+C to stop the dashboard")
        
        if auto_open:
            print(f"🌐 Opening browser in 3 seconds...")
            time.sleep(3)
            webbrowser.open(dashboard_url)
        
        # Launch dashboard with selected storage
        optuna_dashboard.run_server(
            storage=selected_storage,
            host=host,
            port=port
        )
        
    except ImportError:
        print("❌ optuna-dashboard not installed. Install with:")
        print("   pip install optuna-dashboard")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

def display_dashboard_features():
    """Display information about dashboard features."""
    print("""
🎯 OPTUNA DASHBOARD FEATURES:
============================

📊 Real-time Monitoring:
   • Live optimization progress tracking
   • Real-time trial updates and completion status
   • Performance metrics visualization

📈 Interactive Visualizations:
   • Parameter importance plots
   • Optimization history graphs
   • Parallel coordinate plots with real-time updates
   • Hyperparameter relationships analysis

🔍 Study Analysis:
   • Compare multiple optimization studies
   • Best trial identification and analysis
   • Parameter distribution analysis
   • Convergence tracking

🛠️ Study Management:
   • View detailed trial information
   • Filter and search trials
   • Export study results
   • Delete completed studies

💡 Navigation Tips:
   • Use the sidebar to switch between studies
   • Click on trials for detailed parameter information
   • Use the "Analytics" tab for advanced visualizations
   • Check "Optimization History" for progress tracking

🌐 Access from anywhere:
   • Dashboard runs on local web server
   • Access from any browser on your network
   • Share URL with team members for collaboration
""")

def main():
    parser = argparse.ArgumentParser(
        description="Launch Optuna Dashboard for EmptyDrops optimization monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 launch_optuna_dashboard.py                    # Launch with defaults
  python3 launch_optuna_dashboard.py --port 8888       # Custom port
  python3 launch_optuna_dashboard.py --no-auto-open    # Don't open browser
  python3 launch_optuna_dashboard.py --info            # Show features info
        """
    )
    
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for the dashboard server (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1",
                       help="Host for the dashboard server (default: 127.0.0.1)")
    parser.add_argument("--no-auto-open", action="store_true",
                       help="Don't automatically open browser")
    parser.add_argument("--info", action="store_true",
                       help="Show dashboard features information and exit")
    parser.add_argument("--latest", action="store_true",
                       help="Automatically use the most recent study (no selection prompt)")
    
    args = parser.parse_args()
    
    print("🧬 EmptyDrops Optuna Dashboard Launcher")
    print("=" * 50)
    
    if args.info:
        display_dashboard_features()
        return
    
    # Find study databases
    db_files = find_study_databases()
    
    if not db_files:
        print("\n💡 To create studies, run:")
        print("   python3 hyperparameter_optimization.py")
        print("   python3 test_fast_optimization.py")
        return
    
    # Create storage URLs
    storage_urls = create_storage_urls(db_files)
    
    # Launch dashboard
    auto_open = not args.no_auto_open
    launch_dashboard(storage_urls, args.host, args.port, auto_open, args.latest)

if __name__ == "__main__":
    main() 