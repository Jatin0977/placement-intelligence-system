import sys
import os
import runpy

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Execute placement_intel application explicitly on every Streamlit rerun cycle
target_script = os.path.join(os.path.dirname(__file__), "placement_intel", "app.py")
runpy.run_path(target_script, run_name="__main__")

