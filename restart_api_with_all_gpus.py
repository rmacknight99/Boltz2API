#!/usr/bin/env python3
"""
Restart the API with proper CUDA environment set from the beginning.
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

def kill_existing_api():
    """Kill any existing Flask API processes."""
    try:
        # Find Flask processes
        result = subprocess.run([
            "pgrep", "-f", "app.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    print(f"🔪 Killing existing API process {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
                    
        time.sleep(2)  # Give processes time to die
        print("✅ Existing processes cleaned up")
        
    except Exception as e:
        print(f"⚠️  Error cleaning up processes: {e}")

def start_api_with_proper_env():
    """Start the API with CUDA_VISIBLE_DEVICES set from the beginning."""
    
    # Set up environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # All GPUs visible from start
    env["PYTHONUNBUFFERED"] = "1"  # For real-time output
    
    print("🚀 Starting Flask API with CUDA_VISIBLE_DEVICES=0,1,2,3")
    print("This should prevent CUDA context switching errors")
    
    # Start the API
    try:
        subprocess.run([
            sys.executable, "app.py"
        ], env=env)
    except KeyboardInterrupt:
        print("\n🛑 API stopped by user")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

def main():
    print("🔧 Restarting Boltz API with Proper CUDA Environment")
    print("=" * 60)
    
    # Step 1: Kill existing processes
    kill_existing_api()
    
    # Step 2: Start with proper environment
    start_api_with_proper_env()

if __name__ == "__main__":
    main() 