"""
stop_all.py - Stop all Preventra services
"""

import subprocess
import sys
import platform
import signal
import os

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              PREVENTRA - SHUTDOWN MANAGER                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

def stop_windows():
    """Stop services on Windows"""
    print("\n🔍 Searching for Preventra processes on Windows...")
    
    # Kill Streamlit
    try:
        subprocess.run('taskkill /F /IM streamlit.exe', shell=True, capture_output=True)
        print("   ✓ Streamlit stopped")
    except:
        pass
    
    # Kill Python processes running our scripts
    try:
        result = subprocess.run('tasklist', shell=True, capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        for line in lines:
            if 'python' in line.lower():
                # Extract PID and check if it's our process
                parts = line.split()
                if len(parts) > 1:
                    try:
                        subprocess.run(f'taskkill /F /PID {parts[1]}', shell=True, capture_output=True)
                    except:
                        pass
        print("   ✓ Python processes stopped")
    except:
        pass
    
    # Kill Uvicorn (FastAPI)
    try:
        subprocess.run('taskkill /F /IM uvicorn.exe', shell=True, capture_output=True)
        print("   ✓ API backend stopped")
    except:
        pass

def stop_unix():
    """Stop services on Unix/Linux/Mac"""
    print("\n🔍 Searching for Preventra processes on Unix/Linux/Mac...")
    
    # Kill Streamlit
    try:
        subprocess.run("pkill -f 'streamlit run'", shell=True)
        print("   ✓ Streamlit stopped")
    except:
        pass
    
    # Kill Backend
    try:
        subprocess.run("pkill -f 'backend.py'", shell=True)
        print("   ✓ API backend stopped")
    except:
        pass
    
    # Kill Uvicorn
    try:
        subprocess.run("pkill -f 'uvicorn'", shell=True)
        print("   ✓ Uvicorn stopped")
    except:
        pass

def main():
    print_banner()
    
    system = platform.system()
    
    print(f"💻 Detected OS: {system}")
    
    if system == "Windows":
        stop_windows()
    else:
        stop_unix()
    
    print("\n✅ All Preventra services stopped!")
    print("\n💡 To restart, run: python run_all.py")

if __name__ == "__main__":
    main()