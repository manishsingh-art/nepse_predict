import subprocess
import time
import sys
import os

def run_app():
    print("🚀 Starting NEPSE Predictor v5.0 Web App...")
    
    # 1. Start Backend
    print("📦 Starting FastAPI Backend (Port 8000)...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 2. Wait for backend to initialize
    time.sleep(2)
    
    # 3. Start Frontend
    print("🎨 Starting React Frontend (Port 5173)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="./frontend",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    print("\n✅ Servers are running!")
    print("🔗 Frontend: http://localhost:5173")
    print("🔗 Backend:  http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop both servers.")
    
    try:
        while True:
            # Optionally print logs or just wait
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    run_app()
