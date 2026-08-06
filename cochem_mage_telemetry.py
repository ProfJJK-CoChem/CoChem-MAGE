# %%
import asyncio
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import threading
import time

class MageTelemetryBridge:
    """
    Segment 3B: Asynchronous Telemetry Bridge for CoChem-MAGE.
    Provides a non-blocking FastAPI backend that allows CoChem-DOCK
    to stream real-time progress of GC-MS simulations.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8055):
        self.host = host
        self.port = port
        self.app = FastAPI(title="CoChem-MAGE Telemetry API", version="1.0")
        
        # Configure CORS for React frontend (CoChem-DOCK)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], # In production, restrict to localhost / DOCK ports
            allow_credentials=True,
            allow_methods=["GET"],
            allow_headers=["*"],
        )
        
        # Thread-safe state dictionary
        self._lock = threading.Lock()
        self.current_state: Dict[str, Any] = {
            "status": "IDLE",
            "progress_percent": 0.0,
            "current_operation": "Awaiting batch submission...",
            "active_isomer": None,
            "optimization_rs": None,
            "error_flag": None
        }
        
        self._setup_routes()

    def _setup_routes(self):
        """Maps the HTTP GET routes for the DOCK interface."""
        @self.app.get("/mage/status")
        async def get_status():
            with self._lock:
                return self.current_state

        @self.app.get("/mage/health")
        async def health_check():
            return {"service": "CoChem-MAGE Telemetry", "status": "ONLINE"}

    def update_state(self, status: str, progress: float, operation: str, 
                     isomer: Optional[str] = None, rs: Optional[float] = None, 
                     error: Optional[str] = None):
        """Thread-safe method for the MAGE engine to report its current progress."""
        with self._lock:
            self.current_state["status"] = status
            self.current_state["progress_percent"] = round(progress, 2)
            self.current_state["current_operation"] = operation
            if isomer is not None:
                self.current_state["active_isomer"] = isomer
            if rs is not None:
                self.current_state["optimization_rs"] = rs
            if error is not None:
                self.current_state["error_flag"] = error

    def launch_background_server(self):
        """Spawns the Uvicorn ASGI server in a daemon thread to prevent kernel blocking."""
        print(f"📡 Booting MAGE Telemetry Bridge on http://{self.host}:{self.port}...")
        
        def run_server():
            # Disable Uvicorn access logs to prevent Jupyter cell spam
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="critical")
            
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1) # Allow port binding

# Execute Telemetry Test
if __name__ == "__main__":
    telemetry = MageTelemetryBridge()
    telemetry.launch_background_server()
    
    # Simulate a MAGE pipeline run reporting back to the UI
    print("\\n🧪 Simulating MAGE Pipeline Execution...")
    telemetry.update_state("RUNNING", 10.0, "Ingesting hardware profile and SMILES...")
    time.sleep(1.5)
    
    telemetry.update_state("RUNNING", 45.0, "Computing RRKM Fragmentation (GPU)...", isomer="Aspirin")
    time.sleep(2)
    
    telemetry.update_state("OPTIMIZING", 80.0, "Van Deemter flow optimization active...", rs=0.85)
    time.sleep(2)
    
    telemetry.update_state("COMPLETE", 100.0, "Chromatogram compiled and SCRIBE payload saved.", rs=1.62)
    print("✅ Pipeline complete. Telemetry state reads:")
    
    # Read back the final state
    import requests
    try:
        response = requests.get("http://127.0.0.1:8055/mage/status")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to query local telemetry: {e}")
# %%