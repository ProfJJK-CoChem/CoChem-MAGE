# %%
import os
import json
import ipywidgets as widgets
from IPython.display import display, clear_output

class UnityInstallerDashboard:
    """
    Batch 4: Ecosystem Bootstrapping for CoChem-UNITY.
    Provides an interactive widget dashboard to select pipeline modules,
    now featuring the CoChem-MAGE GC-MS emulation subsystem.
    """
    def __init__(self, manifest_path="cochem_deployment_manifest.json"):
        self.manifest_path = manifest_path
        self.modules = self._define_modules()
        self.checkboxes = {}
        
    def _define_modules(self):
        """Defines the available CoChem subsystems and their repository locators."""
        return {
            "CORE": {"name": "CoChem-CORE", "desc": "Mandatory. Base orchestrator, registry, and environment silo generator.", "req": True, "repo": "ProfJJK/CoChem-CORE"},
            "MINT": {"name": "CoChem-MInt", "desc": "Mandatory. Unified GUI ingestion and geometry validation.", "req": True, "repo": "ProfJJK/CoChem-MInt"},
            "TOPOS": {"name": "CoChem-TOPOS", "desc": "Symmetry assignment and Eckart frame alignment.", "req": False, "repo": "ProfJJK/CoChem-TOPOS"},
            "TORQ": {"name": "CoChem-TORQ", "desc": "Torsional discovery, deduplication, and ML-PES generation.", "req": False, "repo": "ProfJJK/CoChem-TORQ"},
            "MAGE": {"name": "CoChem-MAGE", "desc": "NEW: GC-MS Emulator. Predicts Retention Indices and simulates EI fragmentation.", "req": False, "repo": "ProfJJK/CoChem-MAGE"},
            "SCRIBE": {"name": "CoChem-SCRIBE", "desc": "Optional. Local LLM agent for automated narrative reporting. (Requires high RAM/VRAM).", "req": False, "repo": "ProfJJK/CoChem-SCRIBE"}
        }

    def render_ui(self):
        """Builds and displays the interactive Jupyter dashboard."""
        print("🧪 CoChem-UNITY: Pipeline Deployment Forge\n" + "="*45)
        
        ui_elements = []
        for key, mod in self.modules.items():
            cb = widgets.Checkbox(
                value=mod["req"], 
                description=f"{mod['name']}",
                disabled=mod["req"], # Lock mandatory modules
                tooltip=mod["repo"]
            )
            label = widgets.Label(value=mod["desc"])
            row = widgets.HBox([cb, label])
            self.checkboxes[key] = cb
            ui_elements.append(row)
            
        self.deploy_btn = widgets.Button(
            description="Generate Manifest & Deploy",
            button_style="success",
            icon="rocket"
        )
        self.output_log = widgets.Output()
        
        self.deploy_btn.on_click(self._on_deploy_clicked)
        
        dashboard = widgets.VBox(ui_elements + [widgets.HTML("<hr>"), self.deploy_btn, self.output_log])
        display(dashboard)

    def _on_deploy_clicked(self, b):
        """Callback to write the deployment JSON when the user locks their selection."""
        with self.output_log:
            clear_output()
            
            selected_modules = []
            for key, cb in self.checkboxes.items():
                if cb.value:
                    selected_modules.append({
                        "id": key,
                        "name": self.modules[key]["name"],
                        "repo_target": self.modules[key]["repo"]
                    })
                    
            manifest = {
                "version": "2.0",
                "orchestrator_directive": "MAGE_INTEGRATED",
                "selected_modules": selected_modules
            }
            
            try:
                with open(self.manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=4)
                print(f"✅ Success! Deployment manifest locked with {len(selected_modules)} modules.")
                print(f"📄 Saved to: {os.path.abspath(self.manifest_path)}")
                print("➡️ Next Step: Run the Stage 0.0 notebook cell to initiate silo construction.")
            except Exception as e:
                print(f"❌ Critical Fault writing manifest: {e}")

# Execute UI Test
if __name__ == "__main__":
    installer = UnityInstallerDashboard()
    installer.render_ui()
# %%