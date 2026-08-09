# %%
import os
import json
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

class MageExporter:
    """
    Stage 3.0: Visualization and Narrative Handoff.
    Generates interactive HTML chromatograms and SCRIBE JSON payloads.
    """
    def __init__(self, output_dir="./cochem_mage_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_gaussian_peak(self, rt, intensity, width=0.05, resolution=500):
        """Generates a Gaussian peak array for plotting with high resolution (500 points) (MAGE-09)."""
        x = np.linspace(rt - (width * 4), rt + (width * 4), resolution)
        y = intensity * np.exp(-0.5 * ((x - rt) / width) ** 2)
        return x, y

    def build_interactive_chromatogram(self, job_queue, filename="mage_chromatogram.html"):
        """Compiles the theoretical GC-IMS-MS chromatogram into an interactive HTML widget."""
        fig = go.Figure()
        
        # Read predicted_tr from job_queue (MAGE-08)
        rts = []
        for j in job_queue:
            tr = j.get("predicted_tr") or j.get("estimated_rt")
            if tr is not None:
                rts.append(tr)
        max_rt = max(rts + [15.0]) + 2.0
        
        baseline_x = np.linspace(0, max_rt, 1000)
        global_y = np.zeros_like(baseline_x)

        for job in job_queue:
            if job.get("status") not in ["COMPUTED", "CACHED"]:
                continue
                
            # Read predicted_tr calculated by Stage 3.0 (MAGE-08)
            rt = job.get("predicted_tr")
            if rt is None:
                rt = job.get("estimated_rt")
            if rt is None:
                # Fallback only if predicted_tr and estimated_rt are both missing
                rt = max(1.5, min(15.0, job.get("logp", 2.0) * 1.5 + (job.get("mw", 100) / 50.0)))
            
            job["estimated_rt"] = round(rt, 2)
            intensity = job.get("peak_intensity", 1.0e6 * (1.0 + (job.get("tpsa", 0) / 100.0)))
            
            x_peak, y_peak = self._generate_gaussian_peak(rt, intensity, resolution=500)
            
            # Add individual traces for hover intelligence
            fig.add_trace(go.Scatter(
                x=x_peak, y=y_peak,
                mode='lines',
                name=job.get("smiles", "Unknown"),
                line=dict(width=2),
                fill='tozeroy',
                hovertemplate=(
                    f"<b>SMILES:</b> {job.get('smiles')}<br>"
                    f"<b>Class:</b> {job.get('chemical_class', 'N/A')}<br>"
                    f"<b>RT:</b> {rt:.2f} min<br>"
                    f"<b>CCS:</b> {job.get('ccs', 0)} Å²<br>"
                    f"<b>LogP:</b> {job.get('logp', 0):.2f}<extra></extra>"
                )
            ))

        fig.update_layout(
            title="CoChem-MAGE: Theoretical GC-IMS-MS Chromatogram",
            xaxis_title="Retention Time (Minutes)",
            yaxis_title="Simulated Abundance",
            template="plotly_white",
            hovermode="closest",
            showlegend=True
        )

        out_path = os.path.join(self.output_dir, filename)
        fig.write_html(out_path)
        print(f"📊 Interactive Chromatogram rendered to: {out_path}")
        return out_path

    def export_scribe_payload(self, job_queue, instrument_profile, filename="mage_scribe_payload.json"):
        """Serializes the batch metadata for CoChem-SCRIBE LLM ingestion."""
        payload = {
            "module": "CoChem-MAGE",
            "timestamp": datetime.now().isoformat(),
            "instrument_parameters": instrument_profile,
            "results": []
        }

        for job in job_queue:
            if job.get("status") in ["COMPUTED", "CACHED"]:
                payload["results"].append({
                    "smiles": job.get("smiles"),
                    "chemical_class": job.get("chemical_class"),
                    "mw": job.get("mw"),
                    "tpsa": job.get("tpsa"),
                    "ccs_proxy": job.get("ccs"),
                    "estimated_rt_min": job.get("estimated_rt", job.get("predicted_tr"))
                })

        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=4)
        print(f"📝 SCRIBE payload serialized to: {out_path}")
        return out_path
# %%