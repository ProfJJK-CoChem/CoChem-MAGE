import logging
from typing import Any
logger = logging.getLogger(__name__)
import hashlib  # SHA-256 artifact provenance tracking
# %%
import os
import json
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

def _safe_float(val, default=0.0) -> Any:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class MageExporter:
    """
    Stage 3.0: Visualization and Narrative Handoff.
    Generates interactive HTML chromatograms and SCRIBE JSON payloads.
    """
    def __init__(self, output_dir="./cochem_mage_output") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _generate_gaussian_peak(self, rt, intensity, width=0.05, resolution=500) -> Any:
        """Generates a Gaussian peak array for plotting with high resolution (500 points) (MAGE-09)."""
        x = np.linspace(rt - (width * 4), rt + (width * 4), resolution)
        y = intensity * np.exp(-0.5 * ((x - rt) / width) ** 2)
        return x, y

    def build_interactive_chromatogram(self, job_queue, filename="mage_chromatogram.html") -> Any:
        """Compiles the theoretical GC-IMS-MS chromatogram into an interactive HTML widget."""
        fig = go.Figure()
        
        if not job_queue or not isinstance(job_queue, (list, tuple)):
            job_queue = []
        valid_jobs = [j for j in job_queue if isinstance(j, dict)]

        # Read predicted_tr from job_queue (MAGE-08)
        rts = []
        for j in valid_jobs:
            tr = j.get("predicted_tr") or j.get("estimated_rt")
            if tr is not None:
                rts.append(tr)
        max_rt = max(rts + [15.0]) + 2.0
        
        baseline_x = np.linspace(0, max_rt, 1000)
        global_y = np.zeros_like(baseline_x)

        for job in valid_jobs:
            if job.get("status") not in ["COMPUTED", "CACHED"]:
                continue
                
            # Read predicted_tr calculated by Stage 3.0 (MAGE-08)
            rt = job.get("predicted_tr")
            if rt is None:
                rt = job.get("estimated_rt")
            if rt is None or not isinstance(rt, (int, float)):
                rt = max(1.5, min(15.0, _safe_float(job.get("logp"), 2.0) * 1.5 + (_safe_float(job.get("mw"), 100.0) / 50.0)))
            
            job["estimated_rt"] = round(rt, 2)
            intensity = _safe_float(job.get("peak_intensity"), 1.0e6 * (1.0 + (_safe_float(job.get("tpsa"), 0.0) / 100.0)))
            
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
                    f"<b>LogP:</b> {_safe_float(job.get('logp')):.2f}<extra></extra>"
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
        logger.info(f"📊 Interactive Chromatogram rendered to: {out_path}")
        return out_path

    def export_scribe_payload(self, job_queue, instrument_profile, filename="mage_scribe_payload.json") -> Any:
        """Serializes the batch metadata for CoChem-SCRIBE LLM ingestion."""
        payload = {
            "module": "CoChem-MAGE",
            "timestamp": datetime.now().isoformat(),
            "instrument_parameters": instrument_profile if isinstance(instrument_profile, dict) else {},
            "results": []
        }

        if not job_queue or not isinstance(job_queue, (list, tuple)):
            job_queue = []
        valid_jobs = [j for j in job_queue if isinstance(j, dict)]

        for job in valid_jobs:
            if job.get("status") in ["COMPUTED", "CACHED"]:
                payload["results"].append({
                    "smiles": job.get("smiles"),
                    "chemical_class": job.get("chemical_class"),
                    "mw": job.get("mw"),
                    "tpsa": job.get("tpsa"),
                    "ccs_proxy": job.get("ccs"),
                    "estimated_rt_min": job.get("estimated_rt", job.get("predicted_tr")),
                    "provenance_tag": job.get("provenance_tag", "[D]" if job.get("status") == "COMPUTED" else "[E]")
                })

        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info(f"📝 SCRIBE payload serialized to: {out_path}")
        return out_path

    def build_head_to_tail_ms_plot(self, exp_spectrum, pred_spectrum, filename="head_to_tail_ms.html") -> Any:
        """
        Generates interactive head-to-tail m/z stick plot comparing experimental (top) vs predicted (bottom) MS spectra (Suggestion 68).
        exp_spectrum, pred_spectrum: dict of {mz: intensity} or list of (mz, intensity) tuples.
        """
        def normalize_spectrum(spec) -> Any:
            mz_list = []
            int_list = []
            if isinstance(spec, dict):
                for k, v in spec.items():
                    try:
                        mz_f = float(k)
                        int_f = float(v)
                        mz_list.append(mz_f)
                        int_list.append(int_f)
                    except (ValueError, TypeError):
                        continue
            elif isinstance(spec, (list, tuple, np.ndarray)):
                for item in spec:
                    try:
                        if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                            mz_f = float(item[0])
                            int_f = float(item[1])
                            mz_list.append(mz_f)
                            int_list.append(int_f)
                    except (ValueError, TypeError):
                        continue
            mz_vals = np.array(mz_list, dtype=float)
            int_vals = np.array(int_list, dtype=float)
            if len(int_vals) > 0 and np.max(int_vals) > 0:
                int_vals = (int_vals / np.max(int_vals)) * 100.0
            return mz_vals, int_vals

        exp_mz, exp_int = normalize_spectrum(exp_spectrum)
        pred_mz, pred_int = normalize_spectrum(pred_spectrum)

        fig = go.Figure()

        # Upper plot: Experimental MS (positive sticks)
        for mz, i in zip(exp_mz, exp_int):
            fig.add_trace(go.Scatter(
                x=[mz, mz], y=[0, i],
                mode='lines',
                line=dict(color='crimson', width=2),
                showlegend=False,
                hoverinfo='text',
                text=f"Exp m/z: {mz:.1f}, Intensity: {i:.1f}%"
            ))

        # Lower plot: Predicted MS (negative sticks)
        for mz, i in zip(pred_mz, pred_int):
            fig.add_trace(go.Scatter(
                x=[mz, mz], y=[0, -i],
                mode='lines',
                line=dict(color='royalblue', width=2),
                showlegend=False,
                hoverinfo='text',
                text=f"Pred m/z: {mz:.1f}, Intensity: {i:.1f}%"
            ))

        # Reference zero line
        all_mz = np.concatenate([exp_mz, pred_mz]) if len(exp_mz) or len(pred_mz) else np.array([0, 500])
        min_mz = max(0, np.min(all_mz) - 10) if len(all_mz) > 0 else 0
        max_mz = (np.max(all_mz) + 10) if len(all_mz) > 0 else 500
        fig.add_shape(type="line", x0=min_mz, y0=0, x1=max_mz, y1=0, line=dict(color="black", width=1))

        fig.update_layout(
            title="CoChem-MAGE: Head-to-Tail Mass Spectral Comparison (Experimental Top / Predicted Bottom)",
            xaxis_title="m/z (Mass-to-Charge Ratio)",
            yaxis_title="Relative Abundance (Top: Exp + | Bottom: Pred -)",
            template="plotly_white",
            yaxis=dict(range=[-110, 110])
        )

        out_path = os.path.join(self.output_dir, filename)
        fig.write_html(out_path)
        logger.info(f"📊 Head-to-tail MS plot rendered to: {out_path}")
        return out_path

    def export_to_parquet(self, job_queue, filename="mage_ri_catalog.parquet") -> Any:
        """
        Exports Retention Index (RI) catalog dataset into PyArrow Parquet format (Suggestion 69).
        """
        import pandas as pd
        if not job_queue or not isinstance(job_queue, (list, tuple)):
            return pd.DataFrame()
        valid_jobs = [j for j in job_queue if isinstance(j, dict)]
        records = []
        for job in valid_jobs:
            records.append({
                "id": str(job.get("id", "") if job.get("id") is not None else ""),
                "smiles": str(job.get("smiles", "") if job.get("smiles") is not None else ""),
                "chemical_class": str(job.get("chemical_class", "N/A") if job.get("chemical_class") is not None else "N/A"),
                "mw": _safe_float(job.get("mw"), 0.0),
                "logp": _safe_float(job.get("logp"), 0.0),
                "tpsa": _safe_float(job.get("tpsa"), 0.0),
                "predicted_ri": _safe_float(job.get("predicted_ri"), 0.0),
                "predicted_tr": _safe_float(job.get("predicted_tr"), 0.0),
                "status": str(job.get("status", "UNKNOWN") if job.get("status") is not None else "UNKNOWN"),
                "provenance_tag": str(job.get("provenance_tag") or ("[D]" if job.get("status") == "COMPUTED" else "[E]"))
            })
        df = pd.DataFrame(records)
        out_path = os.path.join(self.output_dir, filename)
        try:
            df.to_parquet(out_path, engine="pyarrow", index=False)
            logger.info(f"📦 RI catalog exported to PyArrow Parquet: {out_path}")
        except Exception as e:
            # Fallback to fastparquet or json export if pyarrow engine fails
            try:
                df.to_parquet(out_path, index=False)
                logger.info(f"📦 RI catalog exported to Parquet (fallback): {out_path}")
            except Exception as ex:
                json_path = out_path.replace(".parquet", ".json")
                df.to_json(json_path, orient="records", indent=4)
                logger.error(f"⚠️ Parquet export error ({e}). Exported catalog as JSON: {json_path}")
                return json_path
        return out_path

# %%