import logging
from typing import Any
logger = logging.getLogger(__name__)
import os
import re
import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import dask.dataframe as dd
from pathlib import Path

def scaffold_directories(base_path: str | Path) -> Any:
    """
    Zero-Click Deployment: Silently handle directory scaffolding.
    Uses pathlib for all OS paths.
    """
    base = Path(base_path)
    os.makedirs(base, exist_ok=True)
    return base

def parse_massive_xyz(file_path: str | Path) -> Any:
    """
    Parses massive atomic coordinate files (XYZ) using pyarrow to prevent OOM exceptions.
    Returns a PyArrow Table.
    """
    path = Path(file_path)
    parse_options = csv.ParseOptions(delimiter=' ', ignore_empty_lines=True)
    
    # XYZ format typically features two header lines before coordinate data
    read_options = csv.ReadOptions(skip_rows=2, column_names=["Element", "X", "Y", "Z"])
    convert_options = csv.ConvertOptions(
        column_types={"Element": pa.string(), "X": pa.float64(), "Y": pa.float64(), "Z": pa.float64()}
    )
    
    return csv.read_csv(str(path), parse_options=parse_options, read_options=read_options, convert_options=convert_options)

def extract_coords_from_log_regex(log_path: str | Path) -> Any:
    """
    Auto-extracts atomic coordinates from logs via regex before passing to context.
    """
    path = Path(log_path)
    # Regex to capture atomic elements and 3 float coordinates
    coord_pattern = re.compile(r'^\s*([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
    coords = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = coord_pattern.match(line)
            if match:
                coords.append(match.groups())
    return coords

def substitute_isotopes_for_kie(xyz_path: str | Path, output_path: str | Path, target='H', isotope='D') -> Any:
    """
    Dynamically substitutes specific atoms (e.g., 1H for 2D) to calculate KIEs 
    without requiring redrawing in external GUI tools.
    """
    fin_path = Path(xyz_path)
    fout_path = Path(output_path)
    
    with open(fin_path, 'r', encoding='utf-8') as fin, open(fout_path, 'w', encoding='utf-8') as fout:
        lines = fin.readlines()
        if not lines: return
        
        # Preserve standard XYZ header
        fout.write(lines[0]) 
        fout.write(lines[1]) 
        
        for line in lines[2:]:
            parts = line.split()
            if not parts: continue
            
            if parts[0] == target:
                parts[0] = isotope
            
            if len(parts) >= 4:
                fout.write(f"{parts[0]:<4} {parts[1]:>12} {parts[2]:>12} {parts[3]:>12}\n")
            else:
                fout.write(line)

def _largest_triangle_three_buckets(data, threshold=1000) -> Any:
    """
    Core implementation of Largest Triangle Three Buckets (LTTB) algorithm.
    Downsamples 1,000,000-point spectra to 1,000 points for fast UI rendering.
    """
    data_len = len(data)
    if threshold >= data_len or threshold == 0:
        return data

    sampled = np.zeros((threshold, 2))
    sampled[0] = data[0]
    sampled[threshold - 1] = data[data_len - 1]

    every = (data_len - 2) / (threshold - 2)
    a = 0
    next_a = 0

    for i in range(threshold - 2):
        avg_x = 0
        avg_y = 0
        avg_range_start = int((i + 1) * every) + 1
        avg_range_end = int((i + 2) * every) + 1
        avg_range_end = min(avg_range_end, data_len)
        avg_range_length = avg_range_end - avg_range_start

        while avg_range_start < avg_range_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        range_offs = int((i + 0) * every) + 1
        range_to = int((i + 1) * every) + 1
        
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1
        max_area_point = data[range_offs]
        
        while range_offs < range_to:
            area = abs((point_ax - avg_x) * (data[range_offs][1] - point_ay) -
                       (point_ax - data[range_offs][0]) * (avg_y - point_ay)) * 0.5
            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs
            range_offs += 1

        sampled[i + 1] = max_area_point
        a = next_a

    return sampled

def process_massive_spectra(csv_path: str | Path) -> Any:
    """
    Uses Dask for large spectral datasets to prevent OOM errors, 
    extracting data to numpy for downstream LTTB rendering.
    """
    path = Path(csv_path)
    df = dd.read_csv(str(path))
    # Retrieve just the required plotting columns and run LTTB 
    data = df[['Wavenumber', 'Intensity']].compute().to_numpy()
    return _largest_triangle_three_buckets(data, threshold=1000)

def translate_cryptic_traceback(error_msg: str) -> str:
    """
    Translates cryptic C++ or runtime tracebacks into human-readable actionable hardware terms.
    """
    error_msg_lower = error_msg.lower()
    
    if "segmentation fault" in error_msg_lower or "segfault" in error_msg_lower:
        return "Hardware Translation: You ran out of RAM on step 4. Try requesting a larger memory allocation."
    elif "scf failed" in error_msg_lower:
        return "Hardware Translation: Electronic structure failed to converge. Often occurs if your integration grid is too tight early on (start optimization loops on defgrid1, tighten to defgrid1 near minimum)."
    elif "oom" in error_msg_lower or "out of memory" in error_msg_lower or "killed" in error_msg_lower:
        return "Hardware Translation: Your process was killed because it ran out of RAM."
    elif "disk quota exceeded" in error_msg_lower or "no space left on device" in error_msg_lower:
        return "Hardware Translation: You ran out of storage space. Clean up old scratch or log files."
    else:
        return "Hardware Translation: Unknown error. Please ensure your geometry is sane and that you are using the CREST/ORCA GOAT combination approach for conformer generation."

if __name__ == "__main__":
    # Scaffold baseline directories upon standalone execution
    scaffold_directories("D:/Gdrive/__CoChem/GitHub-Repo/CoChem-EXEC/data")
