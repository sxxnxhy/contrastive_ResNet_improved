"""
Find the maximum common X-range across all Raman files.
This will help us decide on a preprocessing strategy.
"""

import os
import numpy as np
from tqdm import tqdm

BASE_DATA_DIR = './data'
CLASS_NAMES = ["2-CEES", "2-CEPS", "DMMP", "4-NP"]
RAMAN_DIRS = {
    "2-CEES": "Raman_single/CEES",
    "2-CEPS": "Raman_single/CEPS",
    "DMMP": "Raman_single/DMMP",
    "4-NP": "Raman_single/4NP"
}

def load_raman_file(filepath):
    """Load Raman spectrum from CSV file."""
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.ndim == 1:
            return None, None
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except:
        return None, None

# Find global min and max across ALL files
global_min = float('inf')
global_max = float('-inf')

print("=" * 60)
print("FINDING GLOBAL COMMON X-RANGE")
print("=" * 60)

for class_name in CLASS_NAMES:
    print(f"\nProcessing {class_name}...")
    class_folder = os.path.join(BASE_DATA_DIR, RAMAN_DIRS[class_name])
    
    if not os.path.exists(class_folder):
        print(f"  Folder not found: {class_folder}")
        continue
    
    raman_files = [f for f in os.listdir(class_folder) if f.endswith('.csv')]
    
    class_min = float('inf')
    class_max = float('-inf')
    
    # Sample 1000 files per class for speed
    sample_size = min(1000, len(raman_files))
    sample_files = np.random.choice(raman_files, sample_size, replace=False)
    
    for filename in tqdm(sample_files, desc=f"  Sampling {class_name}"):
        filepath = os.path.join(class_folder, filename)
        x, y = load_raman_file(filepath)
        
        if x is not None and len(x) > 0:
            class_min = min(class_min, x.min())
            class_max = max(class_max, x.max())
    
    print(f"  Class range: [{class_min:.3f}, {class_max:.3f}]")
    
    global_min = max(global_min, class_min)  # Maximum of minimums
    global_max = min(global_max, class_max)  # Minimum of maximums

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\n🔍 Global COMMON range (intersection of all files):")
print(f"   [{global_min:.3f}, {global_max:.3f}]")
print(f"   Range span: {global_max - global_min:.3f} cm⁻¹")

print(f"\n💡 Recommendation:")
if global_max - global_min < 100:
    print("   ⚠️  Common range is too small!")
    print("   Suggestion: Use interpolation to a common grid instead.")
    print("   Proposed grid: [300, 3400] with 4096 points")
else:
    print(f"   ✅ Use common range [{global_min:.1f}, {global_max:.1f}]")
    print(f"   This covers {global_max - global_min:.1f} cm⁻¹")
    print(f"   Extract this range from each file, then interpolate to fixed length.")

print("\n" + "=" * 60)
print("INTERPOLATION STRATEGY")
print("=" * 60)
print("""
Option 1: COMMON RANGE (Conservative)
  - Extract only the overlapping region from all files
  - Pros: No extrapolation, uses real data only
  - Cons: May lose important spectral features outside common range

Option 2: FULL RANGE INTERPOLATION (Recommended)
  - Define a common grid: [300, 3400] cm⁻¹
  - Interpolate each file to this grid
  - Pros: Preserves all spectral information
  - Cons: Files with shorter ranges will have interpolated edges
  
Option 3: ZERO-PADDING (Current - Not recommended)
  - Pad missing regions with zeros
  - Pros: Simple
  - Cons: Creates artificial discontinuities, confuses the model

RECOMMENDATION: Use Option 2 (Full Range Interpolation)
  - Set target grid: np.linspace(300, 3400, 4096)
  - Use scipy.interpolate.interp1d with fill_value for extrapolation
  - This gives consistent 4096-point vectors for all files
""")
