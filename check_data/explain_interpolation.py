"""
Demonstrate how np.interp solves BOTH X-range AND length problems simultaneously.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("HOW np.interp() SOLVES BOTH X-RANGE AND LENGTH PROBLEMS")
print("=" * 80)

# Simulate 3 different files with different ranges AND different lengths
files = {
    "File A (short range, few points)": {
        "x": np.linspace(400, 1500, 500),      # 500 points, short range
        "desc": "500 points in [400, 1500]"
    },
    "File B (wide range, medium points)": {
        "x": np.linspace(401, 3399, 2040),     # 2040 points, wide range
        "desc": "2040 points in [401, 3399]"
    },
    "File C (full range, many points)": {
        "x": np.linspace(300, 3200, 7935),     # 7935 points, full range
        "desc": "7935 points in [300, 3200]"
    }
}

# Add some peaks to visualize
for name, data in files.items():
    x = data["x"]
    # Create synthetic peaks at different positions
    if "short" in name:
        y = np.exp(-((x - 800) / 100)**2) + 0.5 * np.exp(-((x - 1200) / 80)**2)
    elif "wide" in name:
        y = np.exp(-((x - 1000) / 150)**2) + 0.7 * np.exp(-((x - 2500) / 200)**2)
    else:
        y = np.exp(-((x - 800) / 100)**2) + 0.5 * np.exp(-((x - 1800) / 150)**2) + 0.3 * np.exp(-((x - 2800) / 100)**2)
    data["y"] = y

# Target common grid (what all files will be interpolated to)
TARGET_LENGTH = 4096
TARGET_X_MIN = 300
TARGET_X_MAX = 3400
common_grid = np.linspace(TARGET_X_MIN, TARGET_X_MAX, TARGET_LENGTH)

print(f"\n🎯 TARGET: All files will be converted to:")
print(f"   - X-range: [{TARGET_X_MIN}, {TARGET_X_MAX}]")
print(f"   - Length: {TARGET_LENGTH} points")
print(f"   - Grid spacing: {(TARGET_X_MAX - TARGET_X_MIN) / TARGET_LENGTH:.4f} cm⁻¹")

# Interpolate all files to common grid
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('np.interp() Solves BOTH X-range AND Length Problems', fontsize=16, fontweight='bold')

for idx, (name, data) in enumerate(files.items()):
    x_orig = data["x"]
    y_orig = data["y"]
    
    # INTERPOLATION: This single operation solves BOTH problems!
    y_interpolated = np.interp(common_grid, x_orig, y_orig, left=0.0, right=0.0)
    
    # Normalize
    y_min, y_max = y_interpolated.min(), y_interpolated.max()
    y_normalized = (y_interpolated - y_min) / (y_max - y_min + 1e-10)
    
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"  📥 INPUT:")
    print(f"     X-range: [{x_orig.min():.1f}, {x_orig.max():.1f}]")
    print(f"     Length: {len(x_orig)} points")
    print(f"     Data span: {x_orig.max() - x_orig.min():.1f} cm⁻¹")
    print(f"     Resolution: {(x_orig.max() - x_orig.min()) / len(x_orig):.4f} cm⁻¹/point")
    
    print(f"\n  📤 OUTPUT (after np.interp):")
    print(f"     X-range: [{common_grid.min():.1f}, {common_grid.max():.1f}]")
    print(f"     Length: {len(y_interpolated)} points ✅")
    print(f"     Zero-padding: {(y_interpolated == 0).sum()} points ({(y_interpolated == 0).sum()/len(y_interpolated)*100:.1f}%)")
    print(f"     Data region: {(y_interpolated != 0).sum()} points ({(y_interpolated != 0).sum()/len(y_interpolated)*100:.1f}%)")
    
    print(f"\n  ✅ PROBLEMS SOLVED:")
    print(f"     ✓ X-range unified: {len(x_orig)} points spanning {x_orig.max()-x_orig.min():.0f} cm⁻¹")
    print(f"       → {TARGET_LENGTH} points spanning {TARGET_X_MAX-TARGET_X_MIN:.0f} cm⁻¹")
    print(f"     ✓ Length unified: {len(x_orig)} → {TARGET_LENGTH}")
    print(f"     ✓ Peak positions preserved at correct wavenumbers")
    
    # Plot original
    ax = axes[idx, 0]
    ax.plot(x_orig, y_orig, 'b-', linewidth=1, label=f'Original ({len(x_orig)} pts)')
    ax.set_title(f'{name}\nOriginal: {data["desc"]}')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(300, 3400)
    ax.axvline(x_orig.min(), color='red', linestyle='--', alpha=0.5, label='Data range')
    ax.axvline(x_orig.max(), color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot interpolated
    ax = axes[idx, 1]
    ax.plot(common_grid, y_normalized, 'g-', linewidth=0.5, label=f'Interpolated ({len(y_normalized)} pts)')
    ax.set_title(f'{name}\nAfter np.interp: {TARGET_LENGTH} points, [300, 3400]')
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Normalized Intensity [0, 1]')
    ax.set_xlim(300, 3400)
    ax.axvline(x_orig.min(), color='red', linestyle='--', alpha=0.5, label='Original data range')
    ax.axvline(x_orig.max(), color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight zero-padded regions
    zero_regions = np.where(y_normalized == 0)[0]
    if len(zero_regions) > 0:
        ax.axvspan(common_grid[0], common_grid[zero_regions[0]] if zero_regions[0] > 0 else common_grid[0], 
                   alpha=0.2, color='red', label='Zero-padded')
        if zero_regions[-1] < len(common_grid) - 1:
            ax.axvspan(common_grid[zero_regions[-1]], common_grid[-1], alpha=0.2, color='red')

plt.tight_layout()
plt.savefig('interpolation_explanation.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print("✅ Visualization saved: interpolation_explanation.png")
print("="*80)

print(f"\n{'='*80}")
print("SUMMARY: WHY THIS WORKS")
print("="*80)
print("""
np.interp(common_grid, x_original, y_original, left=0, right=0) does 3 things:

1️⃣  RESAMPLING (solves length problem):
   - Takes original data with ANY number of points (500, 2040, 7935, ...)
   - Interpolates to EXACTLY 4096 points
   - Uses linear interpolation to estimate values at new X positions

2️⃣  X-RANGE ALIGNMENT (solves range problem):
   - Takes original data with ANY X-range ([400,1500], [300,3200], ...)
   - Maps to common X-range [300, 3400]
   - Finds where original data exists on the common grid

3️⃣  ZERO-PADDING (handles missing data):
   - left=0: If common_grid starts before original data, pad with 0
   - right=0: If common_grid extends beyond original data, pad with 0
   - Preserves peaks at their correct wavenumber positions

RESULT:
   - Every file → 4096 points
   - Every file → [300, 3400] cm⁻¹ range
   - Peaks appear at correct positions
   - Model gets consistent input shape
   - No information loss (peaks preserved)
   
BEFORE: 500~7935 points, various X-ranges → Model confused ❌
AFTER:  4096 points, unified [300,3400] → Model can learn ✅
""")
