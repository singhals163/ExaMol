import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Define vibrant and high-contrast colors from a cohesive academic palette
baseline_color = '#4A148C'   # Deep Richter Purple for "Baseline Always"
front_loaded_color = '#00ACC1' # Vibrant Teal for "Front Loaded"
no_training_color = '#7CB342'   # Crisp Green for "No Training"

# 1. Generate realistic synthetic data (based on the original distribution)
np.random.seed(42) # For consistent results

# Baseline Always: centered at higher energy with multiple peaks
baseline_data = np.concatenate([
    np.random.normal(loc=7.2, scale=0.4, size=300),
    np.random.normal(loc=6.5, scale=0.6, size=100),
    np.random.normal(loc=7.8, scale=0.2, size=100)
])

# Front Loaded: centered slightly lower with a broader spread
front_loaded_data = np.concatenate([
    np.random.normal(loc=6.8, scale=0.5, size=350),
    np.random.normal(loc=5.5, scale=0.7, size=100),
    np.random.normal(loc=7.6, scale=0.3, size=50)
])

# No Training: centered lowest with the largest spread
no_training_data = np.concatenate([
    np.random.normal(loc=6.5, scale=0.6, size=300),
    np.random.normal(loc=5.0, scale=1.0, size=150),
    np.random.normal(loc=7.3, scale=0.4, size=50)
])

# Combine into a single DataFrame
df = pd.DataFrame({
    'Redox Energy': np.concatenate([baseline_data, front_loaded_data, no_training_data]),
    'Method': np.concatenate([
        ['Baseline Always'] * len(baseline_data),
        ['Front Loaded'] * len(front_loaded_data),
        ['No Training'] * len(no_training_data)
    ])
})

# 2. Set Aesthetic Parameters for Academic Publishing
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Liberation Sans', 'Helvetica', 'Arial', 'DejaVu Sans'] # Professional sans-serif fonts
rcParams['axes.linewidth'] = 1.2 # Make axis lines cleaner

# 3. Initialize and Polishing the Plot
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
fig, ax = plt.subplots(figsize=(10, 6), dpi=300) # Large, high-resolution figure

# Remove standard top and right axes for a clean "open" plot, keeping only essential grid and data
sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True, offset=5) # Also remove left/bottom lines, grid gives structure

# Horizontal Grid Lines - Subtle and professional
ax.yaxis.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.1)
ax.xaxis.grid(False) # No vertical lines for focus on overlap

# 4. Create Multi-Overlay Density Plots (smooth, filled distributions)
# Using sns.kdeplot creates clean, smooth filled distributions that clearly show overlap.
# I've chosen defined vivid colors for maximum distinction and colorharmony.

# Baseline Always (Indigo)
sns.kdeplot(data=df[df['Method'] == 'Baseline Always']['Redox Energy'], 
            label='Baseline Always', color=baseline_color,
            fill=True, alpha=0.35, linewidth=2.5, ax=ax)

# Front Loaded (Vibrant Teal)
sns.kdeplot(data=df[df['Method'] == 'Front Loaded']['Redox Energy'],
            label='Front Loaded', color=front_loaded_color,
            fill=True, alpha=0.35, linewidth=2.5, ax=ax)

# No Training (Vivid Lime Green)
sns.kdeplot(data=df[df['Method'] == 'No Training']['Redox Energy'],
            label='No Training', color=no_training_color,
            fill=True, alpha=0.35, linewidth=2.5, ax=ax)

# 5. Define Axis Ticks and Labels for Legibility
# Set explicit tick positions and format them boldly
ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
# Set explicit y-tick labels, removing standard counts to make it general and clean
# Label as 'Density' or 'Frequency (Density)' if density is used. Original says Frequency. I will make a dense histogram style to maintain original label.
# Or better, just smooth fills and use 'Frequency' label but make counts clean.
# Multi-trajectories density plot with fills is correct. The original uses line and fill, multi-area.
# Let's just create defined hex codes and test for contrast.
# Ok, let's keep original *explicit* labels, but clean font.
# I've set up new vivid colors, and smooth fills.

# X-axis label: Larger, bold font
ax.set_xlabel("Redox Energy", fontsize=20, fontweight='bold', labelpad=15)
ax.set_xticklabels([str(t) for t in ax.get_xticks()], fontsize=16)

# Y-axis label: Larger, bold font, labeled consistently
ax.set_ylabel("Frequency", fontsize=20, fontweight='bold', labelpad=15)
ax.set_yticklabels([str(t) for t in ax.get_yticks()], fontsize=16)

# 6. Compact and Floating Legend
plt.legend(frameon=False, fontsize=16, loc='best') # No frame, large font, let Matplotlib find the best spot to not obscure main data.

# 7. Overall Container Styling
ax.set_title("", fontsize=0) # Remove any potential title space

# Final Polish: remove left/bottom axes, keep grid, data is everything. This is what 'better for paper' means. Clean data.
# The `sns.despine()` call above already does this.

# Tighten the overall figure layout
plt.tight_layout()

# 8. Show or Save the Figure
# Comment/uncomment as needed
plt.show() 
# plt.savefig('redox_energy_frequency_polished.png', dpi=600) # Save high-res for paper