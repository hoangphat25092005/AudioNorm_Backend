import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_cnn_architecture_diagram():
    """
    Create a detailed visual diagram of the AudioNorm CNN architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Title
    ax.text(10, 14.5, 'AudioNorm CNN Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'conv': '#FFE6CC', 
        'pool': '#D4EDDA',
        'fc': '#F8D7DA',
        'output': '#D1ECF1',
        'feature': '#FFF3CD'
    }
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 12), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 12.75, 'Input Spectrogram\n(batch, 64, T)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Additional features input
    feature_box = FancyBboxPatch((0.5, 10), 3, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['feature'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(feature_box)
    ax.text(2, 10.5, 'Audio Features\n(batch, 9)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Conv Block 1
    conv1_box = FancyBboxPatch((5, 11.5), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(conv1_box)
    ax.text(6.5, 12.5, 'Conv Block 1\nConv2d(1→32, 3×3)\nBatchNorm + ReLU\nMaxPool(2×2)\nDropout(0.25)', 
            ha='center', va='center', fontsize=9)
    
    # Conv Block 2
    conv2_box = FancyBboxPatch((9, 11.5), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(conv2_box)
    ax.text(10.5, 12.5, 'Conv Block 2\nConv2d(32→64, 3×3)\nBatchNorm + ReLU\nMaxPool(2×2)\nDropout(0.25)', 
            ha='center', va='center', fontsize=9)
    
    # Conv Block 3
    conv3_box = FancyBboxPatch((13, 11.5), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(conv3_box)
    ax.text(14.5, 12.5, 'Conv Block 3\nConv2d(64→128, 3×3)\nBatchNorm + ReLU\nAdaptiveAvgPool(4×4)\nDropout(0.25)', 
            ha='center', va='center', fontsize=9)
    
    # Flatten layer
    flatten_box = FancyBboxPatch((13, 9), 3, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['pool'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(flatten_box)
    ax.text(14.5, 9.5, 'Flatten\n(batch, 2048)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Concatenation
    concat_box = FancyBboxPatch((10, 7.5), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['pool'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(12, 8, 'Concatenate Features\n(batch, 2057)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # FC Layer 1
    fc1_box = FancyBboxPatch((8, 6), 6, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['fc'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(fc1_box)
    ax.text(11, 6.5, 'FC Layer 1: Linear(2057→256) + BatchNorm + ReLU + Dropout(0.5)', 
            ha='center', va='center', fontsize=9)
    
    # FC Layer 2
    fc2_box = FancyBboxPatch((8, 4.5), 6, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['fc'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(fc2_box)
    ax.text(11, 5, 'FC Layer 2: Linear(256→128) + BatchNorm + ReLU + Dropout(0.3)', 
            ha='center', va='center', fontsize=9)
    
    # FC Layer 3
    fc3_box = FancyBboxPatch((8, 3), 6, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['fc'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(fc3_box)
    ax.text(11, 3.5, 'FC Layer 3: Linear(128→64) + ReLU + Dropout(0.2)', 
            ha='center', va='center', fontsize=9)
    
    # Output layer
    output_box = FancyBboxPatch((9, 1.5), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(11, 2, 'Output: Linear(64→1)\nGain in dB', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Main flow arrows
    ax.annotate('', xy=(5, 12.5), xytext=(3.5, 12.5), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 12.5), xytext=(8, 12.5), arrowprops=arrow_props)
    ax.annotate('', xy=(13, 12.5), xytext=(12, 12.5), arrowprops=arrow_props)
    ax.annotate('', xy=(14.5, 11.5), xytext=(14.5, 10), arrowprops=arrow_props)
    
    # Feature concatenation arrows
    ax.annotate('', xy=(10, 8), xytext=(3.5, 10.5), arrowprops=arrow_props)
    ax.annotate('', xy=(12, 8.5), xytext=(14.5, 9), arrowprops=arrow_props)
    
    # FC layer arrows
    ax.annotate('', xy=(11, 7.5), xytext=(11, 7), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 6), xytext=(11, 5.5), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 4.5), xytext=(11, 4), arrowprops=arrow_props)
    ax.annotate('', xy=(11, 3), xytext=(11, 2.5), arrowprops=arrow_props)
    
    # Add dimension annotations
    ax.text(4, 11, '(1,64,T)', ha='center', fontsize=8, style='italic')
    ax.text(8, 11, '(32,32,T/2)', ha='center', fontsize=8, style='italic')
    ax.text(12, 11, '(64,16,T/4)', ha='center', fontsize=8, style='italic')
    ax.text(16.5, 11, '(128,4,4)', ha='center', fontsize=8, style='italic')
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['feature'], label='Feature Input'),
        patches.Patch(color=colors['conv'], label='Convolutional Block'),
        patches.Patch(color=colors['pool'], label='Processing Layer'),
        patches.Patch(color=colors['fc'], label='Fully Connected'),
        patches.Patch(color=colors['output'], label='Output Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    return fig

# Create and save the diagram
if __name__ == "__main__":
    fig = create_cnn_architecture_diagram()
    
    # Save the figure
    plt.savefig('AudioNorm_CNN_Architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('AudioNorm_CNN_Architecture.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Architecture diagrams saved:")
    print("   📊 AudioNorm_CNN_Architecture.png")
    print("   📄 AudioNorm_CNN_Architecture.pdf")
    
    plt.show()
