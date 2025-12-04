import matplotlib.pyplot as plt
import seaborn as sns

def set_style(context='talk'):
    """
    Set plot style for 'talk' or 'publication'.
    
    Args:
        context (str): 'talk' or 'publication'
    """
    # Base style
    sns.set_theme(style="ticks") # "ticks" is cleaner than "whitegrid", no grid by default
    
    if context == 'talk':
        font_scale = 1.5
        rc_params = {
            'axes.labelsize': 16,
            'axes.titlesize': 20,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'font.family': 'Arial'
        }
    else: # publication
        font_scale = 1.0
        rc_params = {
            'axes.labelsize': 8,
            'axes.titlesize': 12,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'font.family': 'Arial'
        }
        
    sns.set_context("notebook", font_scale=font_scale, rc=rc_params)
    
    # Ensure no grid
    plt.rcParams['axes.grid'] = False
    
def despine(ax=None):
    """Remove top and right spines."""
    sns.despine(ax=ax, top=True, right=True)
