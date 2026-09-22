"""Plot saved analysis tables; this module does not fit models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(out: Path):
    dest = out / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    table = pd.read_csv(out / 'tables' / 'main_models.csv')
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(table))
    ax.barh(y-.18, table['CV_Accuracy'], .34, label='Repeated group CV', color='#a5b9c5')
    ax.barh(y+.18, table['Accuracy'], .34, label='Leave one field out', color='#226d85')
    ax.set(yticks=y, yticklabels=table['Method'], xlabel='Accuracy', xlim=(0,1), title='60-m five-class model comparison')
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    save_figure(fig, dest / 'model_generalization')

    path = out / 'altitude' / '07_table5_altitude_midpoint.csv'
    if path.exists():
        frame = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(7,4))
        for method, group in frame.groupby('Method'):
            group = group.sort_values('Altitude')
            ax.plot(group['Altitude'], group['Accuracy'], marker='o', label=method)
        ax.set(xlabel='Flight altitude (m)', ylabel='LOFO accuracy', xticks=[20,40,60,80,100,120], ylim=(0,1), title='Fields 1–3 across flight heights')
        ax.legend()
        save_figure(fig, dest / 'altitude_performance')

    path = out / 'sensor' / '10_table6_sensor_transfer_midpoint.csv'
    marker = out / 'sensor_status.json'
    sensor_current = not marker.exists() or json.loads(marker.read_text(encoding='utf-8'))['completed']
    if path.exists() and sensor_current:
        frame = pd.read_csv(path)
        frame = frame[frame['Method'] == 'MidpointCompact_LR']
        fig, ax = plt.subplots(figsize=(9,4))
        ax.barh(frame['Strategy'], frame['Accuracy'], color='#226d85')
        ax.set(xlabel='Held-field accuracy', xlim=(0,1), title='Midpoint compact model: paired L1/H300 cohort')
        save_figure(fig, dest / 'sensor_transfer')
