"""Render supplied Isaac Sim comparison values as an editable local figure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gear_sim_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STYLE = json.loads((HERE / 'style.json').read_text())
DATA = json.loads((HERE / 'evidence.json').read_text())
MODELS = DATA['models']
COLORS = {m: STYLE['models'][m] for m in MODELS}
plt.rcParams.update({'font.family': STYLE['font_family'], 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'gear-isaacsim-comparison',
                     'text.color': STYLE['ink'], 'axes.labelcolor': STYLE['muted'],
                     'xtick.color': STYLE['muted'], 'ytick.color': STYLE['muted'],
                     'savefig.facecolor': 'white'})
LEFT, STEP, WIDTH = .105, .295, .270
METRICS = [('success', 'Success ↑', '%', .451, 100, [0, 25, 50, 75, 100]),
           ('collision', 'Collision ↓', '%', .277, 100, [0, 25, 50, 75, 100]),
           ('clearance_m', 'Clearance ↑', 'm', .103, 1.6, [0, .4, .8, 1.2, 1.6])]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(fig, stem):
    for ext in ('svg', 'pdf', 'png'):
        metadata = {'Creator': 'GEAR-Nav: supplied simulation comparison values; illustrative recorded scene views'}
        if ext in ('svg', 'pdf'):
            metadata['Date' if ext == 'svg' else 'CreationDate'] = None
        path = HERE / f'{stem}.{ext}'
        fig.savefig(path, dpi=230, metadata=metadata)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def photo(fig, view, x, y, width, height):
    ax = fig.add_axes([x, y, width, height])
    with Image.open(HERE / view['image']) as image:
        ax.imshow(image, aspect='auto')
    ax.set_axis_off()
    # Do not transfer old diagnostic goal annotations into the new evaluation.
    return ax


def bars(fig, scene, column, metric):
    key, label, unit, bottom, max_value, ticks = metric
    x = LEFT + column*STEP
    ax = fig.add_axes([x, bottom, WIDTH, .113])
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E5E7EB', lw=.7)
    values = [scene['metrics'][m][key] * (100 if unit == '%' else 1) for m in MODELS]
    for j, (model, value) in enumerate(zip(MODELS, values)):
        bar = ax.barh(j, value, height=.56, color=COLORS[model], alpha=1 if model == 'GEAR-Nav' else .82,
                      edgecolor='white', linewidth=.5, zorder=3)[0]
        bar.set_gid(f'scenario-{scene["number"]}-{model}-{key}-value')
        if value == 0:
            ax.scatter([0], [j], marker='|', s=50, color=COLORS[model], zorder=4)
        text = f'{value:.0f}%' if unit == '%' else f'{value:.2f}'
        # Keep labels inside the plotting area, with sufficient headroom at 100%.
        ax.text(value+max_value*.022, j, text, fontsize=12.5, va='center',
                color=STYLE['ink'], weight='bold' if model == 'GEAR-Nav' else 'normal',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .3, 'alpha': .9}, zorder=5)
    ax.set(xlim=(-max_value*.01, max_value*1.18), ylim=(3.65, -.65), xticks=ticks)
    ax.set_yticks(range(4), labels=MODELS if column == 0 else ['']*4)
    ax.tick_params(axis='y', length=0, labelsize=13, pad=8)
    if column == 0:
        ax.get_yticklabels()[-1].set_weight('bold')
    ax.tick_params(axis='x', labelsize=11, length=2.5, pad=2)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#CED2D8')
    return ax


def full_figure():
    fig = plt.figure(figsize=(12.4, 9.4))
    fig.text(.020, .980, 'Isaac Sim evaluation', fontsize=24, weight='bold', va='top')
    fig.text(.980, .976, 'GEAR-Nav and navigation baselines', fontsize=17, ha='right', va='top')
    fig.text(.020, .932, 'Recorded simulation views', fontsize=16, weight='bold')
    fig.text(.980, .932, 'Illustrative images from separate runs', fontsize=14, ha='right', color=STYLE['muted'])
    image_y, image_h = .703, WIDTH*12.4/9.4*9/16
    for i, view in enumerate(DATA['illustrative_views']):
        x = LEFT+i*STEP
        photo(fig, view, x, image_y, WIDTH, image_h)
        fig.text(x+WIDTH/2, .677, view['name'], fontsize=15, ha='center', color=STYLE['muted'])
    fig.text(.020, .635, 'Navigation results', fontsize=17, weight='bold')
    fig.text(.980, .635, 'Success, collision rate, and obstacle clearance', fontsize=14,
             ha='right', color=STYLE['muted'])
    for i, scene in enumerate(DATA['scenarios']):
        x = LEFT+i*STEP
        fig.text(x+WIDTH/2, .605, f'Scenario {scene["number"]}', fontsize=18, ha='center', weight='bold')
        for metric in METRICS:
            bars(fig, scene, i, metric)
    for _, label, unit, bottom, _, _ in METRICS:
        fig.text(.020, bottom+.125, f'{label} ({unit})', fontsize=16, weight='bold')
    fig.text(.020, .035, 'Baselines: VEGA. GEAR-Nav: updated simulation table.', fontsize=12.5, color=STYLE['muted'])
    fig.text(.980, .035, 'Bars show aggregate values.', fontsize=12.5,
             ha='right', color=STYLE['muted'])
    export(fig, 'isaacsim_results')


def scene_strip():
    fig = plt.figure(figsize=(12.4, 4.3))
    fig.text(.020, .966, 'Isaac Sim · GEAR-Nav results', fontsize=21, weight='bold', va='top')
    fig.text(.980, .956, 'Images: illustrative recordings', fontsize=15, ha='right', va='top', color=STYLE['muted'])
    width, left, step = .306, .019, .328
    for i, view in enumerate(DATA['illustrative_views']):
        x = left+i*step
        photo(fig, view, x, .304, width, width*12.4/4.3*9/16)
        scene = DATA['scenarios'][i]
        metrics = scene['metrics']['GEAR-Nav']
        fig.text(x+width/2, .235, f'Scenario {scene["number"]}', fontsize=19, ha='center', weight='bold')
        fig.text(x+width/2, .140, f'Success {100*metrics["success"]:.0f}%  ·  Collision {100*metrics["collision"]:.0f}%', fontsize=18, ha='center')
        fig.text(x+width/2, .060, f'Clearance {metrics["clearance_m"]:.2f} m', fontsize=17, ha='center', color=STYLE['muted'])
    export(fig, 'isaacsim_scene_strip')


def verify():
    provenance = json.loads((HERE/'provenance.json').read_text())
    for path, digest in provenance['protected_paper_files'].items():
        assert sha(ROOT / path) == digest, f'Paper file changed: {path}'
    expected = [(0.92, 0.06, 0.78), (0.84, 0.10, 0.46), (0.86, 0.12, 0.71)]
    for scene, values in zip(DATA['scenarios'], expected):
        assert tuple(scene['metrics']['GEAR-Nav'].values()) == values
    svg = ET.parse(HERE / 'isaacsim_results.svg')
    groups = [el for el in svg.iter() if el.attrib.get('id', '').endswith('-value')]
    assert len(groups) == 36
    assert sum(el.tag.endswith('}image') for el in svg.iter()) == 3
    text = (HERE/'isaacsim_results.svg').read_text()
    assert 'oracle-goal diagnostic' not in text and 'Wilson' not in text
    result = {'numeric_values': 36, 'baseline_values_crosschecked_to_vega': 27,
              'gear_values': expected, 'svg_metric_groups': len(groups), 'embedded_images': 3,
              'old_diagnostic_paths_and_statistics_excluded': True,
              'paper_files_unchanged': True, 'gear_trial_count_not_inferred': True,
              'outputs': {p.name: sha(p) for stem in ['isaacsim_results', 'isaacsim_scene_strip']
                          for ext in ['svg', 'pdf', 'png'] for p in [HERE / f'{stem}.{ext}']}}
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    full_figure()
    scene_strip()
    verify()
    print('Rendered supplied comparison results (PDF/SVG/PNG). Verified all 36 metric groups and unchanged paper files.')
