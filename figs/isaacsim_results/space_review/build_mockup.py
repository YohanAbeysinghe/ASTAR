"""Print-size compact simulation results proposal; never modifies the paper."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gear_space_review'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
DATA = json.loads((SOURCE / 'evidence.json').read_text())
STYLE = json.loads((SOURCE / 'style.json').read_text())
plt.rcParams.update({'font.family': STYLE['font_family'], 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'gear-compact-result-proposal',
                     'text.color': '#1A1A1A'})


def main():
    width = 245.71811 / 72.27  # The compiled paper's actual column width, inches.
    height = 2.45
    fig = plt.figure(figsize=(width, height))
    fig.text(.012, .976, 'Illustrative recorded views', fontsize=7, va='top', color='#555555')
    for i, view in enumerate(DATA['illustrative_views']):
        x, w = .012+i*.329, .318
        h = w*width/height*9/16
        ax = fig.add_axes([x, .655, w, h])
        with Image.open(SOURCE / view['image']) as image:
            ax.imshow(image, aspect='auto')
        ax.set_axis_off()
        fig.text(x+w/2, .620, view['name'], fontsize=6.8, ha='center')
    ax = fig.add_axes([0, 0, 1, 1], zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    method_x, first_metric, cell = .013, .228, .084
    ax.plot([.012, .988], [.580, .580], color='#333333', lw=.65)
    ax.text(method_x, .491, 'Method', fontsize=7.7, weight='bold', va='center')
    for i, scene in enumerate(DATA['scenarios']):
        xs = [first_metric+(i*3+j)*cell for j in range(3)]
        ax.text(xs[1], .548, f'Scenario {scene["number"]}', fontsize=7.8, ha='center', va='center')
        ax.plot([xs[0]-.036, xs[2]+.036], [.525, .525], color='#999999', lw=.4)
        for x, label in zip(xs, ['S ↑', 'C ↓', 'Cl ↑']):
            ax.text(x, .491, label, fontsize=7.7, ha='center', va='center')
    ax.plot([.012, .988], [.461, .461], color='#555555', lw=.5)
    ys = [.420, .350, .280, .210]
    ax.add_patch(Rectangle((.012, .175), .976, .068, facecolor='#FFF0CC', edgecolor='none', zorder=-1))
    for y, model in zip(ys, DATA['models']):
        weight = 'bold' if model == 'GEAR-Nav' else 'normal'
        ax.text(method_x, y, model, fontsize=7.6, weight=weight, va='center')
        for i, scene in enumerate(DATA['scenarios']):
            m = scene['metrics'][model]
            values = [f'{100*m["success"]:.0f}', f'{100*m["collision"]:.0f}', f'{m["clearance_m"]:.2f}']
            for j, value in enumerate(values):
                label = ax.text(first_metric+(i*3+j)*cell, y, value, fontsize=7.8,
                                ha='center', va='center', weight=weight)
                label.set_gid(f'scene-{i+1}-{model}-{j}')
    ax.plot([.012, .988], [.172, .172], color='#333333', lw=.65)
    fig.text(.012, .123, 'S: success (%); C: collision (%); Cl: clearance (m).', fontsize=6.8)
    fig.text(.012, .071, 'Baselines: VEGA. GEAR-Nav: updated simulation table.', fontsize=6.8)
    fig.text(.012, .019, 'Images are separate recordings; protocol matching is pending.', fontsize=6.5, color='#555555')
    for ext in ['pdf', 'svg', 'png']:
        path = HERE / f'compact_proposal.{ext}'
        metadata = {'Creator': 'GEAR-Nav compact layout proposal'}
        if ext in ['pdf', 'svg']:
            metadata['CreationDate' if ext == 'pdf' else 'Date'] = None
        fig.savefig(path, dpi=360, metadata=metadata)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    root = HERE.parents[2]
    for path, digest in json.loads((HERE/'protected_files.json').read_text()).items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest, path
    text_width = 505.89/72.27
    full_height = text_width*9.4/12.4
    full_area = text_width*full_height
    metrics = {
        'submission_proof_pages': 8, 'column_width_in': width,
        'text_width_in': text_width, 'text_height_in': 684/72.27,
        'current_bar_figure_at_full_width': {'width_in': text_width, 'height_in': full_height, 'area_sq_in': full_area},
        'compact_proposal': {'width_in': width, 'height_in': height, 'area_sq_in': width*height},
        'figure_area_reduction_percent': 100*(1-width*height/full_area),
        'comparison_excludes_external_captions_and_float_spacing': True,
        'pipeline_image_height_in': text_width*458.92/504.57,
        'model_image_height_in': width*357.41/244.8,
        'teaser_image_height_in': text_width*222.97/504.57,
        'realworld_image_height_in': width*518.4/892.8,
        'paper_files_unchanged': True,
    }
    (HERE/'space_measurements.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
