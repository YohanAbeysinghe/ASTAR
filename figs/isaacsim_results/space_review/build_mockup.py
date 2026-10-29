"""Print-size qualitative scene strip; evaluations remain in the paper table."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gear_space_review'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MarkerPath
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
DATA = json.loads((SOURCE / 'evidence.json').read_text())
STYLE = json.loads((SOURCE / 'style.json').read_text())
GOALS = json.loads((HERE / 'goal_annotations.json').read_text())['views']
# Same five-point geometry and gold fill as teaser/compact/build_compact.py.
theta = np.arange(10)*np.pi/5 + np.pi/2
radii = np.where(np.arange(10)%2 == 0, 1., .43)
star_vertices = np.column_stack([np.cos(theta)*radii, np.sin(theta)*radii])
GOAL_STAR = MarkerPath(np.vstack([star_vertices, star_vertices[0]]), closed=True)
plt.rcParams.update({'font.family': STYLE['font_family'], 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'gear-compact-result-proposal',
                     'text.color': '#1A1A1A'})


def main():
    width = 245.71811 / 72.27  # The compiled paper's actual column width, inches.
    height = .85
    fig = plt.figure(figsize=(width, height))
    for i, view in enumerate(DATA['illustrative_views']):
        x, w = .012+i*.329, .318
        h = w*width/height*9/16
        ax = fig.add_axes([x, .235, w, h])
        with Image.open(SOURCE / view['image']) as image:
            ax.imshow(image, aspect='auto')
        ax.set_axis_off()
        goal = GOALS[i]
        assert goal['image'] == view['image']
        assert hashlib.sha256((SOURCE / view['image']).read_bytes()).hexdigest() == goal['image_sha256']
        u, v = goal['goal_marker_uv_top_left']
        ax.plot(u, 1-v, transform=ax.transAxes, linestyle='none', marker=GOAL_STAR,
                markersize=8, markerfacecolor='#FFC52F', markeredgecolor='#242424',
                markeredgewidth=.6, zorder=5, gid=f'goal-star-{i+1}')
        fig.text(x+w/2, .080, f"({chr(97+i)}) {view['name']}", fontsize=7, ha='center')
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
        'qualitative_only': True,
        'evaluation_values_in_proposal': 0,
        'goal_annotations': 3,
        'evaluation_location': 'Existing paper table tab:isaac_sim',
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
