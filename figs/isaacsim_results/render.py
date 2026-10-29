"""Render editable Isaac Sim figure masters from frozen measured evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gear_sim_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STYLE = json.loads((HERE / 'style.json').read_text())
DATA = json.loads((HERE / 'evidence.json').read_text())
plt.rcParams.update({'font.family': STYLE['font_family'], 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'gear-isaacsim-results',
                     'text.color': STYLE['ink'], 'axes.labelcolor': STYLE['muted'],
                     'xtick.color': STYLE['muted'], 'ytick.color': STYLE['muted'],
                     'savefig.facecolor': 'white'})
LEFT, STEP, WIDTH = .019, .328, .306


def interval(values):
    return f'{100*values[0]:.1f}–{100*values[1]:.1f}%'


def export(fig, stem):
    for ext in ('svg', 'pdf', 'png'):
        metadata = {'Creator': 'GEAR-Nav: recorded Isaac Sim poses and validated 150-trial results'}
        if ext in ('svg', 'pdf'):
            metadata['Date' if ext == 'svg' else 'CreationDate'] = None
        fig.savefig(HERE / f'{stem}.{ext}', dpi=220, metadata=metadata)
        if ext == 'svg':
            path = HERE / f'{stem}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def image_panel(fig, scene, x, bottom, height):
    ax = fig.add_axes([x, bottom, WIDTH, height])
    image = Image.open(HERE / scene['image'])
    ax.imshow(image, extent=(0, 1, 1, 0), aspect='auto')
    ax.set_axis_off()
    # The manually located target marker is an annotation, never a projected pose.
    u, v = scene['goal_marker_uv']
    ax.scatter([u], [v], s=170, marker='*', c=STYLE['goal'],
               edgecolors=STYLE['ink'], linewidths=.8, zorder=4)
    ax.text(u+.045, v+.012, 'Goal', fontsize=15, color='white', va='center',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#333333')])
    ax.text(.035, .95, 'GEAR-Nav', va='bottom', fontsize=14, color='white',
            bbox={'facecolor': '#222222', 'alpha': .72, 'edgecolor': 'none', 'pad': 3})
    return ax


def transform(scene, xy):
    return (np.array(xy) - np.array(scene['start_world_xyz_m'][:2])) @ np.array(scene['world_to_start_frame']).T


def trajectory_panel(fig, scene, x):
    ax = fig.add_axes([x+.037, .326, .259, .209])
    ax.set_facecolor('#FBFBFC')
    ax.set_axisbelow(True)
    ax.grid(color='#E4E6E9', lw=.6)
    tracks = scene['tracks']
    for track in tracks:
        xy = transform(scene, track['world_xy_m'])
        line, = ax.plot(xy[:, 0], xy[:, 1], color=STYLE['path'], alpha=.16, lw=1.0)
        line.set_gid(f"{scene['scenario']}-{track['episode']}-recorded-poses")
    selected = next(t for t in tracks if t['episode'] == scene['display_episode'])
    xy = transform(scene, selected['world_xy_m'])
    ax.plot(xy[:, 0], xy[:, 1], color=STYLE['path'], lw=2.5, zorder=4)
    for track in tracks:
        if track['contact']:
            xy = transform(scene, track['world_xy_m'])
            ax.plot(xy[:, 0], xy[:, 1], color=STYLE['contact'], lw=1.9, zorder=5)
            ax.scatter(*xy[-1], marker='X', s=60, c=STYLE['contact'], edgecolors='white', linewidths=.6, zorder=8)
            ax.annotate('Trial 24: contact', xy[-1], xytext=(13, 20), textcoords='offset points',
                        fontsize=12, color=STYLE['contact'],
                        arrowprops={'arrowstyle': '-', 'lw': .7, 'color': STYLE['contact']})
    goal = transform(scene, [scene['goal_world_xyz_m'][:2]])[0]
    region = Circle(goal, scene['goal_radius_m'], facecolor=STYLE['goal'],
                    edgecolor='#B28A2B', alpha=.13, lw=1, linestyle='--', zorder=1)
    ax.add_patch(region)
    ax.scatter(*goal, marker='*', s=160, c=STYLE['goal'], edgecolors=STYLE['ink'], linewidths=.7, zorder=7)
    ax.scatter([0], [0], marker='o', s=37, c=STYLE['ink'], edgecolors='white', linewidths=.7, zorder=7)
    ax.set(xlim=(-1, 10.5), ylim=(-3.5, 3.5), xticks=[0, 2, 4, 6, 8, 10], yticks=[-2, 0, 2])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Forward (m)', fontsize=14, labelpad=2)
    ax.set_ylabel('Left (m)', fontsize=14, labelpad=2)
    ax.tick_params(labelsize=12, length=2.5, pad=2)
    for spine in ax.spines.values():
        spine.set_color('#D2D5DA')
    return ax


def summary_panel(fig, scene, x):
    box = FancyBboxPatch((x, .064), WIDTH, .160, transform=fig.transFigure,
                          boxstyle='round,pad=0.004,rounding_size=0.008',
                          linewidth=.8, edgecolor=STYLE['panel_edge'], facecolor=STYLE['panel_fill'], zorder=-1)
    fig.add_artist(box)
    n, k, contacts = scene['n'], scene['success_count'], scene['contact_count']
    fig.text(x+.013, .191, f'{100*k/n:.0f}%', fontsize=27, weight='bold', va='center')
    fig.text(x+.119, .192, f'{k}/{n} successes', fontsize=17, va='center')
    fig.text(x+.013, .153, f'95% CI  {interval(scene["success_wilson95"])}', fontsize=14)
    fig.text(x+.013, .119, f'Contacts: {contacts}/{n}  |  95% CI  {interval(scene["contact_wilson95"])}', fontsize=12.5)
    ax = fig.add_axes([x+.013, .073, WIDTH-.026, .030])
    for i, t in enumerate(scene['tracks']):
        ax.scatter(i % 25, 1-i//25, marker='X' if t['contact'] else 'o', s=14 if t['contact'] else 8,
                   color=STYLE['contact'] if t['contact'] else STYLE['path'], linewidths=.3)
    ax.set(xlim=(-.7, 24.7), ylim=(-.7, 1.7))
    ax.set_axis_off()


def full_figure():
    fig = plt.figure(figsize=(12.4, 8.8))
    fig.text(LEFT, .977, 'Isaac Sim evaluation', fontsize=23, weight='bold', va='top')
    fig.text(.981, .974, 'GEAR-Nav · oracle-goal diagnostic', fontsize=18, ha='right', va='top')
    fig.text(LEFT, .930, 'Three fixed scenes · 50 pedestrian timings per scene · 150 completed metric trials',
             fontsize=16, color=STYLE['muted'])
    for i, scene in enumerate(DATA['scenes']):
        x = LEFT + i*STEP
        fig.text(x+WIDTH/2, .897, f"Scenario {scene['number']} · {scene['name']}",
                 ha='center', va='top', fontsize=17)
        h = WIDTH * 12.4/8.8 * 9/16
        image_panel(fig, scene, x, .599, h)
        trajectory_panel(fig, scene, x)
        summary_panel(fig, scene, x)
    fig.text(LEFT, .565, 'Recorded robot trajectories', fontsize=17, weight='bold')
    fig.text(.981, .565, 'Shared metric scale; all 50 trials shown in each scene',
             fontsize=14, color=STYLE['muted'], ha='right')
    handles = [Line2D([0], [0], color=STYLE['path'], lw=3, label='GEAR-Nav example'),
               Line2D([0], [0], color=STYLE['path'], alpha=.3, lw=3, label='Other metric trials'),
               Line2D([0], [0], color=STYLE['contact'], lw=2, label='Trial with contact'),
               Line2D([0], [0], color=STYLE['ink'], marker='o', lw=0, ms=5, label='Start'),
               Line2D([0], [0], color=STYLE['goal'], marker='*', markeredgecolor=STYLE['ink'],
                      lw=0, ms=11, label='Goal (1.5 m radius)')]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, .266), ncol=5,
               frameon=False, fontsize=14, handlelength=1.5, columnspacing=1.5)
    fig.text(LEFT, .237, 'Repeated-trial outcomes', fontsize=17, weight='bold')
    fig.text(.981, .237, 'Nominal 95% Wilson intervals', fontsize=14, color=STYLE['muted'], ha='right')
    fig.text(.5, .022, 'One dot per metric trial; red × denotes contact. Images are separate success reruns, excluded from the statistics.',
             fontsize=13, color=STYLE['muted'], ha='center')
    export(fig, 'isaacsim_results')


def scene_strip():
    """A compact standalone scene row for a possible later real/sim composition."""
    fig = plt.figure(figsize=(12.4, 4.15))
    for i, scene in enumerate(DATA['scenes']):
        x = LEFT+i*STEP
        fig.text(x+WIDTH/2, .972, f"Scenario {scene['number']}", fontsize=23, ha='center', va='top')
        fig.text(x+WIDTH/2, .872, scene['name'], fontsize=17, ha='center')
        image_panel(fig, scene, x, .255, WIDTH*12.4/4.15*9/16)
        fig.text(x+WIDTH/2, .181, f"{scene['success_count']}/{scene['n']} successes  ·  {scene['contact_count']}/{scene['n']} contacts",
                 fontsize=18, ha='center')
        fig.text(x+WIDTH/2, .100, f"Success 95% CI: {interval(scene['success_wilson95'])}",
                 fontsize=15, ha='center', color=STYLE['muted'])
    fig.text(.5, .020, 'Isaac Sim · GEAR-Nav with oracle goals · 50 metric trials per scene · separate third-person success reruns',
             fontsize=14, ha='center', color=STYLE['muted'])
    export(fig, 'isaacsim_scene_strip')


def verify():
    provenance = json.loads((HERE/'provenance.json').read_text())
    protected = provenance['protected_realworld_pdf']
    pdf = HERE.parents[1] / protected['path']
    unchanged = hashlib.sha256(pdf.read_bytes()).hexdigest() == protected['sha256']
    assert unchanged, 'The protected real-world PDF changed.'
    assert sum(s['n'] for s in DATA['scenes']) == 150
    assert [s['success_count'] for s in DATA['scenes']] == [50, 49, 50]
    assert [s['contact_count'] for s in DATA['scenes']] == [0, 1, 0]
    for scene in DATA['scenes']:
        assert len(scene['tracks']) == scene['n']
        assert all(len(t['world_xy_m']) == len(t['frames']) > 1 for t in scene['tracks'])
        xy = np.concatenate([transform(scene, t['world_xy_m']) for t in scene['tracks']])
        assert xy[:, 0].min() > -.75 and xy[:, 0].max() < 10
        assert xy[:, 1].min() > -3 and xy[:, 1].max() < 3
    writeup = {'metric_trials': 150, 'trajectory_count': 150, 'successes': 149, 'contact_trials': 1,
               'realworld_pdf_sha256_unchanged': unchanged, 'all_trajectory_samples_in_plot_limits': True,
               'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for stem in ['isaacsim_results', 'isaacsim_scene_strip']
                           for ext in ['svg', 'pdf', 'png'] for p in [HERE/f'{stem}.{ext}']}}
    (HERE/'verification.json').write_text(json.dumps(writeup, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    full_figure()
    scene_strip()
    verify()
    print('Rendered SVG/PDF/PNG masters and verified 150 measured trajectories. Real-world PDF unchanged.')
