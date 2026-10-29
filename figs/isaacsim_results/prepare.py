"""Freeze verified Isaac Sim evidence; no simulator or model inference required."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import NormalDist

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'SimResults/astar_57500_oracle_goal'
EXT = ROOT / 'SimResults/astar_57500_oracle_goal_100'
ASSETS = HERE / 'assets'
SELECTION = [(1, 'episode_far20_s1_21', 'Hospital lobby'),
             (2, 'episode_far20_s2_01', 'Hospital corridor'),
             (3, 'episode_warehouse50_s3_12', 'Warehouse')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def wilson(k, n):
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


def poses_for(directory):
    files = sorted(directory.glob('robot_state_frame_*.json'))
    states = [read(p) for p in files]
    assert states and all(np.isfinite(s['position']).all() for s in states)
    frames = [s['frame_count'] for s in states]
    assert frames == sorted(set(frames))
    # Hash both file names and bytes so missing or reordered samples are detected.
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return states, digest.hexdigest()


def main():
    ASSETS.mkdir(exist_ok=True)
    manifest = read(BASE / 'configs/manifest.json')
    status = read(BASE / 'report/status.json')
    reported = read(BASE / 'report/episodes.json')
    aggregates = read(BASE / 'report/metrics.json')
    video_selection = read(EXT / 'videos/thirds person views/selection.json')
    assert status['complete'] and status['completed'] == manifest['requested'] == len(reported) == 150
    assert sha(BASE / 'configs/manifest.json') == status['manifest_sha256']
    report_rows = {(r['scenario'], r['episode']): r for r in reported}
    aggregate_rows = {r['scenario']: r for r in aggregates}
    scenes, rows, provenance = [], [], []
    for scene_number, display_episode, name in SELECTION:
        scenario = f'scenario{scene_number}'
        group = next(g for g in manifest['groups'] if g['scenario'] == scenario)
        summary = aggregate_rows[scenario]
        assert summary['complete'] and summary['scenario_validated'] and summary['config_matches_manifest']
        # Bind each displayed score to the validated frozen configuration.
        config = Path(group['config'])
        assert sha(config) == group['config_sha256']
        configuration = yaml.safe_load(config.read_text())
        assert group['checkpoint']['goal_modality'] == 'oracle_relative_waypoint_diagnostic'
        tracks, result_rows = [], []
        for episode in group['episodes']:
            path = BASE / 'results' / scenario / f'{episode}.json'
            raw = read(path)
            report = report_rows[(scenario, episode)]
            assert sha(path) == report['source_sha256']
            assert raw['episode'] == episode and not raw.get('error')
            assert raw['duration_frames'] > 0 and raw['measurement_frames'] == raw['duration_frames']
            assert raw['measurement_version'] == 'physx_capsules_fov_v1'
            assert raw['success'] == report['success'] and raw['contact_collision'] == report['collision']
            states, pose_digest = poses_for(path.with_suffix('').with_name(episode + '_evidence'))
            tracks.append({'episode': episode, 'success': raw['success'],
                           'contact': raw['contact_collision'],
                           'frames': [s['frame_count'] for s in states],
                           'world_xy_m': [s['position'][:2] for s in states]})
            row = {key: report[key] for key in ['scenario', 'episode', 'success', 'collision',
                   'goal_progress', 'navigation_error_m', 'fov_clearance_collision_zero_m']}
            rows.append(row)
            result_rows.append(row)
            provenance.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                               'pose_files': len(states), 'pose_sequence_sha256': pose_digest})
        first = read(BASE / 'results' / scenario / f"{group['episodes'][0]}.json")
        first_states, _ = poses_for(BASE / 'results' / scenario / f"{group['episodes'][0]}_evidence")
        w, x, y, z = first_states[0]['quaternion']
        yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        # One rigid frame per scene; do not realign individual trial trajectories.
        frame = [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        k = sum(r['success'] for r in result_rows)
        contacts = sum(r['collision'] for r in result_rows)
        assert (len(result_rows), k, contacts) == (summary['completed'], summary['success_count'], summary['collision_count'])
        rerun_root = EXT / 'random_video_reruns/results' / scenario
        rerun = read(rerun_root / f'{display_episode}.json')
        assert rerun['success'] and not rerun['contact_collision']
        assert display_episode in [r['episode'] for r in video_selection['scenarios'][scenario]]
        pred_root = rerun_root / 'predictions' / display_episode
        # Avoid startup noise; the corridor target is occluded until observation 16.
        observation = 16 if scene_number == 2 else 8
        source_image = pred_root / f'prediction_{observation:05d}_third_person.jpg'
        source_prediction = pred_root / f'prediction_{observation:05d}.json'
        image_name = f'scene_{scene_number}_third_person.jpg'
        shutil.copy2(source_image, ASSETS / image_name)
        image_record = read(source_prediction)
        provenance.append({'path': str(source_image.relative_to(ROOT)), 'sha256': sha(source_image),
                           'prediction': str(source_prediction.relative_to(ROOT)),
                           'prediction_sha256': sha(source_prediction),
                           'frame': image_record['frame'], 'policy_observation_index': observation,
                           'selection': 'Earliest episode ID among available seeded-random success reruns.'})
        scene = {'scenario': scenario, 'number': scene_number, 'name': name,
                 'image': f'assets/{image_name}', 'display_episode': display_episode,
                 'image_frame': image_record['frame'],
                 'goal_marker_uv': {1: [.516, .18], 2: [.50, .18], 3: [.62, .20]}[scene_number],
                 'goal_marker_note': 'Manual image annotation of the target person; not a calibrated projection.',
                 'start_world_xyz_m': first['start'], 'goal_world_xyz_m': first['goal'],
                 'world_to_start_frame': frame, 'tracks': tracks,
                 'goal_radius_m': configuration.get('success_threshold', 1.5),
                 'n': len(result_rows), 'success_count': k, 'contact_count': contacts,
                 'success_wilson95': wilson(k, len(result_rows)),
                 'contact_wilson95': wilson(contacts, len(result_rows)),
                 'mean_fov_human_clearance_contact_zero_m': summary['mean_minimum_fov_clearance_collision_zero_m']}
        scenes.append(scene)
    write(HERE / 'evidence.json', {'scenes': scenes})
    with (HERE / 'trial_outcomes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    planned_extension = read(EXT / 'configs/manifest.json')
    available_extension = sum((EXT / 'results' / g['scenario'] / f'{e}.json').exists()
                              for g in planned_extension['groups'] for e in g['episodes'])
    protected = ROOT / 'GEAR-Nav Overleaf/fig/astar_realworld.pdf'
    write(HERE / 'provenance.json', {
        'protocol': manifest['protocol'], 'checkpoint_step': 57500, 'refinement_rounds': 14,
        'goal_conditioning': 'Exact simulator goal transformed into the robot frame (oracle-goal diagnostic).',
        'metric_trials': 150, 'trials_per_scene': 50,
        'planned_additional_trials': planned_extension['requested'], 'available_additional_trials': available_extension,
        'stale_summary_ignored': str((EXT / 'report/cumulative_summary.json').relative_to(ROOT)),
        'stale_summary_reason': 'Lists 113 completed trials, but all 150 validated base results now exist; no extension results exist.',
        'qualitative_reruns_excluded_from_statistics': True,
        'scene_selection': status['selection_disclosure'],
        'repetition_unit': 'Fixed scene geometry with paired pedestrian initial-idle timings, not independent scenes.',
        'interval': 'Two-sided nominal 95% Wilson score interval, z = NormalDist().inv_cdf(0.975). Descriptive and conditional on this timing protocol.',
        'significance': 'No baseline significance claim: no matched baseline trials are included in this figure.',
        'clearance': 'FOV human-capsule surface gap; contact episodes contribute zero. Not all-obstacle clearance.',
        'trajectory': 'Recorded robot states every 10 simulator frames. Polyline joins samples; no smoothing. Last saved pose may precede termination.',
        'coordinates': 'One rigid transform per scene, fixed from first metric trial heading; forward/left metres relative to configured start.',
        'image_processing': 'Original recorded third-person JPEGs; no generation, retouching, or color adjustment.',
        'video_selection_sha256': sha(EXT / 'videos/thirds person views/selection.json'),
        'manifest_sha256': sha(BASE / 'configs/manifest.json'),
        'report_episodes_sha256': sha(BASE / 'report/episodes.json'),
        'protected_realworld_pdf': {'path': str(protected.relative_to(ROOT)), 'sha256': sha(protected)},
        'sources': provenance,
    })
    print('Verified and froze 150 metric trials, 150 recorded trajectories, and three third-person frames.')


if __name__ == '__main__':
    main()
