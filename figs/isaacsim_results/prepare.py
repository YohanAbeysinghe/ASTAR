"""Freeze supplied simulation comparison results, without editing the paper."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / 'GEAR-Nav Overleaf'
MODELS = ('OmniVLA', 'TIC-VLA', 'NaVILA', 'GEAR-Nav')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def paper_values(text):
    block = text.split(r'\label{tab:isaac_sim}', 1)[1].split(r'\end{table}', 1)[0]
    pattern = r'(OmniVLA|TIC-VLA|NaVILA|\\textbf\{GEAR-Nav\})\s*&(.+?)\\\\'
    result = {}
    for label, body in re.findall(pattern, block, re.S):
        label = label.replace(r'\textbf{', '').replace('}', '')
        values = [float(cell.strip()) for cell in body.split('&')]
        assert len(values) == 9 and label not in result
        result[label] = [dict(zip(('success', 'collision', 'clearance_m'), values[i:i+3]))
                         for i in range(0, 9, 3)]
    assert tuple(result) == MODELS
    for scenes in result.values():
        for row in scenes:
            assert 0 <= row['success'] <= 1 and 0 <= row['collision'] <= 1
            assert row['clearance_m'] >= 0
    return result


def reference_values(text):
    # Only the active simulation table at the start of the reference manuscript.
    block = text.split(r'\begin{tabular}{llrrr}', 1)[1].split(r'\end{tabular}', 1)[0]
    block = re.sub(r'\\textbf\{([^}]+)\}', r'\1', block)
    result, scene = {}, None
    for line in block.splitlines():
        match = re.search(r'Scn([1-3])', line)
        if match:
            scene = int(match.group(1)) - 1
        match = re.search(r'&\s*(OmniVLA|TIC-VLA|NaVILA)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)', line)
        if match:
            assert scene is not None
            result[(match[1], scene)] = tuple(float(match[i]) for i in range(2, 5))
    assert len(result) == 9
    return result


def main():
    main_tex = PAPER / 'main.tex'
    reference = ROOT / 'Overleaf--VEGA/sec/5_results.tex'
    notes = PAPER / 'BASELINE_RESULTS.md'
    values = paper_values(main_tex.read_text())
    baseline_values = reference_values(reference.read_text())
    for model in MODELS[:-1]:
        for i, metrics in enumerate(values[model]):
            assert tuple(metrics.values()) == baseline_values[(model, i)]
    old = json.loads((HERE / 'oracle_goal_diagnostic/provenance.json').read_text())
    photo_sources = [s for s in old['sources'] if s['path'].endswith('_third_person.jpg')]
    assert len(photo_sources) == 3
    names = ('Hospital lobby', 'Hospital corridor', 'Warehouse')
    views = []
    for i, source in enumerate(photo_sources, 1):
        image = HERE / f'assets/scene_{i}_third_person.jpg'
        assert sha(image) == source['sha256']
        views.append({'image': str(image.relative_to(HERE)), 'name': names[i-1],
                      'sha256': sha(image), 'source': source['path'],
                      'source_prediction': source['prediction'], 'frame': source['frame'],
                      'role': 'Illustrative recorded view; not matched to the supplied aggregate results.'})
    evidence = {
        'source_table': 'tab:isaac_sim', 'models': list(MODELS),
        'scenarios': [{'number': i+1, 'metrics': {m: values[m][i] for m in MODELS}} for i in range(3)],
        'illustrative_views': views,
        'baseline_source': 'Previously reported VEGA simulation results, transcribed in the current paper.',
        'gear_source': 'User-supplied aggregate values in the current paper.',
        'baseline_trials_per_scenario': 50,
        'gear_trials_per_scenario': None,
        'confidence_intervals': None,
    }
    write(HERE / 'evidence.json', evidence)
    with (HERE / 'aggregate_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scenario', 'method', 'success_rate', 'collision_rate', 'clearance_m'], lineterminator='\n')
        writer.writeheader()
        for scene in evidence['scenarios']:
            for model, metrics in scene['metrics'].items():
                writer.writerow({'scenario': scene['number'], 'method': model,
                                 'success_rate': metrics['success'], 'collision_rate': metrics['collision'],
                                 'clearance_m': metrics['clearance_m']})
    protected = ['main.tex', 'supplementary.tex', 'BASELINE_RESULTS.md', 'fig/astar_realworld.pdf']
    write(HERE / 'provenance.json', {
        'result_source': 'GEAR-Nav Overleaf/main.tex, table tab:isaac_sim',
        'paper_commit_at_extraction': subprocess.check_output(['git', '-C', str(PAPER), 'rev-parse', 'HEAD'], text=True).strip(),
        'paper_main_sha256': sha(main_tex), 'paper_provenance_notes_sha256': sha(notes),
        'reference_source': str(reference.relative_to(ROOT)), 'reference_sha256': sha(reference),
        'baseline_cells_crosschecked': 27, 'total_supplied_numeric_cells': 36,
        'gear_values': values['GEAR-Nav'],
        'gear_run_provenance': 'Not supplied in the source table; no checkpoint, trial count, goal modality, or per-trial logs are inferred.',
        'rates': 'Original 0–1 success and collision rates are multiplied by 100 for display as percentages.',
        'clearance': 'Meters, retaining the supplied table definition: average minimum clearance over non-colliding trials.',
        'uncertainty': 'No intervals, p-values, or trial counts are inferred for GEAR-Nav from rounded aggregate rates.',
        'protocol_comparability': 'The paper records that comparison protocol matching remains unverified; plotted values are descriptive.',
        'images': views,
        'old_diagnostic': 'oracle_goal_diagnostic/ contains the superseded 150-trial oracle-goal figure and evidence. Its trajectories and counts are excluded from this comparison.',
        'protected_paper_files': {str((PAPER / p).relative_to(ROOT)): sha(PAPER / p) for p in protected},
        'paper_repository_changes_authorized': False,
    })
    print('Extracted all 36 supplied values; independently matched all 27 baseline values to VEGA. Paper files unchanged.')


if __name__ == '__main__':
    main()
