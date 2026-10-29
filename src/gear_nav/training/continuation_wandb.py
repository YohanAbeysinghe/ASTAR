"""Start an independent W&B continuation without changing its source run."""

from __future__ import annotations

from pathlib import Path

import wandb


def init_continuation_wandb(
    run_dir: Path,
    config: dict,
    *,
    enabled: bool,
    source_run_path: str | None,
    source_optimizer_step: int,
    name: str,
    resume: bool,
) -> wandb.Run:
    """Create a linked run once; later calls resume its persisted, distinct ID.

    The caller supplies the source checkpoint in ``config`` and logs subsequent
    updates on the returned run with ``optimizer_step``. This function never
    resumes, rewinds, or forks the source run. Lineage is recorded explicitly in
    the continuation config so this works without W&B's private-preview forking.
    """
    if not enabled:
        return wandb.init(mode="disabled")
    if source_optimizer_step < 0:
        raise ValueError("source_optimizer_step must be nonnegative")

    entity = config.get("wandb_entity")
    project = config.get("project_name", "astar")
    source_id = None
    if source_run_path is not None:
        parts = source_run_path.strip("/").split("/")
        if len(parts) != 3 or not all(parts):
            raise ValueError("source_run_path must be entity/project/run_id")
        entity, project, source_id = parts
        source_run_path = "/".join(parts)

    run_dir = Path(run_dir)
    id_path = run_dir / "wandb_id.txt"
    run_config = {
        **config,
        "continuation_source_run": source_run_path,
        "continuation_source_optimizer_step": source_optimizer_step,
        "optimizer_step_start": source_optimizer_step,
    }
    if source_run_path is not None:
        run_config.update({
            "continuation_source_url": f"https://wandb.ai/{entity}/{project}/runs/{source_id}",
            "continuation_history_mode": "linked_run",
        })
    else:
        run_config["continuation_history_mode"] = "new_run"
    if resume:
        if not id_path.is_file():
            raise FileNotFoundError(f"Cannot resume continuation W&B; missing {id_path}")
        run_id = id_path.read_text().strip()
        if not run_id or run_id == source_id:
            raise ValueError("Continuation W&B ID must be nonempty and different from the source run ID")
        run = wandb.init(
            id=run_id,
            resume="must",
            reinit="create_new",
            entity=entity,
            project=project,
            dir=str(run_dir),
        )
        if run.id != run_id:
            raise RuntimeError("W&B returned a different run ID; refusing to attach continuation logging")
        run.config.update(run_config, allow_val_change=True)
    else:
        if id_path.exists():
            raise FileExistsError(f"Continuation W&B ID already exists at {id_path}; use resume=True")
        run_id = wandb.util.generate_id()
        if run_id == source_id:
            raise ValueError("Generated continuation ID equals the source run ID")
        kwargs = {
            "id": run_id,
            "reinit": "create_new",
            "name": name,
            "entity": entity,
            "project": project,
            "dir": str(run_dir),
            "config": run_config,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(**kwargs)
        if run.id != run_id:
            raise RuntimeError("W&B returned a different run ID; refusing to attach continuation logging")
        id_path.write_text(run.id + "\n")

    run.define_metric("optimizer_step")
    run.define_metric("*", step_metric="optimizer_step")
    if not resume:
        run.log({"optimizer_step": source_optimizer_step})
    return run
