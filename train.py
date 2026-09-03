import logging
import os
import re
from hashlib import sha256
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def underdeep_experiment_code(experiment_name: str) -> str:
    """Return a stable Underdeep-compatible code for a display name.

    Underdeep experiment codes permit only 3--500 Latin letters, digits,
    underscores, and hyphens. Keep a valid configured name unchanged; append
    a digest after normalization to avoid collisions such as ``data/a`` and
    ``data-a``.
    """
    if re.fullmatch(r"[A-Za-z0-9_-]{3,500}", experiment_name):
        return experiment_name

    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", experiment_name).strip("-_")
    normalized = normalized or "experiment"
    normalized = normalized[:487]
    return f"{normalized}-{sha256(experiment_name.encode()).hexdigest()[:12]}"


class UnderdeepCallback(TrainerCallback):
    """Forward the metrics emitted by ``transformers.Trainer`` to Underdeep."""

    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.run.log(logs, step=state.global_step)
        return control


def underdeep_init(cfg, run_name: str, experiment_name: str, log_dir: str):
    import underdeep

    experiment_code = underdeep_experiment_code(experiment_name)

    config_dict = OmegaConf.to_container(
        cfg,
        # The nested trainer configuration contains relative interpolations.
        # Resolving the whole tree here makes those references recursive.
        resolve=False,
        throw_on_missing=False,
    )
    config_dict["log_dir"] = log_dir
    config_dict["underdeep_run_name"] = run_name
    config_dict["underdeep_experiment"] = experiment_name
    config_dict["underdeep_experiment_code"] = experiment_code

    # A run can only be created inside an existing experiment. Creating with
    # exist_ok also makes concurrent launch attempts safe.
    underdeep.Client(project=cfg.underdeep_project).experiments.add(
        code=experiment_code,
        name=experiment_name,
        exist_ok=True,
    )

    return underdeep.init_run(
        project=cfg.underdeep_project,
        experiment=experiment_code,
        name=run_name[:127],
        parameters=config_dict,
        file_path=os.path.join(log_dir, "underdeep-metrics-{uid}.txt"),
    )


def get_checkpoint(output_dir):
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None

def get_total_devices():
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None:
        return int(world_size)
    return 1

def compute_accumulation_steps(train_batch_size, per_device_train_batch_size):
    total_devices = get_total_devices()
    # compute steps needed for gradient accumulation
    div = per_device_train_batch_size*total_devices
    steps = train_batch_size/div
    if not steps.is_integer():
        raise ValueError(
            "train_batch_size must be divisible by "
            f"per_device_batch*total_devices={div}"
        )
    return int(steps)


@hydra.main(config_path="cfgs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "LOCAL_RANK" in os.environ:
        is_main_process = int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        is_main_process = int(os.environ["RANK"]) == 0
    else:
        is_main_process = True

    
    if OmegaConf.is_missing(cfg, "gradient_accumulation_steps"):
        accumulation_steps = compute_accumulation_steps(
            train_batch_size=cfg.train_batch_size,
            per_device_train_batch_size=cfg.per_device_train_batch_size)
        cfg.gradient_accumulation_steps = accumulation_steps


    logger.info(f"Accumulation steps {cfg.gradient_accumulation_steps} ----")

    underdeep_run = None
    if cfg.underdeep_enabled and is_main_process:
        underdeep_run = underdeep_init(
            cfg=cfg,
            experiment_name=cfg.underdeep_experiment,
            run_name=cfg.underdeep_run_name,
            log_dir=cfg.output_dir,
        )
    
    try:
        tokenizer = hydra.utils.instantiate(cfg.make_tokenizer_fn)

        datasets = hydra.utils.instantiate(cfg.make_dataset_fn, tokenizer=tokenizer)

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            **datasets,
        )
        if underdeep_run is not None:
            trainer.add_callback(UnderdeepCallback(underdeep_run))
    except BaseException:
        if underdeep_run is not None:
            underdeep_run.finish()
        raise
    
    try:
        last_checkpoint = get_checkpoint(cfg.output_dir)
        if not last_checkpoint and cfg.resume_from is not None:
            last_checkpoint = get_checkpoint(cfg.resume_from)
    except BaseException:
        if underdeep_run is not None:
            underdeep_run.finish()
        raise
    if last_checkpoint:
        logger.info("Found checkpoint, resuming training run from "
                    f"{last_checkpoint}.")
    else:
        logger.info("No existing checkpoint, initializing new model")
    
    logger.info(f"Training  {datetime.now()}")
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info(f"Training complete {datetime.now()}")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        if cfg.save_final_model:
            logger.info(f"Saving final model at {cfg.output_dir}")
            trainer.model.config.use_cache = True
            trainer.save_model(cfg.output_dir)
            tokenizer.save_pretrained(cfg.output_dir)
            logger.info(f"Done saving {datetime.now()}")

        if is_main_process and cfg.push_to_hub:
            tags = cfg.tags if cfg.tags is not None else []
            trainer.create_model_card({"tags": tags})
        if cfg.push_to_hub:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

        if is_main_process and cfg.call_post_training is not None:
            # used to optionally run processes such as uploading
            # models/results/
            hydra.utils.instantiate(cfg.call_post_training)
    finally:
        if underdeep_run is not None:
            underdeep_run.finish()


if __name__ == "__main__":
    main()
