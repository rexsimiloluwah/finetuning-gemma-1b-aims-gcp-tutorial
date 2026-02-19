import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    """
    Simple experiment logger.
    Logs hyperparameters, LoRA configs, losses, and metrics.
    Can log to console and TensorBoard.
    """
    def __init__(self, cfg, log_dir=None):
        self.cfg = cfg
        self.log_dir = log_dir or os.path.join(cfg.training.output_dir, cfg.experiment_id, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Log experiment config once
        self.log_config()

    def log_config(self):
        print("📋 Experiment Configuration:")
        for section, params in self.cfg.items():
            print(f"--- {section} ---")
            if isinstance(params, dict):
                for k, v in params.items():
                    print(f"{k}: {v}")
            else:
                print(params)
        # Optionally write config to a file
        config_path = os.path.join(self.log_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(str(self.cfg))

    def log_metrics(self, metrics: dict, step: int):
        """
        metrics: dict of metric_name -> value
        step: training step or epoch
        """
        print(f"Step {step} | ", " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        self.writer.close()