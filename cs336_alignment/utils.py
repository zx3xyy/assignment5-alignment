import os
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

def setup_experiment(cfg: Any) -> logging.Logger:
    """
    Sets up the experiment directory, saves the configuration, and configures logging.
    
    Args:
        cfg: A configuration object (e.g., a dataclass) that must have 'output_dir' and 'exp_name' attributes.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Setup experiment directory
    exp_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    return logging.getLogger()
