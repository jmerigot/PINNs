from typing import Any, Dict, List, Optional, Tuple

import sys
import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstorch.utils.load_data(root_path, "burgers_shock.mat")
    exact_u = np.real(data["usol"])
    return {"u": exact_u}

def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):   
    """Define the partial differential equations (PDEs)."""

    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    outputs["f"] = u_t + outputs["u"] * u_x - (0.01 / np.pi) * u_xx

    return outputs


@hydra.main(config_path="/data/home/jmerigot/PINN/configs_pinn/", config_name="config_burgers.yaml", version_base=None)
def main(cfg: DictConfig):
    # Your training code here
    pinnstorch.utils.extras(cfg)
    metric_dict, _ = pinnstorch.train(cfg, read_data_fn=read_data_fn, pde_fn=pde_fn, output_fn=None)
    metric_value = pinnstorch.utils.get_metric_value(metric_dict=metric_dict, metric_names=cfg.optimized_metric)
    return metric_value

if __name__ == "__main__":
    # This prevents Hydra from parsing Jupyter's command-line arguments
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = sys.argv[:1]
    
    main()
