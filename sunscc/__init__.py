__version__ = "0.1.0"

from sunscc import dataset, transforms, model, plot
from sunscc import conf
from sunscc.utils import git_info
from omegaconf import OmegaConf
import platform

OmegaConf.register_new_resolver(
    "host", lambda: platform.node().split(".", 1)[0].lower(), replace=True
)
OmegaConf.register_new_resolver(
    "uname", lambda x: getattr(platform.uname(), x), replace=True
)
OmegaConf.register_new_resolver("git", git_info, replace=True)


__all__ = ["dataset", "transforms", "model", "conf", "plot"]
