import importlib
import pkgutil

def load_arch(cfg):
    arch_name = cfg.arch_name       # e.g. "VECNetPlus_arch"
    base_pkg = "models.archs"
    candidate_classes = ["BaselineArch"]  # indicate the valid class name in the `VECNetPlus_arch.py`

    # walk through all submodules under models.archs (including subdirs)
    for module_info in pkgutil.walk_packages(
        importlib.import_module(base_pkg).__path__,
        prefix=f"{base_pkg}."
    ):
        mod_name = module_info.name  # e.g. "models.archs.xxx"

        # match file name (module name)
        if mod_name.endswith(arch_name):
            module = importlib.import_module(mod_name)

            # find Baseline automatically
            for cls_name in candidate_classes:
                if hasattr(module, cls_name):
                    return getattr(module, cls_name)

            raise ImportError(
                f"Module '{mod_name}' found. BaselineArch is not defined."
            )

    raise ImportError(f"Arch '{arch_name}' not found under {base_pkg}")