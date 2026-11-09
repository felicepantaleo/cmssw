# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""CPU / GPU (alpaka) backend selection.

pyTICL lets a user pick the compute backend per module, *where the module
supports it*.  The validator already rejects a GPU request on a module that has
no alpaka implementation (see :mod:`RecoTICL.Configuration.validator`); this module
implements the mechanism that actually drives a portable (``@alpaka``) module to
CPU (``serial_sync``) or GPU (``cuda_async``) and wires up
``ProcessAcceleratorAlpaka`` on the process.

The portable modules are the HGCAL SoA rechit and layer clustering producers, the
CLUEstering iteration, an iteration on the device and the linking on the device. A
device module reads device products of its own backend only, so an iteration on the
device follows the backend of the iteration it masks from, and the linking on the
device follows the backend of the collections it links, which must agree.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CPU, GPU
from RecoTICL.Configuration.assembler import iteration_labels, device_links_labels

# pyTICL backend name -> alpaka backend string
ALPAKA_BACKEND = {CPU: "serial_sync", GPU: "cuda_async"}


def is_alpaka(module):
    """True if ``module`` is a portable ``@alpaka`` producer."""
    return module.type_().endswith("@alpaka")


def set_module_backend(module, backend):
    """Drive a portable ``@alpaka`` ``module`` to ``backend`` ('cpu' or 'gpu')."""
    if not is_alpaka(module):
        from RecoTICL.Configuration.model import PyTICLError
        raise PyTICLError(
            "module type %r is not a portable @alpaka module; it cannot run on a "
            "GPU backend" % module.type_())
    if backend not in ALPAKA_BACKEND:
        from RecoTICL.Configuration.model import PyTICLError
        raise PyTICLError("unknown backend %r (known: %s)"
                          % (backend, ", ".join(ALPAKA_BACKEND)))
    if not hasattr(module, "alpaka"):
        module.alpaka = cms.untracked.PSet()
    module.alpaka.backend = cms.untracked.string(ALPAKA_BACKEND[backend])
    return module


def add_process_accelerator(process, backend=None):
    """Add (idempotently) ``ProcessAcceleratorAlpaka`` to ``process`` and, if a
    ``backend`` is given, set it process-wide."""
    from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka import ProcessAcceleratorAlpaka
    if not hasattr(process, "ProcessAcceleratorAlpaka"):
        process.ProcessAcceleratorAlpaka = ProcessAcceleratorAlpaka()
    if backend is not None:
        process.ProcessAcceleratorAlpaka.setBackend(ALPAKA_BACKEND.get(backend, backend))
    return process.ProcessAcceleratorAlpaka


def iteration_backend(cfg, it):
    """The backend of the modules of iteration ``it``: its own, or for an iteration on the device the backend of the
    iteration it masks from."""
    if it.on_device and it.masks_from in cfg._by_name:
        return iteration_backend(cfg, cfg._by_name[it.masks_from])
    return it.backend


def device_links_backend(cfg):
    """The backend of the linking on the device: the backend of the collections it links. Raises when they differ."""
    from RecoTICL.Configuration.model import PyTICLError
    backends = {n: iteration_backend(cfg, cfg._by_name[n]) for n in cfg.links_spec.collections if n in cfg._by_name}
    if len(set(backends.values())) > 1:
        raise PyTICLError("the linking on the device reads collections on different backends: %s"
                          % ", ".join("%s on %s" % kv for kv in backends.items()))
    return next(iter(backends.values()), CPU)


def apply_backends(assembled, process):
    """After ``assembled.add_to_process(process)``: drive each portable module to
    its iteration's requested backend and ensure ``ProcessAcceleratorAlpaka`` is
    present.  CPU-only modules are left untouched (GPU on them is rejected by the
    validator)."""
    cfg = assembled.config
    requested = {}
    for it in cfg.iterations:
        for lbl in iteration_labels(it, cfg.target):
            requested[lbl] = iteration_backend(cfg, it)
    if cfg.links_spec and cfg.links_spec.on_device:
        for lbl in device_links_labels(cfg, cfg.target):
            requested[lbl] = device_links_backend(cfg)
    sc = cfg.superclustering_spec
    if sc is not None and sc.kind == "DNNHeterogeneous":
        # the superclustering DNN reads the device tracksters of its source iteration
        requested[cfg.target.supercluster_producer_label] = iteration_backend(cfg, cfg._by_name[sc.source])
    any_alpaka = False
    for label, module in assembled.modules.items():
        if is_alpaka(module):
            any_alpaka = True
            set_module_backend(module, requested.get(label, CPU))
    if any_alpaka:
        add_process_accelerator(process)
    return process
