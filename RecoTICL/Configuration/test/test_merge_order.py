#!/usr/bin/env python3
"""The layer clusters merged on the device are in the order of the merged layer clusters of the host, offline and at
HLT: a CLUEstering trackster indexes its layer clusters in the device merge, and the host collections must give the
same layer clusters at the same indices."""

import sys

from RecoTICL.Configuration.target import TARGETS


def _hlt_module(label):
    # the cfi file of an HLT module is not always named after its label
    import glob, os, re
    import HLTrigger.Configuration.HLT_75e33.modules as modules
    for path in sorted(glob.glob(os.path.join(modules.__path__[0], "*_cfi.py"))):
        with open(path) as f:
            if re.search(r"^%s\s*=" % label, f.read(), re.M):
                name = "HLTrigger.Configuration.HLT_75e33.modules." + os.path.basename(path)[:-3]
                return getattr(__import__(name, fromlist=[label]), label)
    raise LookupError("no HLT_75e33 module cfi defines %s" % label)


def _offline_module(label):
    import RecoTICL.LayerClustering.hgcalLayerClusters_cff as cff
    return getattr(cff, label)


def _label(tag):
    return tag.getModuleLabel() if hasattr(tag, "getModuleLabel") else str(tag).split(":")[0]


def check(name, get, host_merge, device_merge):
    host = [_label(tag) for tag in get(host_merge).layerClusters]
    device = [_label(tag) for tag in get(device_merge).hgcalLayerClustersSoA]
    # each host collection is the conversion of the device collection at the same position
    converted = [_label(get(label).src) for label in host]
    if converted != device:
        print("FAIL [%s]: %s merges %s, converted from %s; %s merges %s"
              % (name, host_merge, host, converted, device_merge, device))
        return 1
    print("OK [%s]: %s and %s merge %s in the same order" % (name, host_merge, device_merge, device))
    return 0


def main():
    rc = check("offline", _offline_module, "hgcalMergeLayerClusters", "hgcalSoAMergeLayerClusters")
    hlt = TARGETS["HLT"]
    rc |= check("HLT", _hlt_module, hlt.merged_lc, hlt.merged_lc_soa)
    return rc


if __name__ == "__main__":
    sys.exit(main())
