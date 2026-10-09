#!/usr/bin/env python3
"""Liste les ports série USB (ttyACM*, ttyUSB*) : VID:PID, numéro de série, chemin USB physique.

  python3 scripts/detect_ports.py            liste les ports et les symlinks /dev/so101_*
  python3 scripts/detect_ports.py snapshot   mémorise l'état actuel (outputs/ports_snapshot.json)
  python3 scripts/detect_ports.py diff       compare à l'état mémorisé : port apparu ou disparu

Même méthode que lerobot-find-port (débrancher, comparer), en lecture seule de /sys, sur l'hôte.
Les deux bras SO-101 ont la même puce USB (mêmes VID:PID) : seul le branchement un par un,
ou le chemin USB physique, dit lequel est lequel.
"""
import glob
import json
import os
import sys

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "ports_snapshot.json")
KNOWN = {"1a86": "WCH (carte servo SO-101 probable)"}


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def usb_device_dir(tty):
    """Remonte de l'interface tty jusqu'au périphérique USB qui porte idVendor/idProduct/serial."""
    d = os.path.realpath(f"/sys/class/tty/{tty}/device")
    while d != "/" and not os.path.exists(os.path.join(d, "idVendor")):
        d = os.path.dirname(d)
    return None if d == "/" else d


def list_ports():
    ports = {}
    for dev in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        info = {"vid": "", "pid": "", "serial": "", "product": "", "usb_path": ""}
        u = usb_device_dir(os.path.basename(dev))
        if u:
            info.update(vid=read(f"{u}/idVendor"), pid=read(f"{u}/idProduct"), serial=read(f"{u}/serial"),
                        product=read(f"{u}/product"), usb_path=os.path.basename(u))
        ports[dev] = info
    links = {link: os.path.realpath(link) for link in sorted(glob.glob("/dev/so101_*"))}
    return ports, links


def describe(dev, info, links):
    alias = " ".join(f"<- {link}" for link, target in links.items() if target == dev)
    known = KNOWN.get(info["vid"], "")
    return (f"{dev:14} {info['vid']}:{info['pid']}  serial={info['serial'] or '-':18} "
            f"usb={info['usb_path'] or '-':10} {info['product']} {known} {alias}").rstrip()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    ports, links = list_ports()
    if mode == "list":
        if not ports:
            print("Aucun port série USB (ttyACM*/ttyUSB*). Bras branchés en USB ?")
        for dev, info in ports.items():
            print(describe(dev, info, links))
        for link, target in links.items():
            if target not in ports:
                print(f"{link} -> {target} (cible absente : bras débranché ?)")
    elif mode == "snapshot":
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        with open(SNAPSHOT, "w") as f:
            json.dump(ports, f, indent=1)
        print(f"{len(ports)} port(s) mémorisé(s). Branche ou débranche UN bras, puis : make ports-diff")
    elif mode == "diff":
        if not os.path.exists(SNAPSHOT):
            sys.exit("Pas de snapshot : lance d'abord make ports-snapshot")
        with open(SNAPSHOT) as f:
            before = json.load(f)
        added = sorted(set(ports) - set(before))
        removed = sorted(set(before) - set(ports))
        for dev in added:
            print("APPARU   " + describe(dev, ports[dev], links))
        for dev in removed:
            print("DISPARU  " + describe(dev, before[dev], {}))
        if not added and not removed:
            print("Aucun changement depuis le snapshot.")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()