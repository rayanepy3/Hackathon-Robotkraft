#!/usr/bin/env python3
"""Liste les caméras et capture une image de chacune dans --out (outputs/cam_check/ par défaut).

- Webcams V4L2 (dont la caméra poignet) via OpenCV, avec leur chemin stable /dev/v4l/by-id/...
- OAK-D Lite (USB 03e7) via depthai : flux RGB seul, à la résolution demandée.
Mesure le débit réel et signale les images quasi noires (objectif masqué, caméra retournée).
Aucun réglage caméra n'est conservé après la capture.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

DARK = 20  # luminosité moyenne (0-255) en dessous de laquelle l'image est jugée quasi noire


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def by_id_names():
    """/dev/videoN -> chemin stable. Dans le conteneur, /dev/v4l de l'hôte est monté sur /host-v4l."""
    for base in ("/host-v4l/by-id", "/dev/v4l/by-id"):
        if os.path.isdir(base):
            return {"/dev/" + os.path.basename(os.readlink(os.path.join(base, n))): f"/dev/v4l/by-id/{n}"
                    for n in sorted(os.listdir(base))}
    return {}


def v4l_capture_nodes():
    """Nœuds vidéo qui délivrent des images (index 0). Les autres sont des flux de métadonnées."""
    nodes = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int(p[10:]) if p[10:].isdigit() else 999):
        name = os.path.basename(dev)
        if read(f"/sys/class/video4linux/{name}/index") in ("", "0"):
            nodes.append((dev, read(f"/sys/class/video4linux/{name}/name") or "?"))
    return nodes


def usb_speed_v4l(dev):
    """Débit du lien USB (Mb/s) de la webcam : 480 = USB 2, 5000 et plus = USB 3."""
    d = os.path.realpath(f"/sys/class/video4linux/{os.path.basename(dev)}/device")
    while d != "/" and not os.path.exists(os.path.join(d, "speed")):
        d = os.path.dirname(d)
    return read(os.path.join(d, "speed")) if d != "/" else ""


def usb_label(mbps):
    if not mbps:
        return "?"
    m = float(mbps)
    return f"USB {'3' if m >= 5000 else '2' if m >= 480 else '1'} ({int(m)} Mb/s)"


def measure(read_frame, fps, warmup_s=1.5, seconds=2.0):
    """Lit pendant warmup_s (auto-exposition), puis compte les images reçues pendant `seconds`."""
    frame, t0 = None, time.time()
    while time.time() - t0 < warmup_s:
        f = read_frame()
        frame = f if f is not None else frame
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        f = read_frame()
        if f is not None:
            frame, n = f, n + 1
    return frame, n / seconds


def grab_v4l(dev, width, height, fps):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, 0.0, "ouverture impossible"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    def read_frame():
        ok, f = cap.read()
        return f if ok else None

    frame, rate = measure(read_frame, fps)
    cap.release()
    return frame, rate, "" if frame is not None else "aucune image reçue"


def oak_present():
    return any(read(p) == "03e7" for p in glob.glob("/sys/bus/usb/devices/*/idVendor"))


def grab_oak(width, height, fps, timeout_s=30):
    import depthai as dai

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        queue = cam.requestOutput((width, height), fps=fps).createOutputQueue()
        pipeline.start()
        deadline = time.time() + timeout_s
        while not queue.has():
            if time.time() > deadline:
                return None, 0.0, f"aucune image après {timeout_s} s", ""
            time.sleep(0.01)

        def read_frame():
            msg = queue.tryGet()
            if msg is None:
                time.sleep(0.002)
                return None
            return msg.getCvFrame()

        frame, rate = measure(read_frame, fps)
        speed = str(pipeline.getDefaultDevice().getUsbSpeed()).split(".")[-1]
    mbps = {"LOW": "1.5", "FULL": "12", "HIGH": "480", "SUPER": "5000", "SUPER_PLUS": "10000"}.get(speed, "")
    return frame, rate, "", mbps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--wrist", default="", help="chemin configuré pour camera2 (WRIST_CAM)")
    ap.add_argument("--ignore", default="", help="caméras à ne pas ouvrir : morceaux de nom séparés par des virgules")
    ap.add_argument("--out", default="outputs/cam_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stable = by_id_names()
    results = []  # (étiquette, device, chemin stable, image BGR, débit, erreur)
    ignored = [w.strip().lower() for w in args.ignore.split(",") if w.strip()]
    for dev, name in v4l_capture_nodes():
        if any(w in name.lower() for w in ignored):
            print(f"Ignorée (CAM_IGNORE) : {name} ({dev})")
            continue
        frame, rate, err = grab_v4l(dev, args.width, args.height, args.fps)
        results.append((name, dev, stable.get(dev, "-"), frame, rate, err, usb_speed_v4l(dev)))
    if oak_present():
        try:
            frame, rate, err, mbps = grab_oak(args.width, args.height, args.fps)
        except Exception as e:  # depthai indisponible ou caméra occupée (aperçu ou record en cours ?)
            frame, rate, err, mbps = None, 0.0, f"{type(e).__name__}: {e}", ""
        results.append(("OAK-D (RGB)", "USB 03e7", "camera1 via ZMQ", frame, rate, err, mbps))
    else:
        print("OAK-D non détectée (aucun périphérique USB 03e7).")

    print(f"\nCapture demandée : {args.width}x{args.height} @ {args.fps} fps\n")
    tiles = []
    for label, dev, path, frame, rate, err, mbps in results:
        if frame is None:
            print(f"KO    {label:32} {dev:12} {err}")
            continue
        h, w = frame.shape[:2]
        mean = float(frame.mean())
        verdicts = []
        if (w, h) != (args.width, args.height):
            verdicts.append(f"taille {w}x{h} != demandée")
        if rate < 0.9 * args.fps:
            verdicts.append(f"débit faible : {rate:.1f} images/s pour {args.fps} demandées (câble, port, hub, éclairage trop faible ?)")
        if label.startswith("OAK") and mbps and float(mbps) < 5000:
            verdicts.append("OAK-D en USB 2 : ça marche en 640x480 mais c'est moins stable ; port bleu USB 3 + câble USB 3, sans hub")
        if mean < DARK:
            verdicts.append("image quasi noire : objectif masqué, caméra retournée ou trop sombre")
        status = "WARN" if verdicts else "OK"
        fname = "".join(c if c.isalnum() else "_" for c in f"{os.path.basename(dev)}_{label}")[:60] + ".jpg"
        cv2.imwrite(os.path.join(args.out, fname), frame)
        print(f"{status:5} {label:32} {dev:12} {w}x{h} {rate:4.1f} fps  luminosité {mean:5.1f}  -> {args.out}/{fname}")
        print(f"      chemin stable : {path}   |   lien {usb_label(mbps)}   |   débit {'OK' if rate >= 0.9 * args.fps else 'FAIBLE'}")
        for v in verdicts:
            print(f"      ! {v}")
        tile = cv2.resize(frame, (320, 240))
        cv2.putText(tile, label[:28], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(tile)
    if tiles:
        cv2.imwrite(os.path.join(args.out, "all.jpg"), np.hstack(tiles))
        print(f"\nPlanche : {args.out}/all.jpg")

    if args.wrist:
        found = [r for r in results if args.wrist in (r[1], r[2])]
        if found:
            print(f"camera2 (poignet) = {args.wrist} : trouvée ({found[0][0]}).")
        else:
            print(f"ATTENTION : camera2 configurée = {args.wrist}, introuvable. Mets dans WRIST_CAM le chemin stable de la caméra poignet ci-dessus.")


if __name__ == "__main__":
    main()