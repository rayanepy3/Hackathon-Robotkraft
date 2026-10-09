"""Outils communs aux épreuves (TRI, ETIQ) : caméras, voix, journal, lancement du modèle.

Aucune commande moteur ici. Le bras n'est piloté que par les commandes officielles, via
make rollout (modèle sur le CPU du PC) ou make rollout-async (modèle sur un GPU distant).

Caméras :
- Oak : images du pont OAK permanent (make oak-bridge-start, localhost:5555). Le même pont sert
  le modèle pendant les essais : l'OAK n'est ouverte qu'une fois pour toute l'épreuve.
- Poignet : ouverte le temps d'une photo, entre deux essais (le modèle l'utilise pendant les essais).
"""
import base64
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "epreuves"


def say(text, wait=False):
    """Affiche et annonce à voix haute (spd-say), sans bloquer sauf wait=True."""
    print(f">>> {text}", flush=True)
    if shutil.which("spd-say") and not os.environ.get("EPREUVE_SILENCE"):
        cmd = ["spd-say", "-l", "fr"] + (["-w"] if wait else []) + [text]
        (subprocess.run if wait else subprocess.Popen)(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Journal:
    """Dossier outputs/epreuves/<nom>_<heure>/ : journal.txt + images annotées de chaque étape."""

    def __init__(self, name):
        self.dir = OUTPUTS / f"{name}_{datetime.datetime.now():%m%d_%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def log(self, msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.dir / "journal.txt", "a") as f:
            f.write(line + "\n")

    def image(self, img, label):
        """Enregistre une image numérotée (ex. 003_avant_essai.jpg) et renvoie son chemin."""
        self.n += 1
        path = self.dir / f"{self.n:03d}_{label}.jpg"
        cv2.imwrite(str(path), img)
        return path


class Oak:
    """Images BGR de l'OAK via le pont permanent. Lance le pont s'il ne tourne pas ; le relance s'il se tait."""

    def __init__(self, port=None, start=True):
        self.port = int(port or os.environ.get("CAM1_ZMQ_PORT", 5555))
        self.started_here = False
        self._sock = None
        if not self._alive():
            if not start:
                raise RuntimeError("pont OAK absent : make oak-bridge-start")
            self._start_bridge()

    def _socket(self):
        if self._sock is None:
            self._sock = zmq.Context.instance().socket(zmq.SUB)
            self._sock.setsockopt(zmq.CONFLATE, 1)  # seule la dernière image compte
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
            self._sock.connect(f"tcp://127.0.0.1:{self.port}")
        return self._sock

    def _alive(self, timeout_ms=2500):
        return bool(self._socket().poll(timeout_ms))

    def _start_bridge(self):
        print("Démarrage du pont OAK...", flush=True)
        subprocess.run(["make", "-C", str(ROOT), "--no-print-directory", "oak-bridge-start"], check=True)
        self.started_here = True
        deadline = time.time() + 60  # l'OAK peut planter puis redémarrer plusieurs fois (USB 2)
        while time.time() < deadline:
            if self._alive(2000):
                return
        raise RuntimeError("pont OAK lancé mais aucune image après 60 s (docker logs rk-oak)")

    def frame(self, timeout_s=10, fresh=True):
        """Dernière image (BGR). fresh=True : jette l'image en attente pour en prendre une postérieure à l'appel."""
        sock = self._socket()
        if fresh and sock.poll(0):
            sock.recv_string()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if sock.poll(500):
                msg = json.loads(sock.recv_string())
                b64 = next(iter(msg["images"].values()))
                img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # le pont envoie des octets RGB (format LeRobot)
        if not self.started_here:
            self._start_bridge()
            return self.frame(timeout_s, fresh)
        raise RuntimeError(f"aucune image de l'OAK depuis {timeout_s} s (docker logs rk-oak)")

    def close(self):
        if self.started_here:
            subprocess.run(["make", "-C", str(ROOT), "--no-print-directory", "oak-bridge-stop"])


def wrist_frame(path=None, warmup=8):
    """Une photo de la caméra poignet (BGR), ouverte puis refermée tout de suite."""
    path = path or make_var("WRIST_CAM")
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"caméra poignet indisponible : {path} (utilisée par un autre programme ?)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    img = None
    for _ in range(warmup):  # laisse l'exposition automatique se régler
        ok, f = cap.read()
        img = f if ok else img
    cap.release()
    if img is None:
        raise RuntimeError(f"caméra poignet : aucune image ({path})")
    return img


def make_var(name):
    """Valeur d'une variable du Makefile (config/robotkraft.env + .env.local + ligne de commande)."""
    out = subprocess.run(["make", "-C", str(ROOT), "--no-print-directory", "-s", f"print-{name}"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def run_policy(policy, text, seconds, fps, async_server="", dry=False, journal=None):
    """Un essai du modèle : la phrase `text` pendant `seconds` s, via les commandes officielles.

    Le bras garde sa pose à la fin (couple maintenu), pour enchaîner les essais sans retomber.
    dry=True : n'envoie rien au bras (test du workflow). Renvoie True si la commande s'est bien terminée.
    """
    target = "rollout-async" if async_server else "rollout"
    args = ["make", "-C", str(ROOT), "--no-print-directory", target,
            f"TASK_TEXT={text}", f"REC_FPS={fps}",
            "ROBOT_EXTRA=--robot.disable_torque_on_disconnect=false"]
    if async_server:
        args += [f"ASYNC_SERVER={async_server}", f"ASYNC_POLICY={policy}", f"ROLLOUT_DURATION_S={seconds}"]
    else:
        args += [f"POLICY_PATH={policy}", "EVAL_EPISODES=1", f"EVAL_EPISODE_TIME_S={seconds}", "EVAL_RESET_TIME_S=0"]
    if journal:
        journal.log(f"modèle : {target} « {text} » {seconds} s" + (" [TEST À BLANC]" if dry else ""))
    if dry:
        time.sleep(1)
        return True
    log = (journal.dir / "modele.log") if journal else Path(os.devnull)
    with open(log, "a") as f:
        f.write(f"\n===== {datetime.datetime.now():%H:%M:%S} {' '.join(args)}\n")
        f.flush()
        code = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode
    # rollout-async s'arrête par un Ctrl+C simulé : un code non nul y est normal.
    ok = code == 0 or bool(async_server)
    if journal and not ok:
        journal.log(f"modèle : erreur (code {code}), voir {log}")
    return ok


def parse_common_args(parser):
    """Options communes aux workflows d'épreuve."""
    parser.add_argument("--policy", required=True, help="modèle (dépôt Hub ou dossier local)")
    parser.add_argument("--fps", type=int, required=True, help="fréquence du dataset du modèle")
    parser.add_argument("--seconds", type=float, required=True, help="durée d'un essai du modèle")
    parser.add_argument("--async-server", default="", help="ip:port du policy_server (vide = modèle sur le CPU)")
    parser.add_argument("--max-tries", type=int, default=4, help="essais maximum pour une même action")
    parser.add_argument("--dry-run", action="store_true", help="vision réelle, aucun mouvement du bras")
    return parser


def die(msg):
    say("Problème")
    sys.exit(f"ERREUR : {msg}")