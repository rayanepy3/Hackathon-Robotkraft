#!/usr/bin/env python3
"""Lance une commande d'enregistrement (make record...) et affiche en direct où on en est.

Une ligne qui se met à jour chaque seconde : ÉPISODE 3/10 (ou PAUSE) et le temps restant,
plus des annonces vocales (Go, plus que 5 secondes, Pause, Fini). Les messages de l'encodeur
vidéo (Svt[...]) sont masqués ; les avertissements et erreurs restent affichés.

  python3 scripts/record_live.py --n 10 --episode-s 20 --reset-s 5 --label bleu -- make record ...
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, required=True, help="nombre d'épisodes demandés")
ap.add_argument("--episode-s", type=float, required=True)
ap.add_argument("--reset-s", type=float, required=True)
ap.add_argument("--label", default="", help="affiché à côté du compteur (ex. bleu)")
ap.add_argument("--dataset-dir", default="", help="pour compter les épisodes sauvegardés")
ap.add_argument("cmd", nargs=argparse.REMAINDER)
args = ap.parse_args()
cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd

VOICE = shutil.which("spd-say") is not None and not os.environ.get("RECORD_LIVE_QUIET")
lock = threading.Lock()
state = {"phase": "DÉMARRAGE (caméra)", "end": None, "k": 0, "warned": False, "done": False}


def say(text):
    if VOICE:
        subprocess.Popen(["spd-say", "-l", "fr", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def status_line():
    s = state
    head = f"{s['phase']}" + (f" {s['k']}/{args.n}" if s["phase"] == "ÉPISODE" else "")
    head += f"  {args.label}" if args.label and s["phase"] == "ÉPISODE" else ""
    if s["end"] is None:
        return f"  >>> {head} ..."
    left = max(0.0, s["end"] - time.time())
    total = args.episode_s if s["phase"] == "ÉPISODE" else args.reset_s
    filled = int(20 * (1 - left / total)) if total else 20
    keys = "(→ = fini, ← = refaire)" if s["phase"] == "ÉPISODE" else "(→ = passer la pause)"
    return f"  >>> {head}  [{'#' * filled}{'.' * (20 - filled)}]  {left:4.0f} s restantes   {keys}"


def ticker():
    while not state["done"]:
        with lock:
            if state["phase"] == "ÉPISODE" and state["end"] and not state["warned"] and state["end"] - time.time() <= 5:
                state["warned"] = True
                say("Plus que 5 secondes")
            sys.stdout.write("\r\033[K" + status_line())
            sys.stdout.flush()
        time.sleep(0.5)


def episodes():
    try:
        with open(os.path.join(args.dataset_dir, "meta", "info.json")) as f:
            return json.load(f)["total_episodes"]
    except (OSError, KeyError, ValueError):
        return 0


def show(line):
    with lock:
        sys.stdout.write("\r\033[K" + line + "\n")


before = episodes() if args.dataset_dir else 0
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
threading.Thread(target=ticker, daemon=True).start()
noise = re.compile(r"Svt\[|^\s*$|^\s*--|^\s*\\$|lerobot-record \\|^docker run")
for raw in proc.stdout:
    for line in raw.replace("\r", "\n").split("\n"):
        line = line.strip()
        if "Recording episode" in line:
            with lock:
                state.update(phase="ÉPISODE", end=time.time() + args.episode_s, k=state["k"] + 1, warned=False)
            say(f"Go, épisode {state['k']}")
        elif "Reset the environment" in line:
            with lock:
                state.update(phase="PAUSE", end=time.time() + args.reset_s)
            say("Pause")
        elif "Re-record episode" in line:
            with lock:
                state["k"] -= 1
            say("On refait")
            show("  ← épisode jeté, on le refait")
        elif "Stop recording" in line:
            with lock:
                state.update(phase="FIN", end=None)
        elif line and not noise.search(line) and re.search(r"WARN|ERR|rror|Traceback|Exception|OAK-D|ERREUR|Overload|packet", line):
            show("  " + line[:220])
code = proc.wait()
state["done"] = True
time.sleep(0.6)
saved = episodes() - before if args.dataset_dir else None
msg = f"Terminé : {saved} épisode(s) sauvegardé(s) sur {args.n} demandé(s)." if saved is not None else "Terminé."
if code != 0:
    msg += f"  (erreur en fin de programme, code {code}" + ("" if not saved else " : les épisodes sauvegardés sont gardés") + ")"
show("\n" + msg)
say("Fini" if saved or code == 0 else "Problème")
sys.exit(0 if (saved or code == 0) else 1)