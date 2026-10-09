#!/usr/bin/env python3
"""Épreuve TRI : enregistre une session de rangement, un épisode par tube, via make record / record-resume.

On range couleur par couleur, dans l'ordre du groupe, toujours dans l'emplacement libre le plus à gauche :
le résultat est juste quel que soit le nombre de tubes par couleur. Le modèle n'apprend donc que
3 gestes, un par phrase (TRI_TEXT_B/J/R de config/robotkraft.env), et le niveau 2 (ordre imposé
au moment du test) ne demande aucune donnée de plus : seul l'enchaînement des couleurs change.

  make record-tri SEQ=BBJRRR      couleurs dans l'ordre où tu les ranges (B bleu, J jaune, R rouge/magenta)

Le PC annonce chaque tube à voix haute. Session interrompue : relancer avec les lettres restantes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

COLORS_FR = {"B": "bleu", "J": "jaune", "R": "rouge"}
QUIET = "--dry-run" in sys.argv


def say(text):
    print(f"\n>>> {text}", flush=True)
    if shutil.which("spd-say") and not QUIET:
        subprocess.run(["spd-say", "-l", "fr", "-w", text], check=False)


def episodes(dataset_dir):
    try:
        with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
            return json.load(f)["total_episodes"]
    except (OSError, KeyError, ValueError):
        return 0


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seq", required=True, help="ex. BBJRRR")
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--task-name", required=True)
ap.add_argument("--text-b", required=True)
ap.add_argument("--text-j", required=True)
ap.add_argument("--text-r", required=True)
ap.add_argument("--dry-run", action="store_true", help="affiche les commandes sans rien lancer")
args = ap.parse_args()

seq = args.seq.upper()
if not seq or len(seq) > 6 or any(c not in COLORS_FR for c in seq):
    sys.exit(f"SEQ={args.seq} invalide : 1 à 6 lettres parmi B, J, R (ex. BBJRRR)")
texts = {"B": args.text_b, "J": args.text_j, "R": args.text_r}

if len(seq) > 1:
    say(f"Session de {len(seq)} tubes : " + ", ".join(COLORS_FR[c] for c in seq))
for i, c in enumerate(seq):
    target = "record-resume" if os.path.isdir(args.dataset_dir) else "record"
    cmd = ["make", target, f"TASK_NAME={args.task_name}", "NUM_EPISODES=1", f"TASK_TEXT={texts[c]}"]
    say(f"Tube {i + 1} sur {len(seq)} : {COLORS_FR[c]}. Leader en position de repos.")
    if args.dry_run:
        print(" ".join(cmd[:4]) + f' TASK_TEXT="{texts[c]}"')
        continue
    before = episodes(args.dataset_dir)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "Recording episode" in line:
            say("Go")
    code = proc.wait()
    if episodes(args.dataset_dir) <= before:
        say("Problème, session arrêtée")
        sys.exit(f"\nÉpisode non sauvegardé au tube {i + 1}. Pour reprendre : make record-tri SEQ={seq[i:]}")
    if code != 0:  # ex. « Overload » de la pince à la déconnexion : l'épisode est sauvegardé quand même
        say("Épisode gardé, mais erreur à la fin. Vérifie la pince")
say("Session terminée")