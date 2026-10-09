# RobotKraft : SO-101 → dataset LeRobot → fine-tuning SmolVLA → rollout

On téléopère le bras follower avec le leader et on enregistre ~50 épisodes d'une tâche de pick-and-place,
vus par deux caméras. On fine-tune `lerobot/smolvla_base` sur ces épisodes, puis le modèle pilote seul
le follower et on mesure son taux de succès.

Tout passe par `make`, qui appelle les commandes officielles `lerobot-*` (LeRobot 0.5.1, celle de l'image
Docker du challenge). Pas de boucle de contrôle maison, pas de modification de LeRobot.
Toute la configuration est dans [config/robotkraft.env](config/robotkraft.env). `make` liste les cibles.

> Lance les commandes interactives (`udev`, `calibrate-*`, `teleop`, `record`, `rollout`, `eval-log`)
> depuis le **terminal intégré de VS Code**. Les touches de `lerobot-record` (→ ← Échap) passent par X11 :
> VS Code est un client X11 et les reçoit, le Terminal GNOME (Wayland) probablement pas.

## Démarrage rapide : du PC éteint au premier épisode

| # | Commande | Ce qu'il faut voir |
|---|---|---|
| 1 | `cd ~/robotkraft && make doctor` | Image OK, caméras OK. Bras et calibrations « ABSENT » au premier passage. |
| 2 | `echo "HF_USER=ton_pseudo_hf" > config/robotkraft.env.local` | Puis édite `TASK_NAME` et `TASK_TEXT` dans `config/robotkraft.env`. |
| 3 | `make hf-login`, puis `make hf-whoami` | Colle un token **write** (huggingface.co/settings/tokens) ou suis le lien de connexion affiché. Ton pseudo s'affiche. |
| 4 | Brancher : follower (alim **12 V**), leader (alim **5 V**), OAK-D (USB 3), caméra poignet | |
| 5 | `make udev` (une seule fois) | Suis les questions. sudo est demandé à la fin, après affichage de la règle. `OK /dev/so101_follower -> ...` |
| 6 | `make cameras` | Ouvre `outputs/cam_check/all.jpg` : camera1 voit tout l'espace de travail, camera2 voit la pince, net. |
| 7 | `make calibrate-follower`, puis `make calibrate-leader` | Fichiers `.json` créés (`make calib-status`). |
| 8 | `make teleop` | Le follower copie le leader. Ctrl+C pour arrêter. |
| 9 | `make record NUM_EPISODES=3` | 3 épisodes d'essai. Ensuite `make check-dataset`. |
| 10 | Si le contrôle est OK : `make record-resume NUM_EPISODES=47` | Le dataset atteint 50 épisodes. |
| 11 | `make push-dataset`, puis entraînement (Phase 3) | |

Le détail de chaque phase, les vérifications physiques et les erreurs courantes sont plus bas.

## Épreuves

| Dossier | Épreuve | Commande |
|---|---|---|
| [TRI/](TRI/) | Tri de tubes par couleur | `make tri ORDER=BJR` |
| [ETIQ/](ETIQ/) | Reconnaissance d'étiquettes et saisie | `make etiquette MOT=...` |
| [MASSE/](MASSE/) | Pesée par oscillation | à faire |
| [COMMUN/](COMMUN/) | Caméras, voix, journal, lancement du modèle (partagé) | |

Chaque épreuve : la vision décide, le modèle (commandes officielles LeRobot) fait le geste, la vision vérifie, et on recommence si raté. `make install-epreuves` installe la vision et l'OCR.

## Comment c'est construit

```
leader (5 V) ──USB──> /dev/so101_leader ──┐
follower (12 V) ─USB─> /dev/so101_follower ┼─> lerobot-record ──> dataset LeRobot v3.0
OAK-D Lite ─USB─> oak_zmq_server.py ─ZMQ─> camera1 (vue fixe)      (.cache/huggingface/lerobot/<HF_USER>/robotkraft_<TASK_NAME>)
webcam poignet ─USB─> OpenCV ──────────> camera2 (poignet)              │
                                                                        ▼ make push-dataset (Hub, privé)
                     serveur GPU : lerobot-train --policy.path=lerobot/smolvla_base
                                                                        │ modèle sur le Hub
                                                                        ▼
                     PC : lerobot-record --policy.path=...  (le modèle pilote le follower)
```

Choix et raisons (sources et vérifications : [docs/REFERENCES.md](docs/REFERENCES.md)) :

- **Clés `camera1` / `camera2` dès l'enregistrement.** Ce sont les noms d'entrée de `smolvla_base` : pas de
  rename_map, et SmolVLA 0.5.1 ignore la camera3 absente. Avec d'autres noms (`top`, `wrist`...), le chargement
  de la policy échoue sur « Feature mismatch ».
- **État = 6 positions du follower, action = 6 positions du leader.** C'est ce que produit `lerobot-record` avec
  un SO-101 : `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- **640x480 à 30 fps, profondeur coupée.** C'est le format du dataset de référence de SmolVLA (svla_so101_pickplace).
  La caméra ZMQ de LeRobot ne redimensionne pas : le pont OAK (`scripts/oak_zmq_server.py`) produit directement
  du 640x480. Ce format couvre tout le champ du capteur 4:3 de l'OAK-D Lite.
- **Docker non privilégié.** L'image du challenge (LeRobot 0.5.1) reçoit en plus `num2words`, sans lequel SmolVLA
  ne se charge pas ([docker/Dockerfile](docker/Dockerfile)). Les conteneurs ne reçoivent que les devices utiles.
  En mode `privileged` (repo du challenge), Docker ignore le mapping `/dev/lerobot_follower:/dev/ttyACM0` :
  le « follower » y est en fait le premier bras branché.
- **Calibrations, datasets, modèles et token HF** sont dans `.cache/`, ignoré par git, comme `outputs/`.
- **Mêmes ids que le repo du challenge** (`follower_arm`, `leader_arm`) : `make import-calib` reprend ses calibrations.

## Déjà validé sans les bras (nuit du 2 au 3 octobre)

| Test | Résultat |
|---|---|
| Image `robotkraft-smolvla:local` | LeRobot 0.5.1, torch 2.10 CPU, SmolVLA importable |
| Caméras dans le conteneur de `make record`, via les classes LeRobot | camera1 `ZMQCamera` et camera2 `OpenCVCamera` : 480x640x3, ~30 images/s |
| Commandes générées par `record`, `record-resume`, `rollout`, `rollout-async` | acceptées par les parseurs de LeRobot 0.5.1 |
| Encodage vidéo en direct, 2 caméras 640x480 à 30 fps | tient le rythme sur ce PC (AV1 comme H.264) |
| `lerobot-train` SmolVLA depuis smolvla_base, et ACT, sur un dataset au format exact de `lerobot-record` | OK (3 pas sur CPU, loss en baisse, checkpoint écrit) |
| Inférence sur le CPU de ce PC (i5-1155G7) | SmolVLA : **5 s** par paquet de 50 actions (1,7 s de mouvement). ACT : **0,3 s** par paquet de 100 actions (3,3 s de mouvement) |
| `check_dataset.py`, `eval_logger.py` | OK sur données de test |

Reste à valider avec le matériel : `make udev`, calibration, téléopération, enregistrement réel, rollout.

## Phase 0 — Installation et détection

```bash
make install    # déjà fait sur ce PC ; ~1 min si l'image du challenge est présente
make doctor     # état général
make ports      # ports série USB, avec leur chemin USB physique
make cameras    # une image par caméra dans outputs/cam_check/, débit, alerte image noire
```

À vérifier physiquement :
- camera1 (OAK-D) voit tout l'espace de travail : le bras, la zone de prise, la zone de dépôt. Mieux vaut la
  surélever et l'incliner vers la table qu'une vue de profil avec beaucoup de mur.
- camera2 (poignet) voit les mors de la pince et l'objet, net. Règle la mise au point à la main (bague de l'objectif)
  sur environ 10 à 20 cm.
- `make cameras` ne doit signaler aucune « image quasi noire » quand la salle est éclairée.

## Phase 1 — Ports stables, calibration, téléopération

```bash
make udev                 # identifie les bras en les branchant un par un, crée /dev/so101_*
make calibrate-follower   # [terminal]
make calibrate-leader     # [terminal]
make teleop               # [terminal]
```

- **`make udev`** demande de débrancher le USB des deux bras, de brancher le follower seul, puis le leader.
  Si les deux cartes ont des numéros de série différents, la règle s'appuie dessus. Sinon, elle s'appuie sur le
  port USB physique : étiquette alors les ports et rebranche toujours chaque bras au même endroit.
- **Calibration** (`lerobot-calibrate`, une fois par bras) : place le bras au milieu de sa course, Entrée.
  Ensuite, bouge chaque articulation d'une butée à l'autre, une par une, pince comprise. Entrée pour finir.
  Si les calibrations existent déjà dans le repo du challenge : `make import-calib`.
- **Si LeRobot demande « Press ENTER to use provided calibration file... or type 'c' »** : un fichier de calibration
  existe déjà pour cet id. Ça arrive en relançant `make calibrate-*`, ou au démarrage si les moteurs ont été
  calibrés ailleurs entre-temps (autre PC). Entrée garde le fichier et le réécrit dans les moteurs. `c` refait la calibration.
- **Téléopération** : avant de lancer, mets le leader dans la même pose que le follower. Au démarrage, le follower
  rejoint d'un coup la pose du leader. Garde une main près de l'alimentation 12 V du follower.

Ne lance jamais `lerobot-setup-motors` (`make setup-motors` refuse exprès) : il réécrit l'EEPROM des moteurs, et
les bras du challenge sont déjà configurés.

## Phase 2 — Enregistrement du dataset

```bash
make record NUM_EPISODES=3          # essai court : vérifie tout avant d'enregistrer en série
make check-dataset                   # rapport + graphiques dans outputs/dataset_check/<dataset>/
make record-resume NUM_EPISODES=47   # complète jusqu'à 50
make push-dataset                    # envoie sur le Hub (privé par défaut)
```

Pendant `lerobot-record` :
- chaque épisode dure au plus `EPISODE_TIME_S` (20 s), puis vient une pause de remise en place de `RESET_TIME_S` (10 s) ;
- **→** termine l'épisode tout de suite (tâche réussie avant la fin), **←** le jette et le refait, **Échap** arrête
  proprement (le dataset reste valide) ;
- garde la même phrase de tâche (`TASK_TEXT`) pour tout le dataset et pour le rollout.

Pour que le modèle généralise (c'est ce que le jury évalue) :
- varie la position de départ de l'objet sur toute la zone : environ 10 épisodes par zone ou variation ;
- varie un peu l'éclairage et ajoute quelques distracteurs dans une partie des épisodes ;
- fais des gestes nets et réguliers, sans hésitation : le modèle copie aussi les hésitations ;
- jette (←) les épisodes ratés plutôt que de les garder.

`make check-dataset` signale en FAIL : mauvais fps, clés ou formes, plusieurs phrases de tâche, NaN.
Il signale en WARN : épisodes trop courts, articulation ou pince immobile, sauts d'état, écart leader/follower,
images noires ou figées. Regarde `overview.png` (diversité des trajectoires) et `frames.jpg` (ce que voit le modèle).

## Phase 3 — Fine-tuning SmolVLA

Le PC n'a pas de GPU NVIDIA. Trois options :

1. **Serveur GPU du challenge (gratuit, recommandé)**. `make train-cmd` affiche les 3 commandes à coller dans un
   terminal JupyterLab du serveur : installation de `lerobot[smolvla]==0.5.1` si absent, `hf auth login`, puis
   l'entraînement en arrière-plan. Le serveur télécharge le dataset depuis le Hub et y renvoie le modèle à la fin.
2. **Une machine NVIDIA avec Docker** (ta tour 5070 Ti, par exemple) : copie ce dossier, `make install-cuda`, puis
   mets `IMAGE=robotkraft-smolvla:cuda` et `POLICY_DEVICE=cuda` dans `config/robotkraft.env.local`, et `make train`.
   Toutes les cibles utilisent alors l'image CUDA, rollout compris.
3. **HF Jobs (payant, facturé à la seconde)** : `make train-hf-job` affiche la commande et le coût sans rien lancer.
   Pour lancer réellement : `make train-hf-job CONFIRM_PAID=yes`.

**ACT en secours.** `make train-cmd` affiche aussi la commande ACT, sur le même dataset. Lance-la en parallèle
sur une deuxième instance GPU : ACT s'entraîne vite et tourne de façon fluide sur le CPU du PC.
Si l'inférence GPU de SmolVLA n'est pas disponible pendant la démo, c'est lui qui assure.

Réglages (`config/robotkraft.env`) :
- `TRAIN_STEPS=20000` : la doc SmolVLA compte environ 4 h sur un A100 complet.
- `BATCH_SIZE=32` : 64 dans la doc (sur A100). À ajuster selon la mémoire GPU, et à baisser en cas de « CUDA out of memory ».
- `SAVE_FREQ=5000` : un checkpoint tous les 5000 pas. `make train-resume` reprend au dernier.

Surveille la loss dans le log : elle doit baisser nettement pendant les premiers milliers de pas.

**Budget temps.** Chaque ligne du log affiche `updt_s` (secondes par pas) : durée totale ≈ `TRAIN_STEPS` × `updt_s`.
Les instances du challenge sont des parts de GPU : 24 Go sur Agora, A100 MIG 20 Go limitée à 4 h sur InnovLab.
20k pas peuvent donc prendre bien plus que les 4 h annoncées pour un A100 entier. Lance l'entraînement dès que le
dataset est poussé. Calcule la durée après quelques minutes et baisse `TRAIN_STEPS` si la fin tombe trop tard
(10k pas, par exemple). Un checkpoint est écrit tous les `SAVE_FREQ` pas : `make train-cmd` donne la commande
`hf upload` pour tester un checkpoint intermédiaire sur le PC sans attendre la fin.

## Phase 4 — Rollout et taux de succès

```bash
make rollout                                                        # SmolVLA du Hub (MODEL_REPO_ID)
make rollout POLICY_PATH=ton_pseudo/act_robotkraft_pickplace        # ou ACT, ou un dossier local
make eval-log RUN=v1 CONDITION=nominal                              # 2e terminal : s / p / e après chaque essai
make eval-report                                                    # tableau + graphique dans outputs/eval/
```

- `make rollout` lance `lerobot-record --policy.path=...`. Le modèle pilote le follower pendant `EVAL_EPISODES`
  essais de `EVAL_EPISODE_TIME_S`, avec une pause de remise en place entre deux. Chaque essai est enregistré
  dans un dataset `eval_...` daté.
- **Où tourne le modèle**, d'après les mesures faites sur ce PC :

  | Policy | Inférence | Comportement du bras |
  |---|---|---|
  | SmolVLA sur le CPU du PC (`make rollout`) | 5 s par paquet de 50 actions | 1,7 s de mouvement puis 5 s d'arrêt. Utilisable pour tester, mets `EVAL_EPISODE_TIME_S=90` |
  | SmolVLA sur une machine GPU | une fraction de seconde | fluide |
  | ACT sur le CPU du PC (`make rollout POLICY_PATH=..._act_...`) | 0,3 s par paquet de 100 actions | fluide (courte pause toutes les 3,3 s) |

  Deux façons de faire tourner SmolVLA sur un GPU :
  - **tout sur la machine GPU** : bras et caméras branchés dessus, ce dossier copié et configuré comme en
    Phase 3, option 2, puis `make udev`, `make import-calib` (ou recalibrer) et `make rollout` ;
  - **inférence déportée** : sur la machine GPU, `make policy-server` ; sur le PC,
    `make rollout-async ASYNC_SERVER=ip_de_la_machine:8080`. Les deux machines doivent se joindre sur le réseau,
    par exemple avec un câble Ethernet direct.
- Conditions à tester (critères du jury) : `nominal`, `position`, `eclairage`, `distracteurs`, `objet_nouveau`.
  Fais 10 essais par condition. L'intervalle de confiance du rapport rappelle qu'avec 10 essais, l'écart entre deux
  versions doit être grand pour être significatif.

## Dépannage

| Symptôme | Cause probable et solution |
|---|---|
| `ERREUR : /dev/so101_follower absent` | Bras débranché, ou `make udev` pas fait. Avec la règle par port USB : bras rebranché sur un autre port. |
| `make teleop` : le follower ne bouge pas | Alimentation 12 V du follower branchée ? Sinon bras inversés dans la règle udev : refais `make udev`. |
| `make teleop` : le follower bouge de travers ou à l'envers | Calibration ratée ou faite avec l'autre bras : refais `make calibrate-follower` et `make calibrate-leader`. |
| `FileExistsError` ou `existe déjà` au record | Le dataset existe : `make record-resume`, ou change `TASK_NAME`. |
| Le record a planté avant le premier épisode sauvegardé | Supprime le dossier vide : `rm -rf .cache/huggingface/lerobot/<HF_USER>/robotkraft_<TASK_NAME>`. |
| Les touches → ← Échap ne font rien | Terminal Wayland : relance depuis le terminal de VS Code. Sinon, ouvre une session « Ubuntu sur Xorg ». |
| `aucune image de l'OAK-D` | Rebranche l'OAK-D sur un port USB 3 et ferme les autres programmes qui l'utilisent. |
| `Record loop is running slower` | PC surchargé : ferme les autres programmes. L'OAK doit être sur un port USB 3. |
| `Feature mismatch` au chargement de la policy | Noms de caméras différents de `camera1` / `camera2`. |
| `'policy.repo_id' argument missing` | Entraînement avec `PUSH_MODEL=true` sans `MODEL_REPO_ID` : garde la config par défaut. |
| `CUDA out of memory` | Baisse `BATCH_SIZE`, par exemple `make train BATCH_SIZE=16`. |

## Fichiers

```
Makefile                 point d'entrée unique (make = aide)
config/robotkraft.env    ports, ids, caméras, tâche, HF_USER, dataset, durées, entraînement
docker/Dockerfile        image du challenge + num2words (extra smolvla)
scripts/install.sh       installation : docker (défaut), cuda, native (conda Miniforge)
scripts/detect_ports.py  ports série USB, modes snapshot/diff
scripts/detect_cameras.py caméras + une capture chacune dans outputs/cam_check/
scripts/udev_rules.sh    symlinks /dev/so101_follower et /dev/so101_leader
scripts/oak_zmq_server.py pont OAK-D -> ZMQ (camera1), repris du repo du challenge, résolution paramétrable
scripts/with_oak.sh      lance le pont OAK le temps d'une commande
scripts/robot_client_zmq.py client d'inférence async officiel + caméra ZMQ (repris du repo du challenge)
tools/check_dataset.py   validation du dataset + graphiques
tools/eval_logger.py     journal des essais -> CSV, tableau, graphique
docs/REFERENCES.md       sources et décisions
```