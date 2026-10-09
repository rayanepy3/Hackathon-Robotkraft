# Références et décisions (LeRobot 0.5.1, image du challenge)

Docs : [SO-101](https://huggingface.co/docs/lerobot/so101) · [IL robots](https://huggingface.co/docs/lerobot/il_robots) · [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) · [Caméras](https://huggingface.co/docs/lerobot/cameras) · [Rename map](https://huggingface.co/docs/lerobot/rename_map) · [Installation](https://huggingface.co/docs/lerobot/installation)
Modèle et données : [smolvla_base/config.json](https://huggingface.co/lerobot/smolvla_base/raw/main/config.json) · [svla_so101_pickplace/info.json](https://huggingface.co/datasets/lerobot/svla_so101_pickplace/raw/main/meta/info.json) · [papier SmolVLA](https://arxiv.org/abs/2506.01844)
Les docs suivent la branche main ; l'image embarque LeRobot 0.5.1, dont le code fait foi (tout ce qui suit est vérifié dans le conteneur).

- État = 6 positions du follower, action = 6 positions du leader (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`, suffixe `.pos`). Ni cartésien ni IK.
- Caméras nommées dès l'enregistrement : `camera1` = vue fixe (RGB de l'OAK-D Lite via ZMQ), `camera2` = poignet (OpenCV, MJPG). Ce sont les clés de smolvla_base : pas de rename_map ; camera3, absente, est ignorée (`empty_cameras=0`).
- 640x480 à 30 fps pour les deux, profondeur désactivée, comme svla_so101_pickplace (30 fps, 640x480). La ZMQCamera ne redimensionne pas : le pont OAK émet directement du 640x480.
- Une seule tâche, une seule phrase (`TASK_TEXT`), passée à `--dataset.single_task` au record comme au rollout : c'est l'instruction donnée au VLA.
- Environ 50 épisodes, variés en position, éclairage et distracteurs (doc SmolVLA : 25 donnaient de mauvais résultats).
- Fine-tuning depuis `lerobot/smolvla_base` (450M, VLM gelé, expert d'action entraîné, chunks de 50 actions). Doc : batch 64, 20k pas ≈ 4 h sur A100.
- 0.5.1 n'a ni `lerobot-rollout` ni `--inference.type=rtc` : rollout = `lerobot-record --policy.path` avec un dataset `eval_...` ; inférence déportée = `lerobot.async_inference` (policy_server / robot_client).
- `lerobot-train` exige `--policy.repo_id` dès que `push_to_hub` est vrai (vrai dans smolvla_base) : `--policy.push_to_hub=false` par défaut.
- L'image du challenge n'a pas `num2words` (extra `lerobot[smolvla]`), requis par le processeur SmolVLM : couche ajoutée par `docker/Dockerfile`.
- Conteneurs non privilégiés : en mode privileged, Docker ignore un `--device` vers un chemin déjà présent (ttyACM0 = premier bras branché, pas forcément le follower).
- Touches de lerobot-record (→ ← Échap) : pynput via X11. Session Wayland : lancer depuis le terminal de VS Code (client X11), pas depuis le Terminal GNOME.