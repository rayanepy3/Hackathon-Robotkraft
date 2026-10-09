# RobotKraft : SO-101 leader/follower -> dataset LeRobot -> fine-tuning SmolVLA -> rollout.
# Point d'entrée unique : `make` liste les cibles, phase par phase.
# Configuration : config/robotkraft.env (+ config/robotkraft.env.local, non versionné).
# Cibles [terminal] : interactives (Entrée, flèches, mouvements du bras), à lancer soi-même.
# Aucune cible n'appelle lerobot-setup-motors : les bras du challenge sont déjà configurés.

SHELL := /bin/bash
.DEFAULT_GOAL := help

include config/robotkraft.env
-include config/robotkraft.env.local

# ---------------------------------------------------------------------------
# Exécution. RUNNER=docker : conteneur NON privilégié avec uniquement les devices
# utiles. En mode privileged, Docker ignore un --device dont le chemin existe déjà
# sur l'hôte : /dev/ttyACM0 serait le premier bras branché, pas forcément le follower.
# ---------------------------------------------------------------------------
TTY_FLAGS := $(shell [ -t 0 ] && echo -it || echo -i)
GROUP_ADD := $(foreach g,dialout video plugdev,$(if $(shell getent group $(g)),--group-add $(shell getent group $(g) | cut -d: -f3)))
DOCKER_OPTS = --user $(shell id -u):$(shell id -g) $(GROUP_ADD) \
	-e HOME=/workspace/.cache/home -e USER=$(shell id -un) -e LOGNAME=$(shell id -un) -v $(CURDIR):/workspace/robotkraft -w /workspace/robotkraft \
	-v $(abspath $(CACHE_DIR)):/workspace/.cache --shm-size=2g
DOCKER_RUN = docker run --rm --init $(TTY_FLAGS) $(DOCKER_OPTS)

export CAM_WIDTH CAM_HEIGHT CAM_FPS CAM1_ZMQ_PORT CAM1_ZMQ_NAME
ifeq ($(RUNNER),native)
export HF_HOME := $(abspath $(CACHE_DIR))/huggingface
export HF_LEROBOT_HOME := $(HF_HOME)/lerobot
run =
run_gpu =
else
run = $(DOCKER_RUN) $(1) $(IMAGE)
run_gpu = $(DOCKER_RUN) --gpus all $(1) $(TRAIN_IMAGE)
endif
TRAIN_IMAGE ?= robotkraft-smolvla:cuda
# Rollout : image CUDA + GPU si POLICY_DEVICE=cuda (machine NVIDIA), sinon image CPU.
run_policy = $(if $(filter cuda,$(POLICY_DEVICE)),$(call run_gpu,$(1)),$(call run,$(1)))

DEV_FOLLOWER = --device $(FOLLOWER_PORT):$(FOLLOWER_PORT)
DEV_LEADER   = --device $(LEADER_PORT):$(LEADER_PORT)
DEV_WRIST    = --device $(WRIST_CAM):$(WRIST_CAM)
# OAK-D : depthai redémarre le device USB (nouveau nœud) -> tout /dev/bus/usb + règle cgroup.
# Réseau de l'hôte : le pont OAK (localhost:5555) est partagé entre conteneurs (pont permanent des épreuves).
DEV_OAK      = --network host -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw'
DEV_ALL_VIDEO = $(foreach d,$(wildcard /dev/video*),--device $(d):$(d)) $(if $(wildcard /dev/v4l/by-id),-v /dev/v4l:/host-v4l:ro)
# X11 : nécessaire aux touches de lerobot-record (→ ← Échap, via pynput).
DEV_X11      = -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix:ro $(if $(XAUTHORITY),-e XAUTHORITY=/tmp/.Xauthority -v $(XAUTHORITY):/tmp/.Xauthority:ro)
OAK_ENV      = -e CAM_WIDTH=$(CAM_WIDTH) -e CAM_HEIGHT=$(CAM_HEIGHT) -e CAM_FPS=$(CAM_FPS) -e CAM1_ZMQ_PORT=$(CAM1_ZMQ_PORT) -e CAM1_ZMQ_NAME=$(CAM1_ZMQ_NAME)

ROBOT_ARGS  = --robot.type=so101_follower --robot.port=$(FOLLOWER_PORT) --robot.id=$(FOLLOWER_ID)
# Options du follower en plus (épreuves : --robot.disable_torque_on_disconnect=false, le bras tient sa pose entre deux essais).
ROBOT_EXTRA ?=
# rollout-async : arrêt du client au bout de ROLLOUT_DURATION_S secondes (0 = jamais, Ctrl+C).
ROLLOUT_DURATION_S ?= 0
TELEOP_ARGS = --teleop.type=so101_leader --teleop.port=$(LEADER_PORT) --teleop.id=$(LEADER_ID)
# Noms camera1/camera2 dès l'enregistrement = clés de lerobot/smolvla_base, sans rename_map.
CAMERAS = {camera1: {type: zmq, server_address: localhost, port: $(CAM1_ZMQ_PORT), camera_name: $(CAM1_ZMQ_NAME), width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS)}, camera2: {type: opencv, index_or_path: $(WRIST_CAM), fourcc: MJPG, width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS)}}

CALIB_DIR      = $(CACHE_DIR)/huggingface/lerobot/calibration
CALIB_FOLLOWER = $(CALIB_DIR)/robots/so_follower/$(FOLLOWER_ID).json
CALIB_LEADER   = $(CALIB_DIR)/teleoperators/so_leader/$(LEADER_ID).json
DATASET_DIR    = $(CACHE_DIR)/huggingface/lerobot/$(DATASET_REPO_ID)
EVAL_REPO_ID   = $(HF_USER)/eval_$(TASK_NAME)_$(shell date +%m%d_%H%M%S)
comma := ,
space := $(subst ,, )
RUN       ?= $(TASK_NAME)
CONDITION ?= nominal

.PHONY: help install install-native install-cuda doctor ports ports-snapshot ports-diff cameras view-oak view-wrist oak-angle hf-login hf-whoami \
	udev import-calib calib-status calibrate-follower calibrate-leader teleop \
	record record-resume record-tri rec tri-b tri-j tri-r check-dataset view-dataset delete-episodes push-dataset train-cmd train train-act train-resume train-hf-job push-model \
	rollout policy-local tri tri-calib tri-voir etiquette etiquette-calib etiquette-voir etiq-rec etiq-saisir etiq-regarder etiq-montrer etiq-reposer install-epreuves need-epreuves oak-bridge-start oak-bridge-stop policy-server rollout-async eval-log eval-report setup-motors prep \
	need-follower need-leader need-wrist need-oak need-hf-user need-image need-calib need-calib-follower

help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*## "} /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36mmake %-20s\033[0m %s\n", $$1, $$2}' $(firstword $(MAKEFILE_LIST))
	@echo ""
	@echo "Config : HF_USER=$(HF_USER)  DATASET=$(DATASET_REPO_ID)  RUNNER=$(RUNNER)"
	@echo "Tâche  : $(TASK_TEXT)"

##@ Phase 0 — Installation et détection (aucun moteur)
install: ## Vérifie Docker et les groupes, construit l'image (challenge + num2words)
	BASE_IMAGE=$(BASE_IMAGE) IMAGE=$(IMAGE) bash scripts/install.sh

install-cuda: ## Machine NVIDIA avec Docker : même chose sur l'image CUDA du challenge (~9 Go)
	TRAIN_IMAGE=$(TRAIN_IMAGE) bash scripts/install.sh cuda

install-native: ## Machine GPU sans Docker (serveur) : env conda + lerobot[feetech,smolvla]==0.5.1
	bash scripts/install.sh native

doctor: ## État du poste : image, bras, caméras, calibrations, compte HF
	@echo "== Image ($(RUNNER))"; [ "$(RUNNER)" != docker ] || docker image inspect $(IMAGE) --format '  OK $(IMAGE) (créée {{.Created}})' 2>/dev/null || echo "  ABSENTE $(IMAGE) : make install"
	@echo "== Bras"; for p in $(FOLLOWER_PORT) $(LEADER_PORT); do [ -e $$p ] && echo "  OK $$p -> $$(readlink -f $$p)" || echo "  ABSENT $$p"; done
	@echo "== Caméras"; [ -e $(WRIST_CAM) ] && echo "  OK camera2 (poignet) -> $$(readlink -f $(WRIST_CAM))" || echo "  ABSENTE camera2 (poignet) $(WRIST_CAM)"; \
		grep -qs 03e7 /sys/bus/usb/devices/*/idVendor && echo "  OK camera1 (OAK-D, USB 03e7)" || echo "  ABSENTE camera1 (OAK-D)"
	@echo "== Calibrations"; for f in $(CALIB_FOLLOWER) $(CALIB_LEADER); do [ -s $$f ] && echo "  OK $$f" || echo "  ABSENTE $$f"; done
	@echo "== Hugging Face"; [ -s $(CACHE_DIR)/huggingface/token ] && echo "  token présent (make hf-whoami)" || echo "  pas de token : make hf-login"; \
		[ "$(HF_USER)" != CHANGE_ME ] && echo "  HF_USER=$(HF_USER)" || echo "  HF_USER non renseigné (config/robotkraft.env.local)"
	@echo "== Dataset"; [ -d $(DATASET_DIR) ] && echo "  $(DATASET_DIR) présent" || echo "  $(DATASET_DIR) pas encore enregistré"

ports: ## Liste les ports série USB (VID/PID, serial, chemin USB)
	@python3 scripts/detect_ports.py

ports-snapshot: ## Mémorise les ports présents, avant de brancher ou débrancher un bras
	@python3 scripts/detect_ports.py snapshot

ports-diff: ## Montre le port apparu ou disparu depuis ports-snapshot
	@python3 scripts/detect_ports.py diff

cameras: need-image prep ## Liste les caméras, capture une image de chacune dans outputs/cam_check/
	$(call run,$(DEV_ALL_VIDEO) $(DEV_OAK)) python3 scripts/detect_cameras.py \
		--width $(CAM_WIDTH) --height $(CAM_HEIGHT) --fps $(CAM_FPS) --wrist $(WRIST_CAM) --ignore "$(CAM_IGNORE)" --out outputs/cam_check

view-oak: need-oak need-image prep ## Aperçu en direct de camera1 (OAK-D) pour régler le cadrage
	$(call run,$(DEV_OAK) $(DEV_X11) $(OAK_ENV)) python3 scripts/view_oak.py

view-wrist: need-wrist need-image prep ## Aperçu en direct de camera2 (poignet) : mise au point, orientation
	$(call run,$(DEV_WRIST) $(DEV_X11) $(OAK_ENV)) python3 scripts/view_wrist.py $(WRIST_CAM)

oak-angle: need-oak need-image prep ## Inclinaison de camera1 (accéléromètre de l'OAK) ; H=.. D=.. en cm pour l'angle visé
	$(call run,$(DEV_OAK)) python3 scripts/oak_angle.py $(if $(H),--h $(H)) $(if $(D),--d $(D))

hf-login: need-image prep ## [terminal] Connexion Hugging Face (token dans .cache/, jamais dans le repo)
	$(call run,) hf auth login

hf-whoami: need-image prep ## Affiche le compte Hugging Face utilisé
	$(call run,) hf auth whoami

##@ Phase 1 — Ports stables, calibration, téléopération
udev: ## [terminal] Crée /dev/so101_follower et /dev/so101_leader (sudo après confirmation)
	bash scripts/udev_rules.sh

import-calib: ## Copie les calibrations du repo du challenge (SRC=~/robotkraft-participant/.cache)
	@src=$(or $(SRC),$(HOME)/robotkraft-participant/.cache)/huggingface/lerobot/calibration; \
	for rel in robots/so_follower/$(FOLLOWER_ID).json teleoperators/so_leader/$(LEADER_ID).json; do \
		if [ -s $$src/$$rel ]; then mkdir -p $(CALIB_DIR)/$$(dirname $$rel) && cp -v $$src/$$rel $(CALIB_DIR)/$$rel; \
		else echo "absent : $$src/$$rel"; fi; done

calib-status: ## Montre les calibrations présentes
	@for f in $(CALIB_FOLLOWER) $(CALIB_LEADER); do [ -s $$f ] && echo "OK $$f ($$(date -r $$f '+%d/%m %H:%M'))" || echo "ABSENTE $$f"; done

calibrate-follower: need-follower need-image prep ## [terminal] lerobot-calibrate du follower (une fois)
	$(call run,$(DEV_FOLLOWER)) lerobot-calibrate $(ROBOT_ARGS)

calibrate-leader: need-leader need-image prep ## [terminal] lerobot-calibrate du leader (une fois)
	$(call run,$(DEV_LEADER)) lerobot-calibrate $(TELEOP_ARGS)

teleop: need-follower need-leader need-calib need-image prep ## [terminal] Le follower suit le leader (Ctrl+C pour arrêter)
	$(call run,$(DEV_FOLLOWER) $(DEV_LEADER)) lerobot-teleoperate $(ROBOT_ARGS) $(TELEOP_ARGS)

##@ Phase 2 — Enregistrement du dataset
record: need-hf-user need-follower need-leader need-wrist need-oak need-calib need-image prep ## [terminal] lerobot-record : NUM_EPISODES épisodes, camera1 + camera2
	@[ -n "$(RECORD_EXTRA)" ] || [ ! -e $(DATASET_DIR) ] || { echo "ERREUR : $(DATASET_DIR) existe déjà. make record-resume pour compléter, ou change TASK_NAME."; exit 1; }
	@[ ! -s $(DATASET_DIR)/meta/info.json ] || python3 -c "import json,sys; f=json.load(open('$(DATASET_DIR)/meta/info.json'))['fps']; f==$(REC_FPS) or sys.exit(f'ERREUR : $(DATASET_REPO_ID) est enregistré à {f} fps, REC_FPS=$(REC_FPS). Change TASK_NAME (nouveau dataset) ou mets REC_FPS={f}.')"
	$(call run,$(DEV_FOLLOWER) $(DEV_LEADER) $(DEV_WRIST) $(DEV_OAK) $(DEV_X11) $(OAK_ENV)) bash scripts/with_oak.sh lerobot-record \
		$(ROBOT_ARGS) $(ROBOT_EXTRA) $(TELEOP_ARGS) '--robot.cameras=$(CAMERAS)' \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(TASK_TEXT)" \
		--dataset.fps=$(REC_FPS) \
		--dataset.num_episodes=$(NUM_EPISODES) \
		--dataset.episode_time_s=$(EPISODE_TIME_S) \
		--dataset.reset_time_s=$(RESET_TIME_S) \
		--dataset.push_to_hub=$(PUSH_TO_HUB) --dataset.private=$(HUB_PRIVATE) \
		--dataset.streaming_encoding=true \
		--display_data=false --play_sounds=false $(RECORD_EXTRA)

# LeRobot 0.5.1 refuse de reprendre sans dossier explicite (il n'écrit pas dans le cache du Hub).
record-resume: RECORD_EXTRA = --resume=true --dataset.root=$(DATASET_DIR)
record-resume: record ## [terminal] Ajoute NUM_EPISODES épisodes au dataset existant

record-tri: need-hf-user ## [terminal] Épreuve TRI : SEQ=BBJRRR = couleurs dans l'ordre de rangement, 1 épisode par tube
	@[ -n "$(SEQ)" ] || { echo "ERREUR : précise SEQ, ex. make record-tri SEQ=BBJRRR (B bleu, J jaune, R rouge)"; exit 1; }
	python3 scripts/tri_record.py --seq $(SEQ) --task-name $(TASK_NAME) --dataset-dir $(DATASET_DIR) \
		--text-b "$(TRI_TEXT_B)" --text-j "$(TRI_TEXT_J)" --text-r "$(TRI_TEXT_R)" $(if $(DRY),--dry-run)

# Enregistrement avec temps qui défile : N épisodes, pause RESET_TIME_S entre deux, dataset créé ou complété.
N ?= 1
REC_LIVE = python3 scripts/record_live.py --n $(N) --episode-s $(EPISODE_TIME_S) --reset-s $(RESET_TIME_S) --dataset-dir $(DATASET_DIR)
REC_TARGET = $(if $(wildcard $(DATASET_DIR)/meta/info.json),record-resume,record)

# Datasets de l'épreuve TRI : TRI_TASK_NAME (TASK_NAME=... sur la ligne de commande l'emporte).
tri-b tri-j tri-r: TASK_NAME=$(TRI_TASK_NAME)

rec: need-hf-user ## [terminal] N épisodes (TASK_TEXT), temps affiché en direct : make rec N=10
	@$(REC_LIVE) -- $(MAKE) --no-print-directory $(REC_TARGET) TASK_NAME=$(TASK_NAME) NUM_EPISODES=$(N) TASK_TEXT="$(TASK_TEXT)"
tri-b: need-hf-user ## [terminal] TRI : N tubes BLEUS, temps affiché : make tri-b N=5
	@$(REC_LIVE) --label bleu -- $(MAKE) --no-print-directory $(REC_TARGET) TASK_NAME=$(TASK_NAME) NUM_EPISODES=$(N) TASK_TEXT="$(TRI_TEXT_B)"
tri-j: need-hf-user ## [terminal] TRI : N tubes JAUNES, temps affiché : make tri-j N=5
	@$(REC_LIVE) --label jaune -- $(MAKE) --no-print-directory $(REC_TARGET) TASK_NAME=$(TASK_NAME) NUM_EPISODES=$(N) TASK_TEXT="$(TRI_TEXT_J)"
tri-r: need-hf-user ## [terminal] TRI : N tubes ROUGES, temps affiché : make tri-r N=5
	@$(REC_LIVE) --label rouge -- $(MAKE) --no-print-directory $(REC_TARGET) TASK_NAME=$(TASK_NAME) NUM_EPISODES=$(N) TASK_TEXT="$(TRI_TEXT_R)"

check-dataset: need-hf-user need-image prep ## Valide le dataset, trace les trajectoires (outputs/dataset_check/)
	$(call run,) python3 tools/check_dataset.py --repo-id $(DATASET_REPO_ID) \
		--fps $(REC_FPS) --width $(CAM_WIDTH) --height $(CAM_HEIGHT) \
		--task "$(TASK_TEXT)" --expected-episodes $(NUM_EPISODES) \
		--out outputs/dataset_check/$(subst /,__,$(DATASET_REPO_ID))

view-dataset: need-hf-user need-image prep ## Une vidéo par épisode (2 caméras côte à côte) dans outputs/dataset_view/, pour trier
	$(call run,) python3 tools/export_episodes.py --repo-id $(DATASET_REPO_ID) --root $(DATASET_DIR) \
		--out outputs/dataset_view/$(subst /,__,$(DATASET_REPO_ID))

# lerobot-edit-dataset : --root et --new_root identiques = modification sur place, l'original part en <dataset>_old.
delete-episodes: need-hf-user need-image prep ## Supprime des épisodes en local : EPISODES="3 7" (sauvegarde dans <dataset>_old)
	@[ -n "$(EPISODES)" ] || { echo 'ERREUR : précise les numéros, ex. make delete-episodes EPISODES="3 7" (voir make view-dataset)'; exit 1; }
	$(call run,) lerobot-edit-dataset --repo_id $(DATASET_REPO_ID) --root $(DATASET_DIR) --new_root $(DATASET_DIR) \
		--operation.type delete_episodes --operation.episode_indices "[$(subst $(space),$(comma),$(strip $(EPISODES)))]"
	@echo "Fait. Ancienne version : $(DATASET_DIR)_old. Les épisodes suivants sont renumérotés : relance make view-dataset."

push-dataset: need-hf-user need-image prep ## Pousse le dataset sur le Hub (privé si HUB_PRIVATE=true)
	$(call run,) python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
		LeRobotDataset('$(DATASET_REPO_ID)').push_to_hub(private=$(if $(filter true,$(HUB_PRIVATE)),True,False))"

##@ Phase 3 — Fine-tuning SmolVLA (machine avec GPU NVIDIA)
train-cmd: need-hf-user ## Affiche les commandes à coller dans le terminal du serveur GPU (JupyterLab)
	@echo "# 1. Une fois : lerobot 0.5.1 avec SmolVLA (sauter si 'lerobot-train --help' marche déjà)"
	@echo "pip install 'lerobot[smolvla]==0.5.1'"
	@echo "# 2. Une fois : token Hugging Face 'write' (lecture du dataset privé, envoi du modèle)"
	@echo "hf auth login"
	@echo "# 3. Entraînement en arrière-plan (survit à la fermeture de l'onglet), modèle poussé sur $(MODEL_REPO_ID)"
	@echo "nohup lerobot-train --policy.path=$(POLICY_BASE) --dataset.repo_id=$(DATASET_REPO_ID) \\"
	@echo "  --batch_size=$(BATCH_SIZE) --steps=$(TRAIN_STEPS) --save_freq=$(SAVE_FREQ) \\"
	@echo "  --output_dir=outputs/train/smolvla_$(TASK_NAME) --job_name=smolvla_$(TASK_NAME) \\"
	@echo "  --policy.device=cuda --wandb.enable=false \\"
	@echo "  --policy.push_to_hub=true --policy.repo_id=$(MODEL_REPO_ID) > train_$(TASK_NAME).log 2>&1 &"
	@echo "# 4. Suivi : tail -f train_$(TASK_NAME).log ; durée totale ≈ $(TRAIN_STEPS) x updt_s (affiché à chaque log) ; nvidia-smi"
	@echo "#    Tester un checkpoint intermédiaire sur le PC (push_to_hub n'envoie qu'à la fin) :"
	@echo "hf upload $(MODEL_REPO_ID) outputs/train/smolvla_$(TASK_NAME)/checkpoints/last/pretrained_model --repo-type model --private"
	@echo "# 5. Secours, en parallèle sur une autre instance GPU : ACT sur le même dataset (inférence fluide sur CPU)"
	@echo "nohup lerobot-train --policy.type=act --dataset.repo_id=$(DATASET_REPO_ID) \\"
	@echo "  --batch_size=$(ACT_BATCH_SIZE) --steps=$(ACT_STEPS) --save_freq=$(ACT_SAVE_FREQ) \\"
	@echo "  --output_dir=outputs/train/act_$(TASK_NAME) --job_name=act_$(TASK_NAME) \\"
	@echo "  --policy.device=cuda --wandb.enable=false \\"
	@echo "  --policy.push_to_hub=true --policy.repo_id=$(ACT_MODEL_REPO_ID) > train_act_$(TASK_NAME).log 2>&1 &"

train: need-hf-user prep ## lerobot-train depuis lerobot/smolvla_base
	@command -v nvidia-smi >/dev/null && nvidia-smi -L || { echo "ERREUR : pas de GPU NVIDIA ici. Entraîne sur le serveur GPU (RUNNER=native) ; payant : make train-hf-job."; exit 1; }
	$(call run_gpu,) lerobot-train \
		--policy.path=$(POLICY_BASE) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--batch_size=$(BATCH_SIZE) --steps=$(TRAIN_STEPS) --save_freq=$(SAVE_FREQ) \
		--output_dir=$(TRAIN_OUTPUT_DIR) --job_name=smolvla_$(TASK_NAME) \
		--policy.device=cuda --wandb.enable=false \
		--policy.push_to_hub=$(PUSH_MODEL) $(if $(filter true,$(PUSH_MODEL)),--policy.repo_id=$(MODEL_REPO_ID))

train-act: need-hf-user prep ## Secours : ACT sur le même dataset (inférence fluide sur CPU)
	@command -v nvidia-smi >/dev/null && nvidia-smi -L || { echo "ERREUR : pas de GPU NVIDIA ici. make train-cmd affiche la commande pour le serveur GPU."; exit 1; }
	$(call run_gpu,) lerobot-train --policy.type=act --dataset.repo_id=$(DATASET_REPO_ID) \
		--batch_size=$(ACT_BATCH_SIZE) --steps=$(ACT_STEPS) --save_freq=$(ACT_SAVE_FREQ) \
		--output_dir=$(ACT_OUTPUT_DIR) --job_name=act_$(TASK_NAME) \
		--policy.device=cuda --wandb.enable=false \
		--policy.push_to_hub=$(PUSH_MODEL) $(if $(filter true,$(PUSH_MODEL)),--policy.repo_id=$(ACT_MODEL_REPO_ID))

train-resume: prep ## Reprend l'entraînement au dernier checkpoint
	$(call run_gpu,) lerobot-train --config_path=$(TRAIN_OUTPUT_DIR)/checkpoints/last/pretrained_model/train_config.json --resume=true

HF_JOB_IMAGE  = python:3.12
HF_JOB_SCRIPT = apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null && pip install -q 'lerobot[smolvla]==0.5.1' && \
	lerobot-train --policy.path=$(POLICY_BASE) --dataset.repo_id=$(DATASET_REPO_ID) \
	--batch_size=$(BATCH_SIZE) --steps=$(TRAIN_STEPS) --save_freq=$(SAVE_FREQ) --output_dir=/tmp/train \
	--job_name=smolvla_$(TASK_NAME) --policy.device=cuda --wandb.enable=false \
	--policy.push_to_hub=true --policy.repo_id=$(MODEL_REPO_ID)
train-hf-job: need-hf-user need-image prep ## PAYANT : entraînement HF Jobs, refusé sans CONFIRM_PAID=yes
	@if [ "$(CONFIRM_PAID)" != yes ]; then \
		echo "PAYANT : HF Jobs $(HF_JOB_FLAVOR), facturé à la seconde, coupé au bout de $(HF_JOB_TIMEOUT)."; \
		echo "Prérequis : dataset sur le Hub (make push-dataset). Le modèle sera poussé sur $(MODEL_REPO_ID)."; \
		echo "Commande qui serait lancée :"; \
		echo "  hf jobs run --flavor $(HF_JOB_FLAVOR) --timeout $(HF_JOB_TIMEOUT) --secrets HF_TOKEN -d $(HF_JOB_IMAGE) bash -c \"$(HF_JOB_SCRIPT)\""; \
		echo "Pour lancer réellement : make train-hf-job CONFIRM_PAID=yes"; exit 1; fi
	$(call run,) hf jobs run --flavor $(HF_JOB_FLAVOR) --timeout $(HF_JOB_TIMEOUT) --secrets HF_TOKEN -d \
		$(HF_JOB_IMAGE) bash -c "$(HF_JOB_SCRIPT)"

push-model: need-hf-user need-image prep ## Pousse le dernier checkpoint local (TRAIN_OUTPUT_DIR) sur le Hub
	$(call run,) hf upload $(MODEL_REPO_ID) $(TRAIN_OUTPUT_DIR)/checkpoints/last/pretrained_model --repo-type model $(if $(filter true,$(HUB_PRIVATE)),--private)

##@ Épreuves — un workflow complet par épreuve (vision + modèle + vérification + reprise)
EP_PY = env PYTHONPATH=$(CURDIR)/COMMUN .venv-epreuves/bin/python
EPREUVE_SECONDS ?= $(if $(ASYNC_SERVER),25,90)
EP_COMMON = --seconds $(EPREUVE_SECONDS) --async-server "$(ASYNC_SERVER)" --max-tries $(MAX_TRIES) $(if $(DRY),--dry-run)
ORDER ?= $(TRI_ORDER)

install-epreuves: ## Python des épreuves (vision, lecture d'étiquettes, MQTT) dans .venv-epreuves
	python3 -m venv .venv-epreuves && env -u PYTHONPATH .venv-epreuves/bin/pip install -q opencv-python numpy pyzmq rapidocr-onnxruntime paho-mqtt

need-epreuves:
	@[ -x .venv-epreuves/bin/python ] || { echo "ERREUR : make install-epreuves d'abord"; exit 1; }

tri: need-epreuves need-follower need-wrist need-oak need-calib-follower need-image prep ## [terminal] Épreuve TRI complète : make tri ORDER=BJR (DRY=1 : sans bouger le bras)
	$(EP_PY) TRI/tri.py run --order $(ORDER) --policy $(TRI_POLICY) --fps $(TRI_POLICY_FPS) $(EP_COMMON) \
		--text-b "$(TRI_TEXT_B)" --text-j "$(TRI_TEXT_J)" --text-r "$(TRI_TEXT_R)"

tri-calib: need-epreuves need-oak need-image prep ## TRI : cliquer les 6 places du porte-tubes et la zone de la table (une fois)
	$(EP_PY) TRI/tri.py calib

tri-voir: need-epreuves need-oak need-image prep ## TRI : vision en direct (couleurs vues sur la table et dans le porte-tubes)
	$(EP_PY) TRI/tri.py voir

etiquette: need-epreuves need-follower need-wrist need-oak need-calib-follower need-image prep ## [terminal] Épreuve ETIQ : make etiquette MOT=IODURE [OU=rack|table] [LECTURE_CAM=oak|poignet]
	@[ -n "$(MOT)" ] || { echo "ERREUR : précise le mot, ex. make etiquette MOT=IODURE"; exit 1; }
	$(EP_PY) ETIQ/etiq.py run --mot "$(MOT)" --ou $(or $(OU),rack) --lecture $(LECTURE_CAM) --policy $(ETIQ_POLICY) --fps $(ETIQ_POLICY_FPS) \
		$(subst --fps-rec $(REC_FPS) ,,$(EP_COMMON)) --text-slot "$(ETIQ_TEXT_SLOT)" --text-zone "$(ETIQ_TEXT_ZONE)"

etiquette-calib: need-epreuves need-oak need-image prep ## ETIQ : cliquer les places du rack et les zones de la table (une fois)
	$(EP_PY) ETIQ/etiq.py calib

etiquette-voir: need-epreuves need-image prep ## ETIQ : lecture des étiquettes en direct (LECTURE_CAM=oak|poignet)
	$(EP_PY) ETIQ/etiq.py voir --lecture $(LECTURE_CAM)

# Sous-gestes ETIQ : un dataset par sous-geste ; le follower garde sa pose à la fin de la commande.
SKILL_REC = $(MAKE) --no-print-directory rec N=$(N) EPISODE_TIME_S=$(SKILL_TIME_S) ROBOT_EXTRA=--robot.disable_torque_on_disconnect=false
etiq-saisir: need-hf-user ## [terminal] ETIQ 1/3 : repos -> tube de la place k saisi et levé : make etiq-saisir PLACE=3 N=5
	@[ -n "$(PLACE)" ] || { echo "ERREUR : PLACE=1..6"; exit 1; }
	@$(SKILL_REC) TASK_NAME=etiq_saisir TASK_TEXT="$(subst {k},$(PLACE),$(ETIQ_TEXT_SLOT))"
etiq-regarder: need-hf-user ## [terminal] ETIQ lire : repos -> caméra poignet au-dessus du rack, immobile : make etiq-regarder N=5
	@$(SKILL_REC) TASK_NAME=etiq_lire TASK_TEXT="$(ETIQ_TEXT_LOOK)"
etiq-montrer: need-hf-user ## [terminal] ETIQ lire : tube levé -> tourné lentement devant l'OAK : make etiq-montrer N=5
	@$(SKILL_REC) TASK_NAME=etiq_lire TASK_TEXT="$(ETIQ_TEXT_SHOW)"
etiq-reposer: need-hf-user ## [terminal] ETIQ 3/3 : devant l'OAK -> remis en place k -> repos : make etiq-reposer PLACE=3 N=5
	@[ -n "$(PLACE)" ] || { echo "ERREUR : PLACE=1..6"; exit 1; }
	@$(SKILL_REC) TASK_NAME=etiq_reposer TASK_TEXT="$(subst {k},$(PLACE),$(ETIQ_TEXT_PLACE))"

etiq-rec: need-hf-user ## [terminal] ETIQ : N saisies enregistrées, PLACE=k (rack) ou ZONE=k (table) : make etiq-rec PLACE=3 N=5
	@[ -n "$(PLACE)$(ZONE)" ] || { echo "ERREUR : PLACE=1..6 (tube du rack) ou ZONE=k (objet sur la table)"; exit 1; }
	@$(MAKE) --no-print-directory rec TASK_NAME=$(ETIQ_TASK_NAME) N=$(N) \
		TASK_TEXT="$(subst {k},$(or $(PLACE),$(ZONE)),$(if $(PLACE),$(ETIQ_TEXT_SLOT),$(ETIQ_TEXT_ZONE)))"

##@ Phase 4 — Rollout et taux de succès
rollout: need-hf-user need-follower need-wrist need-oak need-calib-follower need-image prep ## [terminal] La policy pilote le follower, EVAL_EPISODES essais
	@[[ "$(POLICY_PATH)" != outputs/* ]] || [ -s $(POLICY_PATH)/config.json ] || { echo "ERREUR : policy introuvable : $(POLICY_PATH)"; exit 1; }
	$(call run_policy,$(DEV_FOLLOWER) $(DEV_WRIST) $(DEV_OAK) $(DEV_X11) $(OAK_ENV)) bash scripts/with_oak.sh lerobot-record \
		$(ROBOT_ARGS) $(ROBOT_EXTRA) '--robot.cameras=$(CAMERAS)' \
		--policy.path=$(POLICY_PATH) --policy.device=$(POLICY_DEVICE) \
		--dataset.repo_id=$(EVAL_REPO_ID) \
		--dataset.single_task="$(TASK_TEXT)" \
		--dataset.fps=$(REC_FPS) \
		--dataset.num_episodes=$(EVAL_EPISODES) \
		--dataset.episode_time_s=$(EVAL_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(EVAL_RESET_TIME_S) \
		--dataset.push_to_hub=false --dataset.streaming_encoding=true \
		--display_data=false --play_sounds=false

policy-local: need-image prep ## Copie locale d'un modèle du Hub lisible par le LeRobot du PC : make policy-local POLICY=dbal67/Robotkraft_TRI_v2
	@[ -n "$(POLICY)" ] || { echo "ERREUR : précise POLICY=dépôt du Hub"; exit 1; }
	$(call run,) python3 scripts/policy_local.py $(POLICY)

oak-bridge-start: need-oak need-image prep ## Pont OAK permanent (épreuves) : camera1 sur localhost:5555, partagée vision + modèle
	@if docker ps -q -f name='^rk-oak$$' | grep -q .; then echo "Pont OAK déjà lancé (rk-oak)"; else \
		docker run -d --rm --init --name rk-oak $(DOCKER_OPTS) $(DEV_OAK) $(OAK_ENV) -e OAK_READY_FILE=/tmp/oak_ready_bridge -e OAK_PERSIST=1 -e PYTHONUNBUFFERED=1 \
		$(IMAGE) python3 scripts/oak_zmq_server.py >/dev/null && echo "Pont OAK lancé (rk-oak)"; fi

oak-bridge-stop: ## Arrête le pont OAK permanent
	@docker stop rk-oak >/dev/null 2>&1 && echo "Pont OAK arrêté" || echo "Pont OAK pas lancé"

policy-server: prep ## Machine GPU : serveur d'inférence asynchrone officiel (port 8080)
	$(if $(filter docker,$(RUNNER)),$(DOCKER_RUN) --gpus all -p 8080:8080 $(TRAIN_IMAGE)) \
		python3 -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=$(REC_FPS)

rollout-async: need-follower need-wrist need-oak need-calib-follower need-image prep ## [terminal] Rollout avec inférence sur la machine GPU : ASYNC_SERVER=ip:8080 (tunnel SSH : localhost:8080)
	@[ -n "$(ASYNC_SERVER)" ] || { echo "ERREUR : précise ASYNC_SERVER=ip:8080 (machine qui fait tourner make policy-server)"; exit 1; }
	$(call run,-e ROLLOUT_DURATION_S=$(ROLLOUT_DURATION_S) $(DEV_FOLLOWER) $(DEV_WRIST) $(DEV_OAK) $(OAK_ENV)) bash scripts/with_oak.sh \
		python3 scripts/robot_client_zmq.py \
		$(ROBOT_ARGS) $(ROBOT_EXTRA) '--robot.cameras=$(CAMERAS)' \
		--task="$(TASK_TEXT)" --server_address=$(ASYNC_SERVER) \
		--policy_type=smolvla --pretrained_name_or_path=$(ASYNC_POLICY) --policy_device=cuda \
		--actions_per_chunk=50 --chunk_size_threshold=0.5 --fps=$(REC_FPS)

eval-log: ## [terminal] Note chaque essai : make eval-log RUN=v1 CONDITION=nominal
	python3 tools/eval_logger.py log --run "$(RUN)" --condition "$(CONDITION)" --policy "$(POLICY_PATH)" --task "$(TASK_TEXT)"

eval-report: ## Tableau et graphique des taux de succès (outputs/eval/)
	python3 tools/eval_logger.py report

# Valeur d'une variable (utilisé par COMMUN/commun.py) : make -s print-WRIST_CAM
print-%:
	@echo '$($*)'

# ---------------------------------------------------------------------------
# Garde-fous
# ---------------------------------------------------------------------------
setup-motors:
	@echo "REFUSÉ : lerobot-setup-motors réécrit l'EEPROM des moteurs (ids, baudrate)."
	@echo "Les bras du challenge sont déjà configurés. Servo muet ou mal numéroté : va voir un coach."
	@exit 1

prep:
	@mkdir -p $(CACHE_DIR)/home outputs

need-follower:
	@test -e $(FOLLOWER_PORT) || { echo "ERREUR : $(FOLLOWER_PORT) absent. Follower branché (USB + alim 12 V) ? Règles udev posées (make udev) ?"; exit 1; }
need-leader:
	@test -e $(LEADER_PORT) || { echo "ERREUR : $(LEADER_PORT) absent. Leader branché (USB + alim 5 V) ? Règles udev posées (make udev) ?"; exit 1; }
need-wrist:
	@test -e $(WRIST_CAM) || { echo "ERREUR : caméra poignet $(WRIST_CAM) absente. make cameras, puis corrige WRIST_CAM."; exit 1; }
# L'OAK se débranche et se rebranche seule (redémarrage) après chaque programme : on lui laisse OAK_WAIT_S secondes.
OAK_WAIT_S ?= 20
need-oak:
	@for i in $$(seq 1 $(OAK_WAIT_S)); do grep -qs 03e7 /sys/bus/usb/devices/*/idVendor && exit 0; \
		[ $$i = 1 ] && echo "OAK-D absente : j'attends qu'elle revienne sur l'USB (elle redémarre après chaque programme)..."; sleep 1; done; \
		echo "ERREUR : OAK-D (USB 03e7) non détectée après $(OAK_WAIT_S) s. Branche-la, de préférence sur un port USB 3."; exit 1
need-hf-user:
	@[ "$(HF_USER)" != CHANGE_ME ] || { echo "ERREUR : HF_USER non renseigné. Exemple : echo HF_USER=monpseudo >> config/robotkraft.env.local"; exit 1; }
need-image:
	@[ "$(RUNNER)" != docker ] || docker image inspect $(IMAGE) >/dev/null 2>&1 || { echo "ERREUR : image $(IMAGE) absente. Lance make install."; exit 1; }
need-calib-follower:
	@[ -s $(CALIB_FOLLOWER) ] || { echo "ERREUR : calibration absente : $(CALIB_FOLLOWER). make calibrate-follower (ou make import-calib)."; exit 1; }
need-calib: need-calib-follower
	@[ -s $(CALIB_LEADER) ] || { echo "ERREUR : calibration absente : $(CALIB_LEADER). make calibrate-leader (ou make import-calib)."; exit 1; }