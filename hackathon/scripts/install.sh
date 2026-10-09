#!/usr/bin/env bash
# Installation de l'environnement d'exécution, avec des outils gratuits uniquement.
#   bash scripts/install.sh          portable : Docker + image du challenge (CPU) + couche smolvla
#   bash scripts/install.sh cuda     machine NVIDIA avec Docker : même chose sur l'image CUDA (~9 Go)
#   bash scripts/install.sh native   machine GPU sans Docker (serveur) : env conda Miniforge + lerobot
# Ne lance jamais sudo : affiche la commande à taper quand un droit manque.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-docker}
LEROBOT_SPEC='lerobot[feetech,smolvla]==0.5.1'
CHALLENGE_IMAGE=ghcr.io/alsacedigitale/robotkraft

ok()   { printf '  \033[32mOK\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31mERREUR\033[0m %s\n' "$*" >&2; exit 1; }

check_groups() {
    for g in dialout; do
        if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then ok "groupe $g"
        else warn "groupe $g manquant : sudo usermod -aG $g $USER, puis se déconnecter et se reconnecter"; fi
    done
}

docker_mode() {  # $1 = image de base, $2 = image construite, $3 = options GPU de docker run
    local base=$1 image=$2 gpu=${3:-}
    echo "== Docker"
    command -v docker >/dev/null || die "Docker absent. Ubuntu : https://docs.docker.com/engine/install/ubuntu/"
    docker info >/dev/null 2>&1 || die "Docker inaccessible. sudo usermod -aG docker $USER, puis se reconnecter."
    ok "$(docker --version)"
    check_groups
    echo "== Image de base $base"
    if docker image inspect "$base" >/dev/null 2>&1; then ok "déjà présente"
    else echo "  téléchargement (gratuit, plusieurs Go)..."; docker pull "$base"; fi
    echo "== Image $image = base + num2words (dépendance de lerobot[smolvla] absente de la base)"
    docker build -q --network host --build-arg BASE_IMAGE="$base" -t "$image" docker/ >/dev/null
    # shellcheck disable=SC2086
    docker run --rm $gpu "$image" python3 -c "import importlib.metadata as m, torch, num2words
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('  OK lerobot', m.version('lerobot'), '| torch', torch.__version__, '| GPU visible :', torch.cuda.is_available())"
}

native_mode() {
    echo "== Environnement conda 'robotkraft' : Python 3.12 + ffmpeg + $LEROBOT_SPEC"
    local conda
    conda=$(command -v conda || true)
    [ -n "$conda" ] || { [ -x "$HOME/miniforge3/bin/conda" ] && conda=$HOME/miniforge3/bin/conda; } || true
    if [ -z "$conda" ]; then
        echo "  Miniforge absent : installation dans ~/miniforge3 (gratuit, sans sudo)"
        curl -fsSL -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
        conda=$HOME/miniforge3/bin/conda
    fi
    "$conda" env list | grep -q '^robotkraft ' || "$conda" create -y -n robotkraft python=3.12
    "$conda" install -y -n robotkraft -c conda-forge ffmpeg
    "$conda" run -n robotkraft --no-capture-output pip install "$LEROBOT_SPEC"
    "$conda" run -n robotkraft --no-capture-output python -c "import importlib.metadata as m, torch
print('  OK lerobot', m.version('lerobot'), '| torch', torch.__version__, '| GPU visible :', torch.cuda.is_available())"
    echo "Ensuite : conda activate robotkraft (ou source ~/miniforge3/bin/activate robotkraft), puis make train RUNNER=native"
}

case "$MODE" in
    docker) docker_mode "${BASE_IMAGE:-$CHALLENGE_IMAGE:latest}" "${IMAGE:-robotkraft-smolvla:local}" ;;
    cuda)   docker_mode "$CHALLENGE_IMAGE:cuda" "${TRAIN_IMAGE:-robotkraft-smolvla:cuda}" "--gpus all" ;;
    native) native_mode ;;
    *) die "mode inconnu : $MODE (docker | cuda | native)" ;;
esac

mkdir -p .cache/home outputs
echo "Terminé. Suite : make doctor, puis make cameras."