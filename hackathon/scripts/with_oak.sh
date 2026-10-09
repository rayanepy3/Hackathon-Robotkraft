#!/usr/bin/env bash
# Lance le pont OAK-D -> ZMQ (camera1) en arrière-plan, attend sa première image,
# exécute la commande passée en arguments (lerobot-record, robot_client...), puis arrête le pont.
# L'OAK plante parfois au démarrage (lien USB 2) : jusqu'à OAK_TRIES essais, avec une pause pour qu'elle redémarre.
# Usage : bash scripts/with_oak.sh lerobot-record --robot.type=... ...
set -uo pipefail
cd "$(dirname "$0")/.."

export OAK_READY_FILE=${OAK_READY_FILE:-/tmp/oak_zmq_ready}
OAK_TRIES=${OAK_TRIES:-4}
OAK_PID=""
stop_bridge() {
    [ -n "$OAK_PID" ] && { kill "$OAK_PID" 2>/dev/null || true; wait "$OAK_PID" 2>/dev/null || true; }
    OAK_PID=""
}
trap stop_bridge EXIT

start_bridge() {
    rm -f "$OAK_READY_FILE"
    python3 scripts/oak_zmq_server.py &
    OAK_PID=$!
    for _ in $(seq 1 60); do
        [ -e "$OAK_READY_FILE" ] && return 0
        kill -0 "$OAK_PID" 2>/dev/null || return 1
        sleep 0.5
    done
    return 1
}

# Pont déjà actif sur ce port (pont permanent des épreuves, make oak-bridge-start) : on le réutilise.
if python3 - <<PY 2>/dev/null
import sys, zmq
s = zmq.Context().socket(zmq.SUB); s.setsockopt(zmq.LINGER, 0); s.setsockopt_string(zmq.SUBSCRIBE, "")
s.connect("tcp://127.0.0.1:${CAM1_ZMQ_PORT:-5555}")
sys.exit(0 if s.poll(2500) else 1)
PY
then
    echo "OAK-D : pont déjà actif sur le port ${CAM1_ZMQ_PORT:-5555}, réutilisé" >&2
    "$@"
    exit $?
fi

for try in $(seq 1 "$OAK_TRIES"); do
    start_bridge && break
    stop_bridge
    if [ "$try" = "$OAK_TRIES" ]; then
        echo "ERREUR : l'OAK-D ne démarre pas après $OAK_TRIES essais (branchée en USB 3 ? câble ? déjà utilisée par un autre programme ?)" >&2
        exit 1
    fi
    echo "OAK-D : démarrage raté (essai $try/$OAK_TRIES), nouvel essai dans 6 s..." >&2
    sleep 6
done

"$@"