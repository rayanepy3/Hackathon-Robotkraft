#!/usr/bin/env bash
# Crée des symlinks stables /dev/so101_follower et /dev/so101_leader (règle udev, Linux).
# Les deux bras ont la même puce USB (mêmes VID:PID) : on les identifie en les branchant
# un par un, comme lerobot-find-port. La règle utilise le numéro de série si les deux
# diffèrent, sinon le port USB physique (chaque bras devra alors toujours revenir au même port).
# sudo n'est appelé qu'après confirmation, pour écrire la règle et recharger udev.
set -euo pipefail

RULE_FILE=/etc/udev/rules.d/99-so101.rules

ttys() { ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sort || true; }

wait_new_tty() {  # $1 = ports présents avant ; attend jusqu'à 15 s qu'un nouveau port apparaisse
    local new
    for _ in $(seq 1 30); do
        new=$(comm -13 <(echo "$1") <(ttys) | head -1)
        [ -n "$new" ] && { echo "$new"; return 0; }
        sleep 0.5
    done
    return 1
}

prop() { udevadm info -q property -n "$1" | sed -n "s/^$2=//p"; }

rule_for() {  # $1 = port, $2 = nom du symlink, $3 = méthode (serial | path)
    if [ "$3" = serial ]; then
        echo "SUBSYSTEM==\"tty\", ENV{ID_VENDOR_ID}==\"$(prop "$1" ID_VENDOR_ID)\", ENV{ID_MODEL_ID}==\"$(prop "$1" ID_MODEL_ID)\", ENV{ID_SERIAL_SHORT}==\"$(prop "$1" ID_SERIAL_SHORT)\", SYMLINK+=\"$2\", MODE=\"0666\""
    else
        echo "SUBSYSTEM==\"tty\", ENV{ID_PATH}==\"$(prop "$1" ID_PATH)\", SYMLINK+=\"$2\", MODE=\"0666\""
    fi
}

echo "=== Ports stables pour les bras SO-101 ==="
read -rp "1) Débranche le câble USB des DEUX bras (les alimentations peuvent rester). Entrée quand c'est fait. " _
before=$(ttys)
read -rp "2) Branche le USB du FOLLOWER seul (le bras alimenté en 12 V). Entrée. " _
F=$(wait_new_tty "$before") || { echo "Aucun nouveau port détecté pour le follower."; exit 1; }
echo "   follower -> $F   (serial=$(prop "$F" ID_SERIAL_SHORT), port USB=$(prop "$F" ID_PATH))"
before=$(ttys)
read -rp "3) Branche maintenant le USB du LEADER (le bras alimenté en 5 V). Entrée. " _
L=$(wait_new_tty "$before") || { echo "Aucun nouveau port détecté pour le leader."; exit 1; }
echo "   leader   -> $L   (serial=$(prop "$L" ID_SERIAL_SHORT), port USB=$(prop "$L" ID_PATH))"

F_SERIAL=$(prop "$F" ID_SERIAL_SHORT)
L_SERIAL=$(prop "$L" ID_SERIAL_SHORT)
if [ -n "$F_SERIAL" ] && [ -n "$L_SERIAL" ] && [ "$F_SERIAL" != "$L_SERIAL" ]; then
    method=serial
    echo "Numéros de série distincts : règle par numéro de série, indépendante du port USB."
else
    method=path
    [ -n "$(prop "$F" ID_PATH)" ] && [ -n "$(prop "$L" ID_PATH)" ] || { echo "ERREUR : ni numéro de série distinct ni chemin USB : impossible de distinguer les bras."; exit 1; }
    echo "Numéros de série identiques ou absents : règle par port USB physique."
    echo "ATTENTION : rebrancher toujours le follower et le leader sur ces mêmes ports (étiquette-les)."
fi
rules="# Généré par robotkraft/scripts/udev_rules.sh le $(date '+%F %T')
$(rule_for "$F" so101_follower $method)
$(rule_for "$L" so101_leader $method)"

echo ""
echo "Règles proposées pour $RULE_FILE :"
echo "$rules"
echo ""
read -rp "Écrire ce fichier avec sudo et recharger udev ? [o/N] " answer
[[ "$answer" =~ ^[oOyY]$ ]] || { echo "Rien n'a été écrit."; exit 0; }

echo "$rules" | sudo tee "$RULE_FILE" >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=add
sleep 2

status=0
for pair in "so101_follower:$F" "so101_leader:$L"; do
    link=/dev/${pair%%:*} expected=${pair#*:}
    if [ "$(readlink -f "$link" 2>/dev/null)" = "$expected" ]; then echo "OK   $link -> $expected"
    else echo "KO   $link absent ou ne pointe pas vers $expected (débranche/rebranche le bras)"; status=1; fi
done
exit $status