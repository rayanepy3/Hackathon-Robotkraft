#!/usr/bin/env python3
"""Inclinaison de camera1 (OAK-D) mesurée par son accéléromètre intégré, pour régler ou imprimer son support.

Affiche l'angle de l'axe optique par rapport à l'horizontale (positif = la caméra regarde vers le bas,
négatif = vers le haut) et le roulis (0 = image horizontale).
Avec --h et --d (hauteur de l'objectif au-dessus de la table et distance horizontale jusqu'au centre
de la zone de travail, en cm), affiche aussi l'angle visé arctan(h/d) et la correction à apporter.
La caméra doit être immobile pendant la mesure (2 s).
"""
import argparse
import math
import time

import depthai as dai
import numpy as np

VFOV = 54.0  # champ vertical du capteur couleur de l'OAK-D Lite (degrés, constructeur)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--h", type=float, help="hauteur de l'objectif au-dessus de la table (cm)")
ap.add_argument("--d", type=float, help="distance horizontale jusqu'au centre de la zone de travail (cm)")
args = ap.parse_args()

with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    imu_model = device.getConnectedIMU()
    print(f"Caméra : {device.getDeviceName()} | IMU : {imu_model} | USB : {str(device.getUsbSpeed()).split('.')[-1]}")
    if not imu_model or imu_model.upper() == "NONE":
        raise SystemExit("Cette OAK n'a pas d'accéléromètre : mesure l'angle au rapporteur, ou calcule-le avec --h et --d.")

    # Rotation IMU -> repère caméra couleur (x à droite, y vers le bas, z = axe optique), lue en EEPROM.
    rot = np.array(device.readCalibration().getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A, True))[:3, :3]
    if abs(np.linalg.det(rot) - 1) > 0.05:
        raise SystemExit(f"Extrinsèques IMU absentes ou invalides en EEPROM :\n{np.round(rot, 3)}")

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    queue = imu.out.createOutputQueue()
    pipeline.start()
    samples, t0 = [], time.time()
    while time.time() - t0 < 2.5:
        for packet in queue.get().packets:
            a = packet.acceleroMeter
            samples.append((a.x, a.y, a.z))

acc = np.array(samples[len(samples) // 5:])  # on ignore le début (filtre du capteur qui démarre)
mean = acc.mean(axis=0)
shake = float(np.linalg.norm(acc.std(axis=0)))
up = rot @ mean / np.linalg.norm(mean)  # au repos, l'accéléromètre mesure la réaction à la gravité : vers le haut
pitch = math.degrees(-math.asin(float(np.clip(up[2], -1, 1))))  # > 0 : l'axe optique plonge sous l'horizontale
roll = math.degrees(math.atan2(float(up[0]), float(-up[1])))

print(f"Mesures : {len(acc)} ; norme {np.linalg.norm(mean):.2f} m/s² (≈ 9,81 attendu) ; agitation {shake:.2f}"
      + ("  -> caméra pas immobile, refais la mesure" if shake > 0.3 else ""))
print(f"Axe optique : {abs(pitch):.1f}° {'vers le BAS' if pitch >= 0 else 'vers le HAUT'} par rapport à l'horizontale")
print(f"Roulis : {roll:+.1f}° ({'image horizontale' if abs(roll) < 2 else 'image penchée, à corriger'})")

if args.h and args.d:
    target = math.degrees(math.atan2(args.h, args.d))
    near = args.h / math.tan(math.radians(min(target + VFOV / 2, 89.9)))
    far = args.h / math.tan(math.radians(target - VFOV / 2)) if target > VFOV / 2 else float("inf")
    print(f"\nAngle visé pour h={args.h:g} cm, d={args.d:g} cm : {target:.1f}° vers le bas")
    print(f"Correction à apporter : incliner de {abs(target - pitch):.1f}° de plus vers le {'bas' if target > pitch else 'haut'}")
    print(f"Avec cet angle, la table est vue de {near:.0f} cm à {'l infini (horizon visible)' if far == float('inf') else f'{far:.0f} cm'} devant la caméra")