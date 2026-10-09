#!/usr/bin/env python3
"""Pont OAK-D Lite -> ZMQ pour la caméra LeRobot `type: zmq` (camera1 = vue fixe, RGB seul).

Repris de scripts/camera/oak_zmq_server.py du repo du challenge, avec la résolution
paramétrable : la ZMQCamera de LeRobot 0.5.1 ne redimensionne pas, et l'écriture du
dataset rejette toute image dont la taille diffère de la config caméra.
Variables d'environnement : CAM_WIDTH, CAM_HEIGHT, CAM_FPS, CAM1_ZMQ_PORT, CAM1_ZMQ_NAME.
"""
import base64
import json
import os
import time

import cv2
import depthai as dai
import zmq

WIDTH = int(os.environ.get("CAM_WIDTH", 640))
HEIGHT = int(os.environ.get("CAM_HEIGHT", 480))
FPS = int(os.environ.get("CAM_FPS", 30))
PORT = int(os.environ.get("CAM1_ZMQ_PORT", 5555))
NAME = os.environ.get("CAM1_ZMQ_NAME", "top")
READY_FILE = os.environ.get("OAK_READY_FILE", "/tmp/oak_zmq_ready")

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.SNDHWM, 20)  # borne la file d'envoi
socket.setsockopt(zmq.LINGER, 0)  # close() n'attend pas les messages en attente
socket.bind(f"tcp://127.0.0.1:{PORT}")  # local seulement (réseau de l'hôte partagé)

RETRIES = int(os.environ.get("OAK_RETRIES", 4))  # essais d'ouverture ; OAK_PERSIST=1 : se relance indéfiniment


def serve():
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        # Capteur 4:3 de l'OAK-D Lite : 640x480 garde tout le champ ; un autre ratio est recadré.
        queue = cam.requestOutput((WIDTH, HEIGHT), fps=FPS).createOutputQueue()
        pipeline.start()
        for _ in range(10):  # laisse l'auto-exposition se stabiliser
            queue.get()
        open(READY_FILE, "w").close()
        print(f"OAK-D -> ZMQ tcp://127.0.0.1:{PORT} camera_name={NAME} {WIDTH}x{HEIGHT}@{FPS}", flush=True)
        while pipeline.isRunning():
            t0 = time.time()
            bgr = queue.get().getCvFrame()
            # ZMQCamera décode le JPEG sans conversion de couleurs : on encode des octets RGB
            # pour que lerobot reçoive bien du RGB.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            msg = {"timestamps": {NAME: time.time()}, "images": {NAME: base64.b64encode(buf).decode()}}
            try:
                socket.send_string(json.dumps(msg), zmq.NOBLOCK)
            except zmq.Again:  # aucun abonné prêt : l'image est abandonnée
                pass
            sleep = 1.0 / FPS - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


# L'OAK plante parfois (lien USB 2) : on attend qu'elle redémarre sur l'USB et on rouvre.
failures = 0
try:
    while True:
        try:
            serve()
            failures = 0
        except RuntimeError as e:
            failures += 1
            persist = os.environ.get("OAK_PERSIST") == "1"
            print(f"OAK-D : {e} (échec {failures}{'' if persist else f'/{RETRIES}'}), nouvel essai dans 6 s", flush=True)
            if os.path.exists(READY_FILE):
                os.remove(READY_FILE)
            if not persist and failures >= RETRIES:
                raise
            time.sleep(6)
except KeyboardInterrupt:
    pass
finally:
    socket.close()
    context.term()
    if os.path.exists(READY_FILE):
        os.remove(READY_FILE)