#!/usr/bin/env python3
"""Aperçu en direct de camera2 (poignet) pour régler la mise au point et l'orientation. Touche q ou fermeture de la fenêtre pour quitter."""
import os
import sys

import cv2

DEV = sys.argv[1]
W, H, FPS = int(os.environ.get("CAM_WIDTH", 640)), int(os.environ.get("CAM_HEIGHT", 480)), int(os.environ.get("CAM_FPS", 30))
WIN = "camera2 (poignet) - q pour quitter"

cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)
if not cap.isOpened():
    sys.exit(f"Impossible d'ouvrir {DEV}")
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, W * 2, H * 2)
while True:
    ok, frame = cap.read()
    if ok:
        cv2.imshow(WIN, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:  # fenêtre fermée à la souris : selon le backend, 0 ou une exception
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) == 0:
            break
    except cv2.error:
        break
cap.release()
cv2.destroyAllWindows()