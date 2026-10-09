#!/usr/bin/env python3
"""Aperçu en direct de camera1 (OAK-D Lite, RGB) pour régler le cadrage. Touche q ou fermeture de la fenêtre pour quitter."""
import os

import cv2
import depthai as dai

W, H, FPS = int(os.environ.get("CAM_WIDTH", 640)), int(os.environ.get("CAM_HEIGHT", 480)), int(os.environ.get("CAM_FPS", 30))
WIN = "camera1 (OAK-D) - q pour quitter"

with dai.Pipeline() as pipeline:
    queue = pipeline.create(dai.node.Camera).build().requestOutput((W, H), fps=FPS).createOutputQueue()
    pipeline.start()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W * 2, H * 2)
    while pipeline.isRunning():
        frame = queue.get().getCvFrame()
        cv2.imshow(WIN, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        try:  # fenêtre fermée à la souris : selon le backend, 0 ou une exception
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) == 0:
                break
        except cv2.error:
            break
cv2.destroyAllWindows()