#!/usr/bin/env python3
"""Client d'inférence asynchrone officiel de LeRobot, avec la caméra ZMQ (camera1 = OAK-D) enregistrée.

En 0.5.1, lerobot.async_inference.robot_client n'importe que les configs caméra OpenCV et RealSense :
`type: zmq` y est refusé. Même contournement que scripts/robot/robot_client_zmq.py du repo du challenge.
"""
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401  (enregistre type: zmq)
from lerobot.async_inference.robot_client import async_client, register_third_party_plugins

if __name__ == "__main__":
    import _thread
    import os
    import threading

    # Arrêt au bout de ROLLOUT_DURATION_S secondes, comme un Ctrl+C : le client officiel déconnecte le bras proprement.
    duration = float(os.environ.get("ROLLOUT_DURATION_S", "0") or 0)
    if duration > 0:
        threading.Timer(duration, _thread.interrupt_main).start()
    register_third_party_plugins()
    try:
        async_client()
    except KeyboardInterrupt:
        pass