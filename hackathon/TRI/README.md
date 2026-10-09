# TRI — tri de liquides par couleur

Ranger les tubes posés sur la table dans le porte-tubes, groupés par couleur (B bleu/cyan, J jaune, R rouge/magenta), dans l'ordre demandé de gauche à droite vu du robot.

Stratégie : couleur par couleur, toujours dans la place libre la plus à gauche. Le modèle (SmolVLA) ne connaît que 3 phrases, une par couleur ; `tri.py` enchaîne, vérifie chaque tube à la caméra et recommence si raté.

| Commande | Rôle |
|---|---|
| `make tri-b N=5` / `tri-j` / `tri-r` | Enregistrer des démonstrations (3 couleurs sur la table) |
| `make tri-calib` | Cliquer les 6 places et la zone de la table (une fois) → `tri_calib.json` |
| `make tri-voir` | Vision en direct |
| `make tri ORDER=BJR` | Épreuve complète (`DRY=1` : sans bouger le bras, `ASYNC_SERVER=ip:8080` : modèle sur GPU) |