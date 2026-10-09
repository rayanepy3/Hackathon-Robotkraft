# ETIQ — reconnaissance d'étiquettes et saisie de l'objet correspondant

Trouver l'objet dont l'étiquette porte le mot donné, puis le saisir. Lecture par OCR (RapidOCR, CPU), du plus simple au plus dur :
1. l'OAK lit les étiquettes à distance ;
2. le bras se lève, la caméra poignet lit le rack de haut ;
3. le bras saisit chaque tube, le tourne devant l'OAK, le repose si ce n'est pas le bon.

| Commande | Rôle |
|---|---|
| `make etiq-saisir PLACE=k N=5`, `etiq-regarder`, `etiq-montrer`, `etiq-reposer PLACE=k` | Démonstrations des sous-gestes (3 datasets, 3 modèles) |
| `make etiquette-calib` / `make etiquette-voir` | Calibration (une fois) / lecture en direct |
| `make etiquette MOT=IODURE` | Épreuve complète (`OU=rack` ou `table`, `DRY=1`) |

Non géré pour l'instant : la variante « cube rouge à retourner ».