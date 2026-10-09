#!/usr/bin/env python3
"""Copie locale d'un modèle du Hub, rendue lisible par le LeRobot du PC (0.5.1).

Un modèle entraîné avec un LeRobot plus récent (serveur GPU) peut contenir dans config.json des champs
que la 0.5.1 ne connaît pas (ex. pretrained_revision) : lerobot-record refuse alors de le charger.
Ce script télécharge le modèle dans outputs/policies/<nom>/, retire de config.json les champs inconnus
de la classe de config installée, puis vérifie que config, poids et pré/post-traitements se chargent.
Le code de LeRobot n'est pas modifié.

  python3 scripts/policy_local.py dbal67/Robotkraft_TRI_v2
"""
import dataclasses
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

repo = sys.argv[1]
dst = Path("outputs/policies") / repo.split("/")[-1]
src = Path(snapshot_download(repo, repo_type="model"))
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)  # copie réelle (le cache du Hub contient des liens)

cfg_path = dst / "config.json"
cfg = json.loads(cfg_path.read_text())
cls = PreTrainedConfig.get_choice_class(cfg["type"])
known = {f.name for f in dataclasses.fields(cls)} | {"type"}
dropped = sorted(k for k in cfg if k not in known)
for k in dropped:
    cfg.pop(k)
cfg_path.write_text(json.dumps(cfg, indent=4))
print(f"{repo} -> {dst}")
print("champs retirés de config.json :", ", ".join(dropped) or "aucun")

# Vérification : chargement complet comme le fera lerobot-record.
config = PreTrainedConfig.from_pretrained(dst)
policy = get_policy_class(config.type).from_pretrained(dst, config=config)
print(f"OK : {config.type} chargé ({sum(p.numel() for p in policy.parameters()) / 1e6:.0f} M paramètres)")
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402

for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    if (dst / name).exists():
        PolicyProcessorPipeline.from_pretrained(dst, config_filename=name)
        print(f"OK : {name}")
print(f"\nÀ utiliser : POLICY_PATH={dst}")