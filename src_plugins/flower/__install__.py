# Installations
# ----------------------------------------
import sys

import src.installer
from src.classes import paths
from src.installer import gitclone

repo_dir = paths.plug_repos / "flower" / "SD_CN_Animation"

gitclone("https://github.com/volotat/SD-CN-Animation/", "c34cfe2976b0153894edca21ec1334685bbe968b")
gitclone("https://github.com/princeton-vl/RAFT", into_dir=repo_dir)

sys.path.append(repo_dir / "RAFT/core")
