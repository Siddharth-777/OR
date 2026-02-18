#IMPORTS
import os
from pathlib import Path

#LOAD ENV FUNCTION
def load_env_file() -> None:

    #PATH DECLARATION
    env_path = Path(".env")

    #PATH NOT EXIST
    if not env_path.exists():
        return

    #READING ENV
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        #SPLIT LINES
        line = raw_line.strip()

        #EDGE CASES
        if not line or line.startswith("#") or "=" not in line:
            continue

        #SPLIT VARIABLES
        k, v = line.split("=", 1)

        k = k.strip()
        v = v.strip().strip('"').strip("'")

        if k and k not in os.environ:
            os.environ[k] = v

#FUNCTION CALL
load_env_file()

SUPABASE_URL = os.getenv("SUPABASE_URL","")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY","")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET","")

