import os
import shutil

def prep_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Delete folder and all contents
    
    os.makedirs(path, exist_ok=True)  # Recreate empty folder