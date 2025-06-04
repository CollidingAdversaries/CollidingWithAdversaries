import os
import shutil
from huggingface_hub import list_repo_files, hf_hub_download, login

# Parameters
repo_id = "TSaala/CollidingAdversaries"
repo_type = "dataset"
destination_dir = "wanted/path/for/data"    # TODO: Adjust the destination directory for your needs!

# Ensure destination exists
os.makedirs(destination_dir, exist_ok=True)

files = list_repo_files(repo_id=repo_id, repo_type=repo_type)
feather_files = [f for f in files if f.endswith(".feather")]

# Download each .feather file and save to destination
for filename in feather_files:
    print(f"Downloading {filename}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type
    )

    dest_path = os.path.join(destination_dir, os.path.basename(filename))
    shutil.copy(downloaded_path, dest_path)
    print(f"Saved to: {dest_path}")

print("All feather files downloaded.")
