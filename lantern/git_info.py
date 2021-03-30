import os
import subprocess


def git_info():
    git_directory = os.environ.get("GUILD_SOURCECODE", ".")
    return dict(
        git_branch=(
            subprocess.run(
                f"cd {git_directory}; git rev-parse --abbrev-ref HEAD",
                shell=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        ),
        git_commit=(
            subprocess.run(
                f"cd {git_directory}; git rev-parse HEAD",
                shell=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        ),
        git_changes=(
            subprocess.run(
                f"cd {git_directory}; git status --porcelain",
                shell=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
            != ""
        ),
    )
