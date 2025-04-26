import subprocess

def run_child_process(_CMD: str, _stdin: str) -> str:
    process = subprocess.Popen(
        [_CMD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = process.communicate(_stdin)

    if process.returncode != 0:
        raise Exception(f"Child process failed!\nSTDERR:\n{stderr}")

    return stdout