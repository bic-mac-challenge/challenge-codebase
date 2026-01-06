import subprocess
from pathlib import Path


def run_submission(docker_image: str, input_dir: str | Path, output_path: str | Path) -> None:
    input_dir = Path(input_dir).absolute()
    output_path = Path(output_path).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{output_path.parent}:/output",
        docker_image,
        "/input",
        f"/output/{output_path.name}"
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("docker_image")
    parser.add_argument("input_dir")
    parser.add_argument("output_path")

    args = parser.parse_args()
    run_submission(args.docker_image, args.input_dir, args.output_path)
