import subprocess
from pathlib import Path

def run_reconstruction(docker_image: str, input_ct: str | Path, recon_dir: str | Path, output_pet: str | Path) -> None:
    input_ct = Path(input_ct).absolute()
    recon_dir = Path(recon_dir).absolute()
    output_pet = Path(output_pet).absolute()
    output_pet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{input_ct}:/input_ct:ro",
        "-v", f"{recon_dir}:/recon:ro",
        "-v", f"{output_pet.parent}:/output",
        docker_image,
        "/input_ct",
        "/recon",
        f"/output/{output_pet.name}"
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("docker_image")
    parser.add_argument("input_ct")
    parser.add_argument("recon_dir")
    parser.add_argument("output_pet")

    args = parser.parse_args()
    run_reconstruction(args.docker_image, args.input_ct, args.recon_dir, args.output_pet)
