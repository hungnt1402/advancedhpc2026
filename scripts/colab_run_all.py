"""Run the lab notebooks on a Colab GPU and publish each completed lab."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


def git(repo, *args, env=None):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env
    )
    if result.returncode:
        # Do not expose credential helper output or credentials in exceptions.
        raise RuntimeError(f"Git {args[0]} failed (exit {result.returncode}).")
    return result.stdout.strip()


def publish(repo, destination, token):
    """Use a temporary askpass helper; never put the token in a URL or file."""
    with tempfile.TemporaryDirectory(prefix="colab-auth-") as auth_dir:
        helper = Path(auth_dir) / "askpass.py"
        helper.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "print('x-access-token' if 'username' in sys.argv[1].lower() "
            "else os.environ['COLAB_PUSH_TOKEN'])\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
        env = dict(os.environ, GIT_ASKPASS=str(helper),
                   GIT_TERMINAL_PROMPT="0", COLAB_PUSH_TOKEN=token)
        git(repo, "add", "--", str(destination.relative_to(repo)))
        if not git(repo, "diff", "--cached", "--name-only"):
            return
        git(repo, "-c", "user.name=Colab Lab Runner", "-c",
            "user.email=colab-runner@users.noreply.github.com", "commit", "-m",
            f"Save Colab results: {destination.name}")
        for attempt in range(3):
            git(repo, "-c", "credential.helper=", "fetch", "origin", "main", env=env)
            git(repo, "rebase", "origin/main")
            try:
                git(repo, "-c", "credential.helper=", "push", "origin", "HEAD:main", env=env)
                return
            except RuntimeError:
                if attempt == 2:
                    raise


def execute_lab(source, work, compile_pdf):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(source, as_version=4)
    lab = source.parent.name
    output = work / "outputs" / lab
    output.mkdir(parents=True)
    # Record the actual GPU and input alongside the measurements.
    notebook.cells.append(nbformat.v4.new_code_cell(
        "import json as _json, platform as _platform\n"
        "from pathlib import Path as _Path\n"
        "from numba import cuda as _cuda\n"
        f"_out = _Path('outputs/{lab}')\n"
        "_out.mkdir(parents=True, exist_ok=True)\n"
        "_device = _cuda.get_current_device()\n"
        "_name = _device.name\n"
        "if isinstance(_name, bytes): _name = _name.decode()\n"
        "(_out / 'environment.json').write_text(_json.dumps({\n"
        "    'gpu': str(_name), 'python': _platform.python_version(),\n"
        "    'compute_capability': list(_device.compute_capability)\n"
        "}, indent=2), encoding='utf-8')\n"
        "if 'image' in globals():\n"
        "    from PIL import Image as _Image\n"
        "    _Image.fromarray(image).save(_out / 'input.png')\n"
        "if 'second' in globals():\n"
        "    _Image.fromarray(second).save(_out / 'input2.png')\n"
    ))
    try:
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(work)}}).execute()
    finally:
        # A failed notebook remains downloadable locally, but is not published.
        nbformat.write(notebook, work / "executed.ipynb")

    logs = []
    for cell in notebook.cells:
        for item in cell.get("outputs", []):
            if item.output_type == "stream":
                logs.append(item.text)
            elif item.output_type == "execute_result":
                logs.append(item.get("data", {}).get("text/plain", ""))
    (output / "run.log").write_text("\n".join(logs), encoding="utf-8")
    if lab != "Lab2" and not (output / "metrics.csv").is_file():
        raise RuntimeError(f"{lab} did not produce metrics.csv")
    if compile_pdf:
        for report in output.glob("Report.*.tex"):
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", report.name],
                    cwd=output, check=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    return output


def run_all(repo, token, labs=range(2, 11), compile_pdf=True, input_dir="/content"):
    repo = Path(repo).resolve()
    labs = list(labs)
    if not labs or any(n not in range(2, 11) for n in labs) or len(set(labs)) != len(labs):
        raise ValueError("Choose distinct lab numbers from 2 to 10.")
    if not token:
        raise ValueError("Add GITHUB_TOKEN to Colab Secrets and enable notebook access.")
    if compile_pdf and not shutil.which("pdflatex"):
        raise RuntimeError("Install LaTeX first, or set COMPILE_PDF=False.")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("The checkout must be clean before running labs.")
    revision = git(repo, "rev-parse", "HEAD")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = repo / "results" / run_id
    summary = {"run_id": run_id, "source_commit": revision, "labs": {}}
    destination.mkdir(parents=True)
    work_root = Path(tempfile.mkdtemp(prefix="colab-labs-"))
    print(f"Results: {destination}\nLocal diagnostics: {work_root}")
    # Snapshot sources so a concurrent GitHub update cannot alter this run.
    sources = {}
    for number in labs:
        source = repo / f"Lab{number}" / f"HPC_Lab{number}.ipynb"
        snapshot = work_root / f"Lab{number}" / source.name
        snapshot.parent.mkdir()
        shutil.copy2(source, snapshot)
        sources[number] = snapshot
    for number in labs:
        lab = f"Lab{number}"
        source = sources[number]
        work = source.parent
        for filename in ("image.jpg", "image2.jpg"):
            image_path = Path(input_dir) / filename
            if image_path.is_file():
                shutil.copy2(image_path, work / filename)
        print(f"Running {lab} ...", flush=True)
        try:
            output = execute_lab(source, work, compile_pdf)
            target = destination / lab
            target.mkdir()
            for artifact in output.iterdir():
                if artifact.suffix.lower() in {".png", ".jpg", ".csv", ".tex", ".pdf", ".npy", ".json"} or artifact.name == "run.log":
                    if artifact.stat().st_size > 90 * 1024 * 1024:
                        raise RuntimeError(f"Artifact too large for GitHub: {artifact.name}")
                    shutil.copy2(artifact, target / artifact.name)
            executed = work / "executed.ipynb"
            if executed.stat().st_size > 90 * 1024 * 1024:
                raise RuntimeError("Executed notebook too large for GitHub")
            shutil.copy2(executed, target / "executed.ipynb")
            summary["labs"][lab] = {"status": "success"}
        except Exception as error:
            # No partial artifacts are published as a successful lab.
            target = destination / lab
            if target.exists():
                shutil.rmtree(target)
            summary["labs"][lab] = {"status": "failed", "error": str(error)}
            print(f"{lab} failed; see {work / 'executed.ipynb'}", flush=True)
        (destination / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Publish after each lab so a later runtime interruption loses less work.
        publish(repo, destination, token)
        print(f"Saved {lab}: {summary['labs'][lab]['status']}", flush=True)
    print(f"https://github.com/hungnt1402/advancedhpc2026/tree/main/results/{run_id}")
    failures = [lab for lab, result in summary["labs"].items() if result["status"] != "success"]
    if failures:
        raise RuntimeError("Failed labs: " + ", ".join(failures) + ". Successful labs are saved.")
    return summary
