import subprocess
import sys
import threading
import time
from pathlib import Path
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DATA_DIR = Path("data")
RUNS_DIR = Path("runs")


def _check(path: Path) -> bool:
    return path.exists()


def _latest_run(prefix: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    matches = sorted(RUNS_DIR.glob(f"{prefix}_*"), reverse=True)
    return matches[0] if matches else None


def _status_table() -> Table:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("step", style="bold")
    table.add_column("status")
    table.add_column("detail", style="dim")

    cleaned = DATA_DIR / "processed" / "cleaned.csv"
    partitions = list((DATA_DIR / "processed" / "partitions").glob("client_*.npz"))
    central = _latest_run("central")
    fl = _latest_run("fl")

    table.add_row(
        "1. Preprocess",
        "[green]done[/green]" if _check(cleaned) else "[yellow]pending[/yellow]",
        str(cleaned) if _check(cleaned) else "",
    )
    table.add_row(
        "2. Partition",
        "[green]done[/green]" if partitions else "[yellow]pending[/yellow]",
        f"{len(partitions)} client partition(s)" if partitions else "",
    )
    table.add_row(
        "3. Central train",
        "[green]done[/green]" if central else "[yellow]pending[/yellow]",
        str(central) if central else "",
    )
    table.add_row(
        "4. Federated train",
        "[green]done[/green]" if fl else "[yellow]pending[/yellow]",
        str(fl) if fl else "",
    )
    return table


def _prompt(label: str, default: str) -> str:
    console.print(f"  [cyan]{label}[/cyan] [dim](default: {default})[/dim]", end=" ")
    value = input().strip()
    return value if value else default


def _run(cmd: list[str]) -> None:
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print(f"\n[red]Command exited with code {result.returncode}[/red]")
    else:
        console.print("\n[green]Done.[/green]")


_CICIDS_DEFAULT_PATHS = ",".join([
    "Data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    "Data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
])

_CICIDS_DROP_COLS = ",".join([
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp",
])


def step_preprocess() -> None:
    console.print("\n[bold]Preprocess (CICIDS2017)[/bold]")
    data_path = _prompt("CSV paths (comma-separated):", _CICIDS_DEFAULT_PATHS)
    label_col = _prompt("Label column:", "Label")
    sep = _prompt("CSV separator:", ",")
    binary = _prompt("Binary classification (true/false):", "true")
    drop_cols = _prompt("Columns to drop (comma-separated):", _CICIDS_DROP_COLS)
    encode_cols = _prompt("String columns to encode as integers (leave blank for none):", "")
    cmd = [
        sys.executable, "-m", "app.main", "preprocess",
        "--data-path", data_path,
        "--label-col", label_col,
        "--sep", sep,
        "--binary", binary,
        "--data-dir", str(DATA_DIR),
    ]
    if drop_cols.strip():
        cmd += ["--drop-cols", drop_cols.strip()]
    if encode_cols.strip():
        cmd += ["--encode-cols", encode_cols.strip()]
    _run(cmd)


def step_partition() -> None:
    console.print("\n[bold]Partition[/bold]")
    label_col = _prompt("Label column:", "Label")
    clients = _prompt("Number of clients:", "2")
    test_size = _prompt("Test fraction:", "0.2")
    mode = _prompt("Partition mode (iid/non-iid):", "iid")
    extra = ["--iid"] if mode.lower() == "iid" else ["--non-iid"]
    _run([
        sys.executable, "-m", "app.main", "partition",
        "--data-dir", str(DATA_DIR),
        "--label-col", label_col,
        "--clients", clients,
        "--test-size", test_size,
        *extra,
    ])


def step_central() -> None:
    console.print("\n[bold]Central training[/bold]")
    epochs = _prompt("Epochs:", "5")
    batch_size = _prompt("Batch size:", "256")
    lr = _prompt("Learning rate:", "0.001")
    device = _prompt("Device (cpu/cuda):", "cpu")
    _run([
        sys.executable, "-m", "app.main", "central-train",
        "--data-dir", str(DATA_DIR),
        "--runs-dir", str(RUNS_DIR),
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--lr", lr,
        "--device", device,
    ])


def _docker_ps_panel() -> None:
    time.sleep(12)
    result = subprocess.run(
        ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        console.print("\n[bold cyan]Running containers right now:[/bold cyan]")
        for line in result.stdout.strip().splitlines():
            console.print(f"  {line}")
        console.print()


def step_federated() -> None:
    console.print("\n[bold]Federated training (Docker)[/bold]")
    clients = _prompt("Number of client containers:", "2")
    rounds = _prompt("Number of FL rounds:", "5")
    mode = _prompt("Partition mode (iid/non-iid):", "iid")
    iid_val = "true" if mode.lower() == "iid" else "false"
    cmd = [
        "docker", "compose",
        "-f", "docker/docker-compose.yml",
        "up", "--build",
        "--scale", f"client={clients}",
    ]
    console.print(f"\n[dim]Running: {' '.join(cmd)}[/dim]")
    console.print("[dim](a container list will print automatically once they are up)[/dim]\n")

    env = {
        **__import__("os").environ,
        "CLIENTS": clients,
        "ROUNDS": rounds,
        "IID": iid_val,
    }

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    t = threading.Thread(target=_docker_ps_panel, daemon=True)
    t.start()

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    if proc.returncode not in (0, 1):
        console.print(f"\n[red]Command exited with code {proc.returncode}[/red]")
    else:
        console.print("\n[green]Done.[/green]")


def reset_project() -> None:
    console.print("\n[bold red]Reset project[/bold red]")
    console.print("[yellow]This will delete all processed data and training results.[/yellow]")
    console.print("[yellow]Raw CSV files in Data/Raw/ will be preserved.[/yellow]")
    console.print("\n[yellow]Type 'reset' to confirm: [/yellow]", end="")
    
    confirm = input().strip().lower()
    if confirm != "reset":
        console.print("[red]Reset cancelled.[/red]")
        return
    
    removed_count = 0
    
    # Remove processed data directory
    processed_dir = DATA_DIR / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        removed_count += 1
        console.print(f"[green]Removed: {processed_dir}[/green]")
    
    # Remove all runs directories
    if RUNS_DIR.exists():
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
                removed_count += 1
        console.print(f"[green]Removed all runs in: {RUNS_DIR}[/green]")
    
    if removed_count == 0:
        console.print("[yellow]Nothing to remove.[/yellow]")
    else:
        console.print(f"[green]Reset complete. Removed {removed_count} items.[/green]")


def _find_metrics_json(run_dir: Path, prefer_final: bool = False) -> dict:
    # locate the primary metrics JSON in a run directory dynamically
    import json
    skip = {"preprocessing_report.json", "partition_report.json", "configs.json"}
    candidates = sorted(run_dir.glob("*metrics*.json"))
    candidates = [p for p in candidates if p.name not in skip]
    if prefer_final:
        finals = [p for p in candidates if "final" in p.name]
        if finals:
            return json.loads(finals[0].read_text())
        over_rounds = [p for p in candidates if "over_rounds" in p.name]
        candidates = [p for p in candidates if p not in over_rounds]
    candidates = [p for p in candidates if "over_rounds" not in p.name]
    if candidates:
        return json.loads(candidates[0].read_text())
    return {}


def show_results() -> None:
    central = _latest_run("central")
    fl = _latest_run("fl")
    if not central and not fl:
        console.print("[yellow]No results yet.[/yellow]")
        return

    table = Table(title="Latest metrics", box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    if central:
        table.add_column("Centralized", justify="right")
    if fl:
        table.add_column("Federated", justify="right")

    central_m: dict = _find_metrics_json(central) if central else {}
    fl_m: dict = _find_metrics_json(fl, prefer_final=True) if fl else {}

    for key in ["accuracy", "f1", "roc_auc", "pr_auc", "precision", "recall"]:
        row = [key]
        if central:
            row.append(f"{central_m.get(key, 'n/a'):.6f}" if isinstance(central_m.get(key), float) else "n/a")
        if fl:
            row.append(f"{fl_m.get(key, 'n/a'):.6f}" if isinstance(fl_m.get(key), float) else "n/a")
        table.add_row(*row)

    console.print(table)


def main() -> None:
    while True:
        console.clear()
        console.print(Panel("[bold]Federated IDS Pipeline[/bold]", expand=False))
        console.print(_status_table())
        console.print()
        console.print("  [bold cyan]1[/bold cyan]  Preprocess data")
        console.print("  [bold cyan]2[/bold cyan]  Partition data")
        console.print("  [bold cyan]3[/bold cyan]  Train centralized model")
        console.print("  [bold cyan]4[/bold cyan]  Run federated training (Docker)")
        console.print("  [bold cyan]5[/bold cyan]  Show latest results")
        console.print("  [bold red]6[/bold red]  Reset project (clear all processed data & results)")
        console.print("  [bold cyan]q[/bold cyan]  Quit")
        console.print()
        console.print("Choice: ", end="")

        choice = input().strip().lower()

        if choice == "1":
            step_preprocess()
        elif choice == "2":
            step_partition()
        elif choice == "3":
            step_central()
        elif choice == "4":
            step_federated()
        elif choice == "5":
            show_results()
        elif choice == "6":
            reset_project()
        elif choice in {"q", "quit", "exit"}:
            break
        else:
            console.print("[red]Unknown choice.[/red]")

        if choice in {"1", "2", "3", "4", "5", "6"}:
            console.print("\nPress Enter to continue...")
            input()


if __name__ == "__main__":
    main()
