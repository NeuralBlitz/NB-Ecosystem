#!/usr/bin/env python3
"""
NeuralBlitz CLI Tool
Command-line interface for partners
"""

import typer
from typing import Optional
import requests
import json

app = typer.Typer(
    name="nb-router", help="NeuralBlitz Omnibus Router CLI", add_completion=False
)

# Global options
api_key_option = typer.Option(
    "--api-key", "-k", help="API key for authentication", envvar="NEURALBLITZ_API_KEY"
)
base_url_option = typer.Option(
    "--url",
    "-u",
    help="Base URL for API",
    envvar="NEURALBLITZ_URL",
    default="http://localhost:8000",
)


@app.command()
def query(input_data: str, api_key: str = api_key_option, url: str = base_url_option):
    """Process data through NeuralBlitz core."""
    typer.echo(f"Processing: {input_data}")

    response = requests.post(
        f"{url}/api/v1/core/process",
        headers={"X-API-Key": api_key},
        json={"input_data": [float(x) for x in input_data.split(",")]},
    )

    if response.status_code == 200:
        typer.echo(json.dumps(response.json(), indent=2))
    else:
        typer.echo(f"Error: {response.status_code}")
        typer.echo(response.text)


@app.command()
def agent(
    action: str,
    agent_type: str = typer.Option("--type", "-t", default="general"),
    task: Optional[str] = typer.Option("--task", None),
    api_key: str = api_key_option,
    url: str = base_url_option,
):
    """Manage LRS agents."""
    if action == "list":
        response = requests.get(
            f"{url}/api/v1/agent/list", headers={"X-API-Key": api_key}
        )
        typer.echo(json.dumps(response.json(), indent=2))
    elif action == "run" and task:
        response = requests.post(
            f"{url}/api/v1/agent/run",
            headers={"X-API-Key": api_key},
            json={"agent_type": agent_type, "task": task},
        )
        typer.echo(json.dumps(response.json(), indent=2))
    else:
        typer.echo("Usage: nb-router agent [list|run] --task 'task description'")


@app.command()
def quantum(
    qubits: int = typer.Option(4, "--qubits", "-q"),
    circuit: int = typer.Option(3, "--circuit", "-c"),
    api_key: str = api_key_option,
    url: str = base_url_option,
):
    """Run quantum simulation."""
    response = requests.post(
        f"{url}/api/v1/quantum/simulate",
        headers={"X-API-Key": api_key},
        json={"qubits": qubits, "circuit_depth": circuit},
    )

    if response.status_code == 200:
        typer.echo(json.dumps(response.json(), indent=2))
    else:
        typer.echo(f"Error: {response.status_code}")


@app.command()
def status(api_key: str = api_key_option, url: str = base_url_option):
    """Check system status."""
    response = requests.get(f"{url}/health", headers={"X-API-Key": api_key})
    typer.echo(json.dumps(response.json(), indent=2))


@app.command()
def capabilities(api_key: str = api_key_option, url: str = base_url_option):
    """List all capabilities."""
    response = requests.get(
        f"{url}/api/v1/capabilities", headers={"X-API-Key": api_key}
    )
    typer.echo(json.dumps(response.json(), indent=2))


@app.command()
def evolve(
    realities: int = typer.Option(4, "--realities", "-r"),
    cycles: int = typer.Option(50, "--cycles", "-n"),
    api_key: str = api_key_option,
    url: str = base_url_option,
):
    """Evolve multi-reality network."""
    response = requests.post(
        f"{url}/api/v1/core/evolve",
        headers={"X-API-Key": api_key},
        json={"num_realities": realities, "nodes_per_reality": 50, "cycles": cycles},
    )

    if response.status_code == 200:
        typer.echo(json.dumps(response.json(), indent=2))
    else:
        typer.echo(f"Error: {response.status_code}")


if __name__ == "__main__":
    app()
