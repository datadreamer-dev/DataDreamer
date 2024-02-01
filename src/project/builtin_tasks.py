import click
from loguru import logger

from .serve import (
    run_cloudflared,
    run_http_server,
    run_jupyter,
    run_ngrok,
    sleep_infinity,
)


@click.option(
    "--tunnel",
    "-t",
    default="cloudflare",
    type=click.Choice(["cloudflare", "ngrok"]),
    help="The tunneling service to use.",
)
@click.option(
    "--hostname", "-h", default=None, type=str, help="The hostname to serve at."
)
@click.option("--password", "-p", default=None, type=str, help="The password to use.")
def jupyter(ctx, tunnel, hostname, password):
    """This command runs Jupyter Lab."""
    logger.info("Running Jupyter Lab...")
    port = run_jupyter(password=password)
    if tunnel == "cloudflare":
        url = run_cloudflared(port, hostname=hostname)
    else:
        url = run_ngrok(port, hostname=hostname)
    logger.info(f"Jupyter Lab is available at URL: {url}")
    sleep_infinity()


@click.option(
    "--tunnel",
    "-t",
    default="cloudflare",
    type=click.Choice(["cloudflare", "ngrok"]),
    help="The tunneling service to use.",
)
@click.option(
    "--hostname", "-h", default=None, type=str, help="The hostname to serve at."
)
def http_server(ctx, tunnel, hostname):
    """This command runs a HTTP server."""
    logger.info("Running HTTP server...")
    port = run_http_server()
    if tunnel == "cloudflare":
        url = run_cloudflared(port, hostname=hostname)
    else:
        url = run_ngrok(port, hostname=hostname)
    logger.info(f"HTTP server is available at URL: {url}")
    sleep_infinity()


def register_builtin_tasks(_main):
    _main.command(hidden=True)(click.pass_context(jupyter))
    _main.command(hidden=True)(click.pass_context(http_server))
