import os
import re
import signal
import socket
import subprocess
import sys
from codecs import iterdecode
from contextlib import closing
from subprocess import PIPE, Popen
from time import sleep


def sleep_infinity():
    """Sleeps forever."""
    while True:
        sleep(9999999)


def find_free_port():
    """Finds a random free port.

    Returns:
        int: The free port.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def kill_port(port):
    """Kills any processes running on the port.

    Args:
        port (int): The port to kill processes running on.
    """

    process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    for p in str(stdout.decode("utf-8")).split("\n")[1:]:
        data = [x for x in p.split(" ") if x != ""]
        if len(data) <= 1:
            continue

        os.kill(int(data[1]), signal.SIGKILL)


def run_ngrok(port, hostname=None):
    """Runs ngrok.

    Args:
        port (int): The port to tunnel.
        hostname (str): The hostname to run the tunnel on.

    Returns:
        str: The URL to access the tunnel.
    """
    if hostname:
        ngrok_command = (
            f""""../$PROJECT_DATA_BIN/ngrok" --domain={hostname} --log=stdout"""
            f""" http {port}"""
        )
    else:
        ngrok_command = f"""
            "../$PROJECT_DATA_BIN/ngrok" --log=stdout http {port}
        """
    process = Popen(ngrok_command, shell=True, env=os.environ, stdout=PIPE)
    if hostname:
        return f"https://{hostname}/"
    else:
        process_output = ""
        if process.stdout:
            for c in iterdecode(iter(lambda: process.stdout.read(1), b""), "utf8"):  # type: ignore[union-attr] # noqa: B950
                process_output += c
                urls = re.findall(r"(https?://\S+)", process_output)
                urls = [url for url in urls if "ngrok-free.app" in url]
                if len(urls) > 0:
                    return urls[0]


def run_cloudflared(port, hostname=None):
    """Runs cloudflared.

    Args:
        port (int): The port to tunnel.
        hostname (str): The hostname to run the tunnel on.

    Returns:
        str: The URL to access the tunnel.
    """
    if hostname:
        tunnel_name = os.environ["PROJECT_JOB_NAME"] + "_" + sys.argv[1]
        cloudflared_delete_command = (
            """"../$PROJECT_DATA_BIN/cloudflared" tunnel delete"""
            f""" --credentials-file ../$PROJECT_DATA_BIN/{tunnel_name}.json"""
            f""" {tunnel_name}"""
        )
        subprocess.run(
            cloudflared_delete_command,
            shell=True,
            env=os.environ,
            stdout=PIPE,
            stderr=PIPE,
        )
        cloudflared_create_command = (
            """"../$PROJECT_DATA_BIN/cloudflared" tunnel create"""
            f""" --credentials-file ../$PROJECT_DATA_BIN/{tunnel_name}.json"""
            f""" {tunnel_name}"""
        )
        subprocess.run(
            cloudflared_create_command,
            shell=True,
            env=os.environ,
            stdout=PIPE,
            stderr=PIPE,
        )
        cloudflared_dns_command = (
            f""""../$PROJECT_DATA_BIN/cloudflared" tunnel route dns -f {tunnel_name}"""
            f""" {hostname}"""
        )
        subprocess.run(
            cloudflared_dns_command,
            shell=True,
            env=os.environ,
            stdout=PIPE,
            stderr=PIPE,
        )
        cloudflared_command = (
            f""""../$PROJECT_DATA_BIN/cloudflared" tunnel run"""
            f""" --url http://localhost:{port}"""
            f""" --credentials-file ../$PROJECT_DATA_BIN/{tunnel_name}.json"""
            f""" {tunnel_name} 2>&1"""
        )
    else:
        cloudflared_command = (
            f""""../$PROJECT_DATA_BIN/cloudflared" tunnel"""
            f""" --url http://localhost:{port} 2>&1"""
        )
    process = Popen(cloudflared_command, shell=True, env=os.environ, stdout=PIPE)
    if hostname:
        return f"https://{hostname}/"
    else:
        process_output = ""
        if process.stdout:
            for c in iterdecode(iter(lambda: process.stdout.read(1), b""), "utf8"):  # type: ignore[union-attr] # noqa: B950
                process_output += c
                urls = re.findall(r"(https?://\S+)", process_output)
                urls = [url for url in urls if (hostname or "trycloudflare.com") in url]
                if len(urls) > 0:
                    return urls[0]


def run_jupyter(port=None, password=None):
    """Runs Jupyter Lab.

    Args:
        port (int): The port to run Jupyter Lab on. If None, a random port is used.
        password (str): The password to protect the tunnel with. If None, no password.

    Returns:
        int: The port Jupyter Lab is running on.
    """
    if port is None:
        port = find_free_port()
    if password is None:
        password = ""
    kill_port(port)
    jupyter_command = (
        f"""jupyter lab --collaborative --ip=0.0.0.0 --port={port}"""
        f""" --IdentityProvider.token='{password}' --ContentsManager.allow_hidden=True &"""
    )
    environ = {
        k: v
        for k, v in os.environ.items()
        if "SLURM" not in k.upper() and not k.startswith("PROJECT_")
    }
    environ["PYTHONPATH"] = os.path.abspath(os.getcwd())
    subprocess.run(jupyter_command, shell=True, check=True, env=environ, cwd="../")
    return port


def run_http_server(port=None):
    """Runs a HTTP server.

    Args:
        port (int): The port to run a HTTP server on. If None, a random port is used.

    Returns:
        int: The port a HTTP server is running on.
    """
    if port is None:
        port = find_free_port()
    kill_port(port)
    http_server_command = f"""
        python3 -m http.server {port} &
    """
    subprocess.run(
        http_server_command, shell=True, check=True, env=os.environ, cwd="../"
    )
    return port
