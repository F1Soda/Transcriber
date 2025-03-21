import json
import subprocess
from typing import Tuple

from pytubefix import YouTube


def po_token_verifier() -> Tuple[str, str]:
    token_object = generate_youtube_token()
    return token_object["visitorData"], token_object["poToken"]


def cmd(command, check=True, shell=True, capture_output=True, text=True):
    """
    Runs a command in a shell, and throws an exception if the return code is non-zero.
    :param command: any shell command.
    :return:
    """
    print(f" + {command}")
    try:
        return subprocess.run(command, check=check, shell=shell, capture_output=capture_output, text=text)
    except subprocess.CalledProcessError as error:
        raise Exception(f"\"{command}\" return exit code: {error.returncode}")


def generate_youtube_token(kek: None) -> Tuple[str, str]:
    print("Generating YouTube token")
    result = cmd("youtube-po-token-generator")
    data = json.loads(result.stdout)
    print(f"Result: {data}")
    return data['visitorData'], data['poToken']


url = "https://www.youtube.com/watch?v=_X5-3_d4Z-g"

yt = YouTube(url, use_po_token=True, po_token_verifier=generate_youtube_token)
print(yt.title)
