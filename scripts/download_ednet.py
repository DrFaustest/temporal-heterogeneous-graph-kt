from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

EDNET_KT1_URL = "https://bit.ly/ednet_kt1"
EDNET_CONTENT_URL = "https://bit.ly/ednet-content"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl_path = shutil.which('curl.exe') or shutil.which('curl')
    if curl_path is None:
        raise RuntimeError('curl.exe is required to download EdNet archives in this environment.')
    command = [curl_path, '-L', '--fail', '--output', str(destination), url]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f'curl download failed for {url} with exit code {completed.returncode}')


def _extract(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description='Download EdNet KT1 and contents archives.')
    parser.add_argument('--output-dir', default='data/raw/ednet', help='Destination directory for EdNet files.')
    parser.add_argument('--skip-extract', action='store_true', help='Download archives only.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    kt1_zip = output_dir / 'ednet_kt1.zip'
    contents_zip = output_dir / 'ednet_contents.zip'
    print(f'Downloading KT1 to {kt1_zip}...', flush=True)
    _download(EDNET_KT1_URL, kt1_zip)
    print(f'Downloading contents to {contents_zip}...', flush=True)
    _download(EDNET_CONTENT_URL, contents_zip)

    if not args.skip_extract:
        print('Extracting KT1...', flush=True)
        _extract(kt1_zip, output_dir / 'KT1')
        print('Extracting contents...', flush=True)
        _extract(contents_zip, output_dir / 'contents')

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
