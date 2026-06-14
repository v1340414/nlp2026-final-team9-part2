#!/usr/bin/env python3
"""Google Drive에서 P3 checkpoint를 다운로드한다.

실행 전 checkpoint_config.json의 google_drive_file_id를 수정해야 한다.
file id는 Google Drive 공유 URL에서 /d/ 뒤에 오는 값이다.
https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests


GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    file_id = payload.get("google_drive_file_id", "").strip()
    if not file_id or file_id == "PASTE_GDRIVE_FILE_ID_HERE":
        raise SystemExit(
            "먼저 checkpoint_config.json의 google_drive_file_id를 설정하세요. "
            "https://drive.google.com/file/d/<FILE_ID>/view 형식의 URL에서 FILE_ID를 사용하면 됩니다."
        )
    return payload


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    return match.group(1) if match else None


def save_response_content(response, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    with temp_path.open("wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    temp_path.replace(destination)


def download_file(file_id, destination):
    session = requests.Session()
    response = session.get(
        GOOGLE_DRIVE_DOWNLOAD_URL,
        params={"id": file_id},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    token = get_confirm_token(response)
    if token:
        response = session.get(
            GOOGLE_DRIVE_DOWNLOAD_URL,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise RuntimeError(
            "Google Drive가 checkpoint 파일 대신 HTML 페이지를 반환했습니다. "
            "파일 공유 설정이 '링크가 있는 모든 사용자 보기 가능'인지 확인하세요."
        )

    save_response_content(response, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="checkpoint_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(config.get("checkpoint_path", "checkpoints/p3_selected_checkpoint.pt"))
    expected_min_size_mb = float(config.get("expected_min_size_mb", 0))

    if output_path.exists() and not args.force:
        print(f"checkpoint가 이미 존재합니다: {output_path}")
        return

    print(f"checkpoint를 다운로드합니다: {output_path}")
    download_file(config["google_drive_file_id"], output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"checkpoint 다운로드 완료: {output_path} ({size_mb:.2f} MB)")
    if expected_min_size_mb and size_mb < expected_min_size_mb:
        print(
            f"경고: checkpoint 크기가 expected_min_size_mb={expected_min_size_mb}보다 작습니다.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
