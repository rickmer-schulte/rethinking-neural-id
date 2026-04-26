from __future__ import annotations

import argparse
import shutil
import string
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from rethinking_neural_id.paths import RepoPaths


WIKITEXT_URL = (
    "https://raw.githubusercontent.com/chengemily1/id-llm-abstraction/"
    "main/corpora/wikitext_sane_ds.txt"
)
IMAGE_SOURCE_URL = "https://ndownloader.figshare.com/files/15250958?private_link=8a039f58c7b84a215b6d"
IMAGE_SUBDIR_NAME = "imagenet_training_single_objs"
CHUNK_SIZE = 1024 * 1024


def _download_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print("The image archive is about 2.7 GB, so this can take a few minutes.")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, open(target, "wb") as handle:
        total_bytes = response.headers.get("Content-Length")
        total = int(total_bytes) if total_bytes is not None else None
        downloaded = 0

        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(
                    f"\rDownloaded {downloaded / 1_000_000:.1f} / "
                    f"{total / 1_000_000:.1f} MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloaded {downloaded / 1_000_000:.1f} MB",
                    end="",
                    flush=True,
                )
        print(file=sys.stderr)
    print(f"Saved download to {target}")
    return target


def _is_html_or_json_response(path: Path) -> bool:
    with open(path, "rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html", b"{"))


def _shard_suffix(index: int) -> str:
    letters = string.ascii_lowercase
    return letters[index // len(letters)] + letters[index % len(letters)]


def prepare_wikitext(
    *,
    output_dir: Path,
    url: str,
    lines_per_shard: int,
    max_shards: int,
    overwrite: bool,
) -> list[Path]:
    expected = [output_dir / f"shard_{_shard_suffix(index)}" for index in range(max_shards)]
    if all(path.exists() for path in expected) and not overwrite:
        print(f"Wikitext shards already exist in {output_dir}")
        return expected

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)

    print("Text source: Cheng et al. (2025), id-llm-abstraction wikitext corpus.")
    print(f"Downloading and sharding wikitext into {output_dir}")
    written: list[Path] = []
    current_handle = None
    try:
        with urlopen(url) as response:
            for line_number, raw_line in enumerate(response):
                shard_index = line_number // lines_per_shard
                if shard_index >= max_shards:
                    break

                line_in_shard = line_number % lines_per_shard
                if line_in_shard == 0:
                    if current_handle is not None:
                        current_handle.close()
                    shard_path = output_dir / f"shard_{_shard_suffix(shard_index)}"
                    current_handle = open(shard_path, "wb")
                    written.append(shard_path)

                current_handle.write(raw_line)
    finally:
        if current_handle is not None:
            current_handle.close()

    print(f"Wrote {len(written)} wikitext shard(s):")
    for path in written:
        print(f"  {path}")
    return written


def _find_image_subdir(extract_root: Path) -> Path:
    matches = [path for path in extract_root.rglob(IMAGE_SUBDIR_NAME) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{IMAGE_SUBDIR_NAME}' inside extracted Figshare archive."
        )
    return matches[0]


def _extract_image_source(source: Path, extract_root: Path) -> Path:
    if source.is_dir():
        if source.name == IMAGE_SUBDIR_NAME:
            return source
        return _find_image_subdir(source)

    if zipfile.is_zipfile(source):
        print(f"Extracting zip archive {source}")
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extract_root)
        return _find_image_subdir(extract_root)

    if tarfile.is_tarfile(source):
        print(f"Extracting tar archive {source}")
        with tarfile.open(source) as archive:
            archive.extractall(extract_root)
        return _find_image_subdir(extract_root)

    if _is_html_or_json_response(source):
        raise ValueError(
            "The Figshare URL returned a browser challenge or permissions response instead of "
            "the image archive. Download the file in a browser from "
            f"{IMAGE_SOURCE_URL}, then rerun with:\n"
            "  uv run python scripts/prepare_raw_data.py --kind images "
            "--image-source /path/to/downloaded/file_or_folder"
        )

    raise ValueError(
        f"Unsupported image source: {source}. Expected a zip archive, tar archive, "
        f"or a folder containing '{IMAGE_SUBDIR_NAME}'."
    )


def prepare_images(
    *,
    output_dir: Path,
    url: str,
    image_source: Path | None,
    overwrite: bool,
) -> Path:
    target_dir = output_dir / IMAGE_SUBDIR_NAME
    if target_dir.exists() and not overwrite:
        print(f"Image folder already exists at {target_dir}")
        return target_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)

    with tempfile.TemporaryDirectory(prefix="rni_figshare_") as tmp_name:
        tmp_root = Path(tmp_name)
        extract_root = tmp_root / "extracted"
        extract_root.mkdir()

        if image_source is None:
            print(
                "Image source: Ansuini et al. (2019) ImageNet object images, "
                "distributed via Figshare."
            )
            source = _download_file(url, tmp_root / "image_data")
        else:
            source = image_source.expanduser().resolve()

        source_dir = _extract_image_source(source, extract_root)
        shutil.copytree(source_dir, target_dir)

    print(f"Prepared image data at {target_dir}")
    return target_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare raw data for reproduction.")
    parser.add_argument("--kind", choices=["all", "text", "images"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing prepared files.")
    parser.add_argument("--wikitext-url", default=WIKITEXT_URL)
    parser.add_argument("--image-url", default=IMAGE_SOURCE_URL)
    parser.add_argument(
        "--image-source",
        type=Path,
        default=None,
        help=(
            "Local Figshare download to use for image data. Can be a zip/tar archive, "
            "the auto-unzipped data folder, or imagenet_training_single_objs itself."
        ),
    )
    parser.add_argument("--lines-per-shard", type=int, default=10_000)
    parser.add_argument("--max-shards", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = RepoPaths.default()

    if args.kind in {"all", "text"}:
        prepare_wikitext(
            output_dir=paths.raw_texts_root / "wikitext",
            url=args.wikitext_url,
            lines_per_shard=args.lines_per_shard,
            max_shards=args.max_shards,
            overwrite=args.overwrite,
        )

    if args.kind in {"all", "images"}:
        prepare_images(
            output_dir=paths.raw_images_root,
            url=args.image_url,
            image_source=args.image_source,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
