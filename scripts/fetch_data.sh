
#!/usr/bin/env bash

# Small data subsets for the dry run.
# Usage: bash dryrun/fetch_data.sh [DATA_DIR]
# Default: data/

set -u

# Resolve paths before changing directories.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Read TRACK relative to the script.
TRACK="$(cat "$SCRIPT_DIR/TRACK" 2>/dev/null || echo all)"

want() {
    [ "$TRACK" = all ] || [ "$TRACK" = "$1" ]
}

# Resolve data directory relative to the current terminal directory.
D="${1:-data}"
case "$D" in
    /*) ;;
    *) D="$PWD/$D" ;;
esac

mkdir -p "$D"
cd "$D" || exit 1

ok() {
    echo "  [ok] $1"
}

ko() {
    echo "  [FAILED] $1 -- related smoke steps will be skipped"
}

# Windows Git Bash download helper.
download() {
    curl.exe -fL --retry 3 --retry-delay 2 "$1" -o "$2"
}

# --------------------------------------------------
# SPEECH: FSDD
# --------------------------------------------------

if want speech; then

    echo "== FSDD (speech classical, 10 MB)"

    if [ -d fsdd ]; then
        ok fsdd
    elif git clone -q --depth 1 \
        https://github.com/Jakobovski/free-spoken-digit-dataset.git fsdd; then
        ok fsdd
    else
        ko fsdd
    fi

fi

# --------------------------------------------------
# VISION: OpenCV video
# --------------------------------------------------

if want vision; then

    echo "== Short street video (OpenCV sample)"

    if [ -f vtest.avi ]; then
        ok vtest.avi
    elif download \
        https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi \
        vtest.avi; then
        ok vtest.avi
    else
        ko vtest.avi
    fi

# --------------------------------------------------
# VISION: COCO
# --------------------------------------------------

    echo "== COCO val2017: annotations + 50 images"

    if [ ! -f coco/annotations/instances_val2017.json ]; then

        mkdir -p coco

        if download \
            https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
            coco/ann.zip &&
            unzip -q -o coco/ann.zip \
                "annotations/instances_val2017.json" -d coco; then

            rm -f coco/ann.zip
            ok annotations
        else
            rm -f coco/ann.zip
            ko annotations
        fi

    fi

    if [ -f coco/annotations/instances_val2017.json ]; then

        mkdir -p coco/val2017

        python - <<'PY'
import json
import random
import urllib.request
import pathlib

annotation_file = pathlib.Path("coco/annotations/instances_val2017.json")

with annotation_file.open() as f:
    a = json.load(f)

random.seed(0)
imgs = random.sample(a["images"], 50)

for im in imgs:
    p = pathlib.Path("coco/val2017") / im["file_name"]

    if not p.exists():
        print(f"Downloading {im['file_name']}")
        urllib.request.urlretrieve(im["coco_url"], p)

ids = {im["id"] for im in imgs}

a["images"] = imgs
a["annotations"] = [
    x for x in a["annotations"]
    if x["image_id"] in ids
]

with annotation_file.open("w") as f:
    json.dump(a, f)

print("  [ok] 50 COCO images")
PY

        if [ $? -ne 0 ]; then
            ko COCO-images
        fi

    else
        echo "  [skip] COCO annotations not available"
    fi

fi

# --------------------------------------------------
# SPEECH: Reference model
# --------------------------------------------------

if want speech; then

    MODEL_DIR="$PROJECT_ROOT/models"
    MODEL_NAME="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

    echo "== Reference on-device streaming model -> $MODEL_DIR"

    mkdir -p "$MODEL_DIR"

    if [ -d "$MODEL_DIR/$MODEL_NAME" ]; then
        ok reference-model

    elif (
        cd "$MODEL_DIR" &&
        download \
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$MODEL_NAME.tar.bz2" \
            "$MODEL_NAME.tar.bz2" &&
        tar xjf "$MODEL_NAME.tar.bz2" &&
        rm -f "$MODEL_NAME.tar.bz2"
    ); then
        ok reference-model

    else
        ko reference-model
    fi

# --------------------------------------------------
# SPEECH: LibriSpeech
# --------------------------------------------------

    echo "== LibriSpeech test-clean (346 MB)"

    if [ -d LibriSpeech/test-clean ]; then
        ok librispeech

    elif (
        download \
            https://www.openslr.org/resources/12/test-clean.tar.gz \
            test-clean.tar.gz &&
        tar xzf test-clean.tar.gz &&
        rm -f test-clean.tar.gz
    ); then
        ok librispeech

    else
        rm -f test-clean.tar.gz
        ko librispeech
    fi

# --------------------------------------------------
# SPEECH: ARPA language model
# --------------------------------------------------

    echo "== Pruned 3-gram language model (~34 MB)"

    if [ -f lm/3-gram.pruned.1e-7.arpa ]; then
        ok arpa

    elif (
        mkdir -p lm &&
        download \
            https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz \
            lm/lm.gz &&
        gunzip -c lm/lm.gz |
            tr '[:upper:]' '[:lower:]' > lm/3-gram.pruned.1e-7.arpa &&
        rm -f lm/lm.gz
    ); then
        ok arpa

    else
        rm -f lm/lm.gz
        ko arpa
    fi

fi

# --------------------------------------------------
# VISION: DAVIS
# --------------------------------------------------

if want vision; then

    echo "== DAVIS 2017 trainval 480p (~800 MB)"

    if [ -d DAVIS ]; then
        ok davis

    elif (
        download \
            https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip \
            DAVIS-2017-trainval-480p.zip &&
        unzip -q DAVIS-2017-trainval-480p.zip &&
        rm -f DAVIS-2017-trainval-480p.zip
    ); then
        ok davis

    else
        rm -f DAVIS-2017-trainval-480p.zip
        ko davis
    fi

fi

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

echo
echo "== Download summary =="
du -sh . 2>/dev/null || true
