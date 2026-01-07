#!/bin/bash
#
# KERAAL Dataset Download Script for HPC
# Usage: bash scripts/hpc/download_data.sh [group]
#
# Examples:
#   bash scripts/hpc/download_data.sh all      # Download all
#   bash scripts/hpc/download_data.sh labeled  # Only labeled (1A, 2A, 3)
#   bash scripts/hpc/download_data.sh group3   # Only Group3
#

set -e

KERAAL_URL="https://keraal.enstb.org/downloads"
DATA_DIR="${HOME}/keraal/data/raw"

echo "============================================================"
echo "KERAAL Dataset Downloader"
echo "============================================================"

# Create data directory
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

download_and_extract() {
    local filename=$1
    local size=$2

    echo ""
    echo "Downloading ${filename} (${size})..."

    if [ -f "${filename}" ]; then
        echo "  File exists, skipping download"
    else
        wget -q --show-progress -O ${filename} "${KERAAL_URL}/${filename}"
    fi

    # Extract
    local dirname="${filename%.tar.xz}"
    if [ -d "${dirname}" ]; then
        echo "  Already extracted: ${dirname}/"
    else
        echo "  Extracting..."
        tar -xJf ${filename}
        echo "  Done: ${dirname}/"
    fi
}

# Parse argument
GROUP=${1:-"labeled"}

case ${GROUP} in
    "labeled")
        echo "Downloading labeled datasets (Group1A, Group2A, Group3)"
        download_and_extract "group1A.tar.xz" "1.4GB"
        download_and_extract "group2A.tar.xz" "338MB"
        download_and_extract "group3.tar.xz" "8.2GB"
        ;;
    "all")
        echo "Downloading ALL datasets"
        download_and_extract "group1A.tar.xz" "1.4GB"
        download_and_extract "group1B.tar.xz" "13GB"
        download_and_extract "group2A.tar.xz" "338MB"
        download_and_extract "group2B.tar.xz" "1.3GB"
        download_and_extract "group3.tar.xz" "8.2GB"
        ;;
    "group1A"|"group1a")
        download_and_extract "group1A.tar.xz" "1.4GB"
        ;;
    "group2A"|"group2a")
        download_and_extract "group2A.tar.xz" "338MB"
        ;;
    "group3")
        download_and_extract "group3.tar.xz" "8.2GB"
        ;;
    "quick")
        echo "Quick start: Group1A + Group2A only"
        download_and_extract "group1A.tar.xz" "1.4GB"
        download_and_extract "group2A.tar.xz" "338MB"
        ;;
    *)
        echo "Usage: $0 [all|labeled|quick|group1A|group2A|group3]"
        echo ""
        echo "Options:"
        echo "  all     - Download all groups (~24GB)"
        echo "  labeled - Download labeled groups only (1A, 2A, 3) ~10GB"
        echo "  quick   - Quick start (1A + 2A only) ~1.7GB"
        echo "  group1A - Patient data with labels (1.4GB)"
        echo "  group2A - Healthy data with labels (338MB)"
        echo "  group3  - Error-labeled data (8.2GB)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Download Complete!"
echo "Data location: ${DATA_DIR}"
echo ""
echo "Directory contents:"
ls -lh ${DATA_DIR}
echo "============================================================"
