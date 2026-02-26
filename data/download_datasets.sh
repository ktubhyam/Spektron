#!/bin/bash
# Spektron Dataset Download Script
# Run this locally to download all required datasets

echo "=== Spektron Dataset Download ==="

# 1. Corn dataset (Eigenvector)
echo "[1/5] Downloading Corn dataset..."
mkdir -p data/raw/corn
wget -O data/raw/corn/corn.mat "http://eigenvector.com/data/Corn/corn.mat"
echo "  → 80 samples × 3 instruments × 700 channels (1100-2498nm)"

# 2. Tablet dataset (IDRC 2002 Shootout)
echo "[2/5] Downloading Tablet dataset..."
mkdir -p data/raw/tablet
wget -O data/raw/tablet/shootout_2002.mat "http://eigenvector.com/data/tablets/shootout_2002.mat"
echo "  → 654 tablets × 2 spectrometers"

# 3. Raman API dataset (Figshare)
echo "[3/5] Downloading Raman API dataset..."
mkdir -p data/raw/raman_api
wget -O data/raw/raman_api/raman_api.zip "https://figshare.com/ndownloader/articles/27931131/versions/1"
cd data/raw/raman_api && unzip -o raman_api.zip && cd ../../..

# 4. ChEMBL IR-Raman (Figshare - large)
echo "[4/5] Downloading ChEMBL IR-Raman pretraining corpus..."
mkdir -p data/raw/chembl
echo "  WARNING: ~220K spectra, may take a while"
wget -O data/raw/chembl/chembl_spectra.zip "https://figshare.com/ndownloader/articles/29037140/versions/2"

# 5. RRUFF Mineral Raman
echo "[5/5] Downloading RRUFF Raman spectra..."
mkdir -p data/raw/rruff
wget -O data/raw/rruff/rruff_raman.zip "http://rruff.info/zipped_data_files/raman/LR-Raman.zip"

echo "=== Downloads complete ==="
echo "Run: python src/data/preprocess.py to process all datasets"
