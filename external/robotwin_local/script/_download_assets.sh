#!/usr/bin/env bash
# TronCamp kit: assets (embodiments / objects / background textures) are already bundled.
# The upstream RoboTwin asset-download flow (huggingface _download.py + unzip) is disabled
# here so it cannot overwrite the bundled, kit-tuned assets. Nothing to download.
echo "[assets] Assets are already bundled with this kit — no download needed. See README."
exit 0
