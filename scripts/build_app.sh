#!/bin/bash
# Builds VideoEnhancer with SwiftPM and wraps the binary into build/VideoEnhancer.app.
# Works with Command Line Tools only (no Xcode required).
#
# Usage: scripts/build_app.sh [release|debug]
set -euo pipefail
shopt -s nullglob

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-release}"
APP="$ROOT/build/VideoEnhancer.app"

cd "$ROOT"
swift build -c "$CONFIG" --arch arm64
BIN_DIR="$(swift build -c "$CONFIG" --arch arm64 --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/VideoEnhancer" "$APP/Contents/MacOS/VideoEnhancer"
cp "$ROOT/Resources/Info.plist" "$APP/Contents/Info.plist"

# SwiftPM resource bundles (shader sources, models) — none yet in Stage 1.
for bundle in "$BIN_DIR"/*.bundle; do
    cp -R "$bundle" "$APP/Contents/Resources/"
done

# Core ML models: precompile to .mlmodelc when Xcode's coremlcompiler is available (faster first
# launch); otherwise ship the .mlpackage and the app compiles it on first use (cached).
mkdir -p "$APP/Contents/Resources/Models"
XCODE_DEV="/Applications/Xcode.app/Contents/Developer"
for package in "$ROOT"/Models/*.mlpackage; do
    if [ -d "$XCODE_DEV" ] && DEVELOPER_DIR="$XCODE_DEV" xcrun --find coremlcompiler >/dev/null 2>&1; then
        DEVELOPER_DIR="$XCODE_DEV" xcrun coremlcompiler compile "$package" "$APP/Contents/Resources/Models" >/dev/null
    else
        cp -R "$package" "$APP/Contents/Resources/Models/"
    fi
done

# Ad-hoc signature: enough to run locally on Apple Silicon.
codesign --force --sign - --timestamp=none "$APP"
echo "Built: $APP"
