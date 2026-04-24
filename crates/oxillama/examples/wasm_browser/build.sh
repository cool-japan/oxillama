#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

echo "Building oxillama-wasm with wasm-pack..."
wasm-pack build \
  --target web \
  --release \
  --out-dir "${SCRIPT_DIR}/pkg" \
  "${WORKSPACE_ROOT}/crates/oxillama-wasm"

echo ""
echo "Build complete! pkg/ is ready."
echo "Start a server: python3 -m http.server 8080"
echo "Then open: http://localhost:8080"
