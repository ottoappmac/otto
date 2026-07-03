#!/bin/bash
# Sign + notarize + staple a Tauri-built .app locally.
#
# Mirrors the "Sign nested backend binaries" and "Notarise and staple" steps
# from .github/workflows/build-app.yml so you can run the full release flow
# on your own Mac without burning GitHub Actions minutes.
#
# Prerequisites:
#   1. ./scripts/build_mac.sh has produced an unsigned .app under
#      app/src-tauri/target/release/bundle/macos/
#   2. A "Developer ID Application" certificate is installed in your
#      keychain. Verify with: security find-identity -v -p codesigning
#   3. Apple ID env vars are set (see CONFIG below).
#
# Usage:
#   ./scripts/sign_and_notarize.sh                # uses env vars
#   ./scripts/sign_and_notarize.sh --skip-dmg     # sign + notarize .app only
#   ./scripts/sign_and_notarize.sh --skip-notary  # sign locally, do not submit
#
# Required env vars (export, or put in .env.signing and source it first):
#   APPLE_SIGNING_IDENTITY  e.g. "Developer ID Application: Joe Smith (XALS6Q82AK)"
#   APPLE_ID                your Apple ID email
#   APPLE_PASSWORD          app-specific password (generate at appleid.apple.com)
#   APPLE_TEAM_ID           e.g. "XALS6Q82AK"
#
# Optional env vars:
#   APP                     path to the .app (auto-detected if unset)

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m==> %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m==> %s\033[0m\n' "$*"; exit 1; }

SKIP_DMG=false
SKIP_NOTARY=false
for arg in "$@"; do
  case "$arg" in
    --skip-dmg)    SKIP_DMG=true ;;
    --skip-notary) SKIP_NOTARY=true ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
  esac
done

# Auto-source a local env file if present (gitignored).
if [ -f ".env.signing" ]; then
  info "Sourcing .env.signing"
  # shellcheck disable=SC1091
  set -a; source .env.signing; set +a
fi

# ---------- Config & validation --------------------------------------------

command -v create-dmg &>/dev/null || fail "create-dmg not found — run: brew install create-dmg"

[ -n "$APPLE_SIGNING_IDENTITY" ] || fail "APPLE_SIGNING_IDENTITY not set"
if [ "$SKIP_NOTARY" = false ]; then
  [ -n "$APPLE_ID" ]       || fail "APPLE_ID not set"
  [ -n "$APPLE_PASSWORD" ] || fail "APPLE_PASSWORD not set"
  [ -n "$APPLE_TEAM_ID" ]  || fail "APPLE_TEAM_ID not set"
fi

if [ -z "$APP" ]; then
  APP=$(find app/src-tauri/target -maxdepth 6 -type d -name "OTTO.app" \
        -path "*/release/bundle/macos/*" -print -quit 2>/dev/null)
fi
[ -n "$APP" ] && [ -d "$APP" ] || fail "OTTO.app not found — run ./scripts/build_mac.sh first (or set APP=<path>)"

DMG_DIR="$(dirname "$APP")/../dmg"

ok "Using app:      $APP"
ok "Using identity: $APPLE_SIGNING_IDENTITY"

STAGING="${TMPDIR:-/tmp}/sign-staging-$$"
rm -rf "$STAGING" && mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT

# sign_one: stage to flat dir, strip existing signature, re-sign with
# Developer ID + hardened runtime + secure timestamp, write back in place.
# The flat staging path prevents codesign from interpreting the parent
# directory as a bundle ("bundle format is ambiguous").
sign_one() {
  local target="$1"
  local staged="$STAGING/$(basename "$target").$RANDOM"
  cp "$target" "$staged"
  codesign --remove-signature "$staged" 2>/dev/null || true
  codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
    --options runtime --timestamp \
    "$staged"
  cat "$staged" > "$target"
  rm -f "$staged"
}

# ---------- 1. Sign nested binaries inside the .app ------------------------

BACKEND="$APP/Contents/Resources/backend"
[ -d "$BACKEND" ] || fail "Backend resources not found at $BACKEND"

info "Signing Mach-O files outside Python.framework..."
find "$BACKEND" -type f \
  \( -name "*.dylib" -o -name "*.so" -o -perm +111 \) \
  ! -path "*/Python.framework/*" \
| while read -r f; do
  file "$f" | grep -qE "Mach-O|shared library" && sign_one "$f"
done

# ---------- 2. Restructure & sign Python.framework -------------------------
# PyInstaller ships a non-standard Python.framework where Python.framework/
# Python is a real binary (not a symlink to Versions/Current/Python) and
# Versions/Current may be a real directory. codesign sees this as ambiguous
# (looks like both an app and a framework) and either refuses to sign or
# produces a corrupt signature that Apple's notariser then rejects with
# "The signature of the binary is invalid".
#
# Fix: convert to the canonical macOS framework symlink layout, add a
# minimal Info.plist (so codesign can identify it as FMWK), then sign the
# versioned bundle and the framework root in turn.

FW="$BACKEND/_internal/Python.framework"
VERS="$FW/Versions/3.12"

if [ -d "$FW" ]; then
  info "Fixing Python.framework structure..."
  [ -d "$VERS" ] || fail "Expected $VERS to exist"

  mkdir -p "$VERS/Resources"
  # Same base64-encoded Info.plist used by the GitHub workflow.
  echo "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPCFET0NUWVBFIHBsaXN0IFBVQkxJQyAiLS8vQXBwbGUvL0RURCBQTElTVCAxLjAvL0VOIgogICJodHRwOi8vd3d3LmFwcGxlLmNvbS9EVERzL1Byb3BlcnR5TGlzdC0xLjAuZHRkIj4KPHBsaXN0IHZlcnNpb249IjEuMCI+CjxkaWN0PgogIDxrZXk+Q0ZCdW5kbGVJZGVudGlmaWVyPC9rZXk+ICAgPHN0cmluZz5vcmcucHl0aG9uLnB5dGhvbjwvc3RyaW5nPgogIDxrZXk+Q0ZCdW5kbGVOYW1lPC9rZXk+ICAgICAgICAgPHN0cmluZz5QeXRob248L3N0cmluZz4KICA8a2V5PkNGQnVuZGxlUGFja2FnZVR5cGU8L2tleT4gIDxzdHJpbmc+Rk1XSzwvc3RyaW5nPgogIDxrZXk+Q0ZCdW5kbGVWZXJzaW9uPC9rZXk+ICAgICAgPHN0cmluZz4zLjEyLjA8L3N0cmluZz4KICA8a2V5PkNGQnVuZGxlU2hvcnRWZXJzaW9uU3RyaW5nPC9rZXk+IDxzdHJpbmc+My4xMjwvc3RyaW5nPgo8L2RpY3Q+CjwvcGxpc3Q+" \
    | base64 --decode > "$VERS/Resources/Info.plist"

  if [ -d "$FW/Versions/Current" ] && [ ! -L "$FW/Versions/Current" ]; then
    warn "Converting Versions/Current from directory to symlink"
    rm -rf "$FW/Versions/Current"
  fi
  [ -e "$FW/Versions/Current" ] || ln -s "3.12" "$FW/Versions/Current"

  # An embedded framework root must contain ONLY Versions/ + symlinks.
  # Anything else triggers: "unsealed contents present in the root
  # directory of an embedded framework". For each real (non-symlink) item
  # at the root other than Versions/, ensure the canonical copy lives in
  # Versions/3.12/ and replace the root item with a symlink.
  info "Python.framework root before cleanup:"
  ls -la "$FW/"
  for item in "$FW"/*; do
    name=$(basename "$item")
    [ "$name" = "Versions" ] && continue
    [ -L "$item" ] && continue
    if [ -e "$FW/Versions/3.12/$name" ]; then
      echo "  $name: exists in Versions/3.12/ — replacing root copy with symlink"
      rm -rf "$item"
    else
      echo "  $name: not in Versions/3.12/ — moving there, adding symlink"
      mv "$item" "$FW/Versions/3.12/$name"
    fi
    ln -s "Versions/Current/$name" "$FW/$name"
  done
  info "Python.framework root after cleanup:"
  ls -la "$FW/"

  if [ -f "$VERS/Python" ]; then
    info "Signing $VERS/Python"
    sign_one "$VERS/Python"
  fi

  info "Signing other Mach-O files inside $VERS"
  find "$VERS" -type f \( -name "*.dylib" -o -name "*.so" \) \
  | while read -r f; do
    file "$f" | grep -qE "Mach-O|shared library" && sign_one "$f"
  done

  info "Sealing the versioned framework bundle"
  codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
    --options runtime --timestamp \
    --identifier "org.python.python" \
    "$VERS"

  info "Sealing the framework root"
  codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
    --options runtime --timestamp \
    --identifier "org.python.python" \
    "$FW"

  codesign --verify --verbose=2 "$FW" || warn "(framework verify warning)"
fi

# ---------- 3. Main backend executable -------------------------------------

if [ -f "$BACKEND/otto-backend" ]; then
  info "Signing $BACKEND/otto-backend"
  sign_one "$BACKEND/otto-backend"
fi

# ---------- 4. Re-sign the outer bundle ------------------------------------

info "Signing main Tauri binary and outer bundle"
codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
  --options runtime --timestamp \
  "$APP/Contents/MacOS/otto"
codesign --force --sign "$APPLE_SIGNING_IDENTITY" \
  --options runtime --timestamp \
  "$APP"

codesign --verify --verbose=2 "$APP" || warn "(outer verify warning)"
codesign --display --verbose=2 "$APP" 2>&1 | grep -E "flags=|Timestamp=" || true

ok "Signing complete"

if [ "$SKIP_NOTARY" = true ]; then
  warn "--skip-notary set; stopping after signing."
  exit 0
fi

# ---------- 5. Notarise & staple the .app ----------------------------------

ZIP="${TMPDIR:-/tmp}/OTTO.zip"
info "Zipping for notary submission: $ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

info "Submitting .app to Apple notary service (this can take 5–30 min)..."
NOTARY_OUT="${TMPDIR:-/tmp}/notary-app.json"
xcrun notarytool submit "$ZIP" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait \
  --output-format json | tee "$NOTARY_OUT" || true

NOTARY_STATUS=$(python3 -c "import json; print(json.load(open('$NOTARY_OUT')).get('status',''))" 2>/dev/null || echo "")
NOTARY_ID=$(python3     -c "import json; print(json.load(open('$NOTARY_OUT')).get('id',''))"     2>/dev/null || echo "")

if [ "$NOTARY_STATUS" != "Accepted" ]; then
  warn "Notarisation status: $NOTARY_STATUS — fetching Apple log..."
  if [ -n "$NOTARY_ID" ]; then
    xcrun notarytool log "$NOTARY_ID" \
      --apple-id "$APPLE_ID" \
      --password "$APPLE_PASSWORD" \
      --team-id "$APPLE_TEAM_ID" || true
  fi
  fail "Notarisation did not succeed"
fi

ok "Notarisation accepted; stapling ticket to .app"
xcrun stapler staple "$APP"

if [ "$SKIP_DMG" = true ]; then
  ok "Done (.app only — DMG skipped)."
  exit 0
fi

# ---------- 6. Rebuild + notarise the .dmg ---------------------------------
# Tauri's original DMG was built before we re-signed the nested backend, so
# it's stale. Recreate it from the now-stapled .app.

mkdir -p "$DMG_DIR"
EXISTING_DMG=$(find "$DMG_DIR" -maxdepth 1 -name "*.dmg" -print -quit 2>/dev/null)
if [ -n "$EXISTING_DMG" ]; then
  info "Replacing existing DMG: $EXISTING_DMG"
  rm -f "$EXISTING_DMG"
  DMG_PATH="$EXISTING_DMG"
else
  DMG_PATH="$DMG_DIR/OTTO.dmg"
  info "Creating new DMG: $DMG_PATH"
fi

create-dmg \
  --volname "OTTO" \
  --volicon "app/src-tauri/icons/icon.icns" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "OTTO.app" 175 190 \
  --hide-extension "OTTO.app" \
  --app-drop-link 425 190 \
  "$DMG_PATH" \
  "$(dirname "$APP")/"

codesign --force --sign "$APPLE_SIGNING_IDENTITY" --timestamp "$DMG_PATH"

info "Submitting .dmg to Apple notary service..."
NOTARY_DMG_OUT="${TMPDIR:-/tmp}/notary-dmg.json"
xcrun notarytool submit "$DMG_PATH" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait \
  --output-format json | tee "$NOTARY_DMG_OUT" || true

NOTARY_DMG_STATUS=$(python3 -c "import json; print(json.load(open('$NOTARY_DMG_OUT')).get('status',''))" 2>/dev/null || echo "")
NOTARY_DMG_ID=$(python3     -c "import json; print(json.load(open('$NOTARY_DMG_OUT')).get('id',''))"     2>/dev/null || echo "")

if [ "$NOTARY_DMG_STATUS" != "Accepted" ]; then
  warn "DMG notarisation status: $NOTARY_DMG_STATUS — fetching Apple log..."
  if [ -n "$NOTARY_DMG_ID" ]; then
    xcrun notarytool log "$NOTARY_DMG_ID" \
      --apple-id "$APPLE_ID" \
      --password "$APPLE_PASSWORD" \
      --team-id "$APPLE_TEAM_ID" || true
  fi
  fail "DMG notarisation did not succeed"
fi

xcrun stapler staple "$DMG_PATH"

ok "All done!"
echo ""
echo "  App:  $APP   (signed, notarised, stapled)"
echo "  DMG:  $DMG_PATH"
