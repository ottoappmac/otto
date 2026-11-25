"""Clicks Allow / OK on macOS permission prompts in the MacArena VM, and nowhere else.

Replaces MacArena's ``vm_files/allow_permission.py`` on the Otto-native track. That script clicks any button named
"Allow", "OK" or "Always Allow" in any window of any process, every 0.1 s, and the VM runs five copies of it. So
every app dialog with an OK button (LibreOffice's Format Cells, Rename Sheet, Insert Rows...) was confirmed with its
defaults about 0.25 s after it opened, before an agent could read or fill it. Measured on the osworld VM: Format
Cells opened at 0.71 s and was gone at 0.97 s; with the clickers paused it stayed open.

Permission prompts come from background-only agents (UserNotificationCenter, universalAccessAuthWarn,
CoreServicesUIAgent, ...). Regular apps (Finder, LibreOffice, Notes...) are the processes System Events reports with
``background only`` false, so only the others are searched.
"""
import subprocess
import sys
import time

BUTTONS = ("Allow", "OK", "Always Allow")

SCRIPT = """
tell application "System Events"
    repeat with proc in (application processes whose background only is true)
        try
            tell proc
                repeat with win in windows
                    try
                        repeat with b in buttons of win
                            set bName to name of b
                            if bName is in {%s} then
                                click b
                                return "Clicked " & bName & " in " & (name of proc)
                            end if
                        end repeat
                    end try
                end repeat
            end tell
        end try
    end repeat
    return ""
end tell
""" % ", ".join(f'"{b}"' for b in BUTTONS)


def main(interval: float = 0.5) -> None:
    while True:
        try:
            out = subprocess.run(["osascript", "-e", SCRIPT], capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                print(time.strftime("%H:%M:%S"), out, flush=True)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
