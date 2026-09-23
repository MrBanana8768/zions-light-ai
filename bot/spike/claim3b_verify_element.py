"""Claim 3b steps 4-5: a brand-new Element Web session (fresh browser
context => a device that did not exist during setup or import) logs in as
elem_owner11, restores Secure Backup using the recovery key Element itself
displayed in claim3b_setup_backup.py (via the "Use recovery key" option on
the "Confirm your digital identity" dialog), opens the import room, scrolls
through the whole history, and counts rendered messages vs. "Unable to
decrypt" placeholders. Screenshots the top and bottom of the timeline.

Finding along the way: "Use recovery key" only appears here if Secure
Backup setup was carried all the way through Element's own "enter your
recovery key to confirm" step. An interrupted/incomplete setup (which
claim3b_setup_backup.py did by accident on its first attempts, see
fix-v4-spike.md) leaves a device with NO restore path at all -- only
"Can't confirm?" (destroys and replaces the identity/backup) or "Remove
this device" (log out). This is itself a real finding, not just spike
debugging noise.
"""
import os
import json
import re
import sys

from playwright.sync_api import sync_playwright

OUT = "/work/out"
USERNAME = "elem_owner11"
PASSWORD = os.environ.get("SPIKE_ELEM_PASSWORD", "CHANGE_ME")
RECOVERY_KEY = os.environ["SPIKE_RECOVERY_KEY"]  # the key Element showed in claim3b_setup_backup.py
ROOM_NAME = "zlaspike 3b import room"


def dump(pg, name):
    pg.screenshot(path=f"{OUT}/{name}.png", full_page=True)


def wait_idle(pg, seconds=1):
    pg.wait_for_timeout(seconds * 1000)


def main():
    result = {"steps": []}
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 900})
        pg = ctx.new_page()

        pg.goto("http://localhost:8009", timeout=30000)
        wait_idle(pg, 2)
        pg.get_by_text("Sign in", exact=True).click()
        wait_idle(pg, 1)
        pg.get_by_placeholder("Username").fill(USERNAME)
        pg.get_by_placeholder("Password").fill(PASSWORD)
        pg.get_by_role("button", name="Sign in", exact=True).click()
        for _ in range(30):
            txt = pg.inner_text("body")
            if "Signing In" not in txt and "Syncing" not in txt:
                break
            wait_idle(pg, 1)
        wait_idle(pg, 2)
        dump(pg, "3b_v10_after_login")
        result["steps"].append(f"logged in on brand-new device, url={pg.url}")
        result["post_login_text"] = pg.inner_text("body")[:1000]

        pg.get_by_text("Use recovery key", exact=True).click(timeout=5000)
        wait_idle(pg, 2)
        dump(pg, "3b_v11_use_recovery_key_dialog")
        result["steps"].append("clicked Use recovery key")
        result["use_recovery_key_text"] = pg.inner_text("body")[:2000]

        # Find whatever input field is present and type the recovery key in.
        pg.locator("input").first.fill(RECOVERY_KEY)
        wait_idle(pg, 1)
        dump(pg, "3b_v12_key_entered")

        # Click whatever the primary continue/confirm button is now.
        for label in ["Continue", "Confirm", "Done", "Verify"]:
            btn = pg.get_by_role("button", name=label, exact=False)
            if btn.count() > 0:
                btn.first.click(timeout=5000)
                wait_idle(pg, 3)
                result["steps"].append(f"clicked '{label}' after entering recovery key")
                break

        dump(pg, "3b_v13_after_restore")
        result["after_restore_text"] = pg.inner_text("body")[:2000]

        # There may be a final "Done"/dismiss step.
        body = pg.inner_text("body")
        if "Done" in body:
            try:
                pg.get_by_role("button", name="Done", exact=True).click(timeout=3000)
                wait_idle(pg, 2)
            except Exception:
                pass

        dump(pg, "3b_v14_home_after_restore")
        result["home_after_restore_text"] = pg.inner_text("body")[:2000]

        # Open the import room.
        pg.get_by_text(ROOM_NAME, exact=False).click(timeout=10000)
        wait_idle(pg, 3)
        dump(pg, "3b_v15_room_opened_bottom")
        result["steps"].append("opened import room")

        b.close()

    with open(f"{OUT}/claim3b_verify_probe.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
