"""Claim 3b step 1: a fresh user logs into Element Web (real browser, real
Element code) and sets up Secure Backup through Element's own UI, in one
continuous session so no second device gets created along the way (a
second device before backup exists triggers a "confirm your identity"
dead-end this script can't solve without a second device to confirm from).
We only *observe* the recovery key Element shows us -- the importer never
generates, holds, or uploads a backup version.
"""
import os
import json
import re
import sys

from playwright.sync_api import sync_playwright

OUT = "/work/out"
USERNAME = sys.argv[1] if len(sys.argv) > 1 else "elem_owner3"
PASSWORD = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SPIKE_ELEM_PASSWORD", "CHANGE_ME")


def dump(pg, name):
    pg.screenshot(path=f"{OUT}/{name}.png", full_page=True)


def wait_idle(pg, seconds=1):
    pg.wait_for_timeout(seconds * 1000)


RECOVERY_KEY_RE = re.compile(r"([A-Za-z0-9]{4}(?:[\s\xa0][A-Za-z0-9]{4}){7,})")


def main():
    result = {"steps": [], "username": USERNAME}
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
        dump(pg, "3b_01_after_login")
        result["steps"].append(f"logged in as {USERNAME}, url={pg.url}")

        body = pg.inner_text("body")
        result["home_screen_text"] = body[:1500]

        recovery_key = None
        if "Back up your chats" in body or "Continue" in body:
            try:
                pg.get_by_text("Continue", exact=True).click(timeout=5000)
                wait_idle(pg, 2)
                dump(pg, "3b_02_after_continue")
                body2 = pg.inner_text("body")
                result["steps"].append(
                    "clicked Continue on backup banner -> landed on Settings/Encryption"
                )
                result["after_continue_text"] = body2[:2000]

                # "Allow key storage" is a toggle that is already ON by default
                # (see 3b_02_after_continue.png) -- clicking it would turn key
                # storage OFF and prompt to delete it instead. Leave it alone
                # and go straight for "Get recovery key" in the Backup section.
                if "Get recovery key" in body2:
                    pg.get_by_text("Get recovery key", exact=False).click(timeout=5000)
                    wait_idle(pg, 2)
                    dump(pg, "3b_03_recovery_key_dialog")
                    result["steps"].append("clicked Get recovery key")

                    pg.get_by_role("button", name="Continue", exact=True).click(timeout=5000)
                    wait_idle(pg, 2)
                    dump(pg, "3b_04_recovery_key_shown")
                    result["steps"].append("clicked Continue to generate the key")
                    dialog_text = pg.inner_text("body")
                    result["recovery_key_dialog_text"] = dialog_text[:3000]
                    m = RECOVERY_KEY_RE.search(dialog_text)
                    if m:
                        recovery_key = m.group(1)

                    # Next screen: "I saved it" continue -> lands on a
                    # confirmation screen that makes you TYPE THE KEY BACK IN
                    # before it will actually finish setup. Do that for real
                    # -- skipping it (as the first attempt with elem_owner7
                    # did) leaves setup incomplete.
                    pg.get_by_role("button", name="Continue", exact=True).click(timeout=5000)
                    wait_idle(pg, 2)
                    dump(pg, "3b_05_confirm_key_screen")
                    result["confirm_key_screen_text"] = pg.inner_text("body")[:2000]

                    if recovery_key and "Enter recovery key" in pg.inner_text("body"):
                        pg.get_by_label("Enter recovery key", exact=False).fill(recovery_key)
                        wait_idle(pg, 1)
                        pg.get_by_role("button", name="Finish set up", exact=True).click(timeout=5000)
                        wait_idle(pg, 2)
                        dump(pg, "3b_06_after_finish_setup")
                        result["steps"].append("typed recovery key back in and clicked Finish set up")
                        result["final_text"] = pg.inner_text("body")[:2000]
                    else:
                        result["steps"].append(
                            "no recovery-key-confirmation field found; setup may be incomplete"
                        )
                else:
                    result["steps"].append("no 'Get recovery key' button found")
            except Exception as e:
                result["steps"].append(f"backup banner flow failed: {e}")
        else:
            result["steps"].append("no backup banner seen")

        result["recovery_key"] = recovery_key
        b.close()

    with open(f"{OUT}/claim3b_setup_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
