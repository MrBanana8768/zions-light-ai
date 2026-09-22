"""Claim 3b, final run: brand-new Element Web device, zero prior devices,
restores elem_owner11's Secure Backup with the recovery key Element itself
displayed during setup, opens the 600-message import room, scrolls to the
very top and back to the bottom, and counts every rendered message vs.
every "Unable to decrypt" (or equivalent) placeholder. Screenshots both
ends. Also checks for trust/verification warnings on the messages and
whether timestamps are shown.
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
ROOM_NAME = "zlaspike 3b FINAL import room"


def dump(pg, name, full_page=False):
    # Element's timeline is a virtualized/absolutely-positioned scroller;
    # full-page screenshots stitch it incorrectly (blank gaps). Viewport
    # screenshots show exactly what's on screen, which is what we want here.
    pg.screenshot(path=f"{OUT}/{name}.png", full_page=full_page)


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
        result["steps"].append(f"logged in on brand-new device, url={pg.url}")
        result["device_id_note"] = "captured via admin API after the run"

        pg.get_by_text("Use recovery key", exact=True).click(timeout=10000)
        wait_idle(pg, 2)
        result["steps"].append("clicked Use recovery key")

        pg.locator("input").first.fill(RECOVERY_KEY)
        wait_idle(pg, 1)
        pg.get_by_role("button", name="Continue", exact=True).click(timeout=5000)
        wait_idle(pg, 3)
        dump(pg, "3b_final_01_after_restore")
        result["after_restore_text"] = pg.inner_text("body")[:1000]
        result["steps"].append("entered recovery key, restored backup")

        body = pg.inner_text("body")
        if "Done" in body:
            try:
                pg.get_by_role("button", name="Done", exact=True).click(timeout=3000)
                wait_idle(pg, 2)
            except Exception:
                pass

        pg.get_by_text(ROOM_NAME, exact=False).click(timeout=10000)
        wait_idle(pg, 3)
        result["steps"].append("opened import room")
        # Element opens a room scrolled to its latest message by default --
        # capture that as the "bottom" evidence shot before we scroll away.
        try:
            pg.get_by_role("button", name="Dismiss", exact=True).click(timeout=2000)
        except Exception:
            pass
        dump(pg, "3b_final_00_bottom_on_open")

        # Element virtualizes/unmounts the timeline as you scroll, so a
        # single DOM snapshot only ever sees a window of it. Collect
        # decrypted message indices (parsed from our own synthetic body
        # text, "[synthetic 3b #N ...]") and undecryptable event ids
        # incrementally into sets while scrolling all the way up, so the
        # final counts are a true union across the whole scroll, not a
        # snapshot.
        collect_js = """() => {
            const tiles = Array.from(document.querySelectorAll('.mx_EventTile'));
            const indices = [];
            let undecryptable = 0;
            for (const el of tiles) {
                const text = el.innerText || '';
                const m = text.match(/\\[synthetic 3b #(\\d+)/);
                if (m) {
                    indices.push(parseInt(m[1], 10));
                } else if (el.querySelector('.mx_DecryptionFailureBody') || /Unable to decrypt/i.test(text)) {
                    undecryptable++;
                }
            }
            return {indices, undecryptable};
        }"""

        scroll_up_js = """() => {
            const candidates = Array.from(document.querySelectorAll('div'));
            let best = null, bestScore = 0;
            for (const el of candidates) {
                if (el.scrollHeight > el.clientHeight + 50) {
                    const score = el.scrollHeight;
                    if (el.querySelector('.mx_EventTile') && score > bestScore) {
                        best = el; bestScore = score;
                    }
                }
            }
            if (best) { best.scrollTop = 0; return true; }
            return false;
        }"""

        seen_indices = set()
        seen_undecryptable_estimate = 0
        no_new_count = 0
        i = 0
        for i in range(300):
            snap = pg.evaluate(collect_js)
            before = len(seen_indices)
            seen_indices.update(snap["indices"])
            seen_undecryptable_estimate = max(seen_undecryptable_estimate, snap["undecryptable"])
            grew = len(seen_indices) > before
            pg.evaluate(scroll_up_js)
            wait_idle(pg, 1.2)
            if not grew:
                no_new_count += 1
                if no_new_count > 15:
                    break
            else:
                no_new_count = 0
            if 0 in seen_indices:
                # give it a couple more iterations to make sure nothing else
                # trickles in, then stop.
                pass
        wait_idle(pg, 1)
        dump(pg, "3b_final_02_scrolled_to_top")
        result["steps"].append(f"scrolled to top of timeline after {i+1} scroll steps")

        result["counts_union"] = {
            "unique_decrypted_indices_seen": len(seen_indices),
            "min_index_seen": min(seen_indices) if seen_indices else None,
            "max_index_seen": max(seen_indices) if seen_indices else None,
            "missing_indices": sorted(set(range(600)) - seen_indices)[:20],
            "missing_count": len(set(range(600)) - seen_indices),
            "undecryptable_estimate": seen_undecryptable_estimate,
        }

        # Check for trust/warning indicators (shields, "unverified" tooltips).
        warnings = pg.evaluate(
            """() => {
                const shields = document.querySelectorAll('[aria-label*="Encrypted by an unverified" i], [aria-label*="unverified" i], .mx_EventTile_e2eIcon');
                return Array.from(shields).slice(0, 5).map(e => e.getAttribute('aria-label') || e.className);
            }"""
        )
        result["warning_indicators_sample"] = warnings

        # Timestamps: hover/check one message's title attribute or visible ts.
        ts_info = pg.evaluate(
            """() => {
                const tsEls = Array.from(document.querySelectorAll('.mx_MessageTimestamp'));
                return tsEls.slice(0, 3).map(e => ({text: e.innerText, title: e.getAttribute('title')}))
                    .concat(tsEls.slice(-3).map(e => ({text: e.innerText, title: e.getAttribute('title')})));
            }"""
        )
        result["timestamp_samples"] = ts_info

        # Scroll back to the bottom for the second screenshot.
        for _ in range(50):
            pg.mouse.wheel(0, 5000)
            wait_idle(pg, 0.2)
        wait_idle(pg, 2)
        dump(pg, "3b_final_03_scrolled_to_bottom")
        result["steps"].append("scrolled back to bottom")

        counts_bottom = pg.evaluate(
            """() => {
                const bodies = Array.from(document.querySelectorAll('.mx_EventTile'));
                let decrypted = 0, undecryptable = 0, other = 0;
                for (const el of bodies) {
                    const text = el.innerText || '';
                    if (el.querySelector('.mx_DecryptionFailureBody') || text.includes('Unable to decrypt')) {
                        undecryptable++;
                    } else if (text.includes('[synthetic 3b')) {
                        decrypted++;
                    } else {
                        other++;
                    }
                }
                return {decrypted, undecryptable, other, total: bodies.length};
            }"""
        )
        result["counts_at_bottom"] = counts_bottom

        b.close()

    with open(f"{OUT}/claim3b_verify_final_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
