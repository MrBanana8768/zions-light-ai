# Zion's Light — client

The SvelteKit front end that replaces OpenWebUI. See the repo root's
`FRONTEND_PLAN.md` and `FRONTEND_HANDOFF.md` for why this exists and what it
must and must not do; see `docs/lanes/L0-scaffold.md` for what this specific
lane (repo scaffold + co-location) built and every decision it took that the
plan left open.

**Scope note.** This is the L0 scaffold only: app shell, theme, design
tokens, i18n seam. No chat logic, no conversation store, no calls to the
compactor live here yet — those land in later lanes (L1–L6, see
`FRONTEND_PLAN.md` §5).

## Requirements

- Node.js **>= 22** (see `package.json`'s `engines.node`; the Dockerfile
  pins an exact version — see `docs/lanes/L0-scaffold.md`).
- npm (ships with Node).

No other runtime dependency. Nothing in this app fetches an asset over the
network at build time or run time except `npm install` itself pulling
packages from the npm registry — the built app is fully self-contained and
must run with no internet access (RunPod pods build with network access but
serve without it).

## Run it

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173, hot reload
```

## Type-check

```bash
npm run check           # svelte-check, one-shot
npm run check:watch     # same, watching
```

## Build for production

```bash
npm run build            # writes to build/ (adapter-node output)
node build/index.js      # runs the built server
```

The built server reads its port and host from the standard adapter-node
environment variables:

| Variable | Default | Notes |
|---|---|---|
| `PORT` | `3000` | This project runs it on **3001** — see `[program:client]` in the repo root's `supervisord.conf`. |
| `HOST` | `0.0.0.0` | |
| `ORIGIN` | — | Set this if the app needs to know its own public URL (e.g. for CSRF checks on form actions). Not required for the scaffold; a later lane should set it once the app takes real user input. |

`npm run preview` (plain `vite preview`) also works for a quick local check
of the built client, but it does not exercise the real adapter-node server —
use `node build/index.js` to test what actually ships.

## What's here (L0 scope)

- `src/app.html` — the pre-paint theme script (`FRONTEND_SPEC.md` §13:
  "theme applied before first paint"). Sets `data-theme` on `<html>` from
  `localStorage` before any stylesheet loads, so there's no light/dark flash.
- `src/lib/styles/tokens.css` — every design token (color, space, type,
  radii, motion) as CSS custom properties, light/dark/system. **No
  `@telos-llc/*` dependency** — see `docs/lanes/L0-scaffold.md` for why.
- `src/lib/styles/{reset,global}.css` — a small reset and base styles
  (focus-visible, `prefers-reduced-motion`).
- `src/lib/theme.svelte.ts` — the post-hydration half of theme switching
  (Svelte 5 runes state shared across components).
- `src/lib/i18n/` — the i18n seam: `t('some.key')` looked up against
  `locales/en.json`. No hard-coded user-facing strings anywhere in this
  lane's components.
- `src/lib/components/shell/` — `AppShell`, `ConversationRail` (empty),
  `MessagePane` (placeholder), `ThemeToggle`. Layout only; no data.
- `src/routes/` — `+layout.svelte` wires the shell and global CSS;
  `+page.svelte` renders the empty message pane.

## Fonts

System font stack (`--zl-font-sans` / `--zl-font-mono` in `tokens.css`), not
a vendored webfont. No Google Fonts, no CDN, no runtime font fetch. If a
brand typeface is wanted later, commit real `.woff2` files under `static/`
and `@font-face` them there — that's a token-file change, not an
architecture change.
