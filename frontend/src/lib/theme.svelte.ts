/**
 * Theme preference store — light / dark / system.
 *
 * This is the POST-HYDRATION half of "theme applied before first paint"
 * (FRONTEND_SPEC.md §13). The PRE-hydration half is the inline script in
 * src/app.html, which reads the same localStorage key synchronously and
 * sets data-theme on <html> before any stylesheet paints, so there is no
 * flash. This module exists so a user's in-app choice (a) persists the
 * same way the inline script reads it back, and (b) reactively updates the
 * DOM attribute after the app has hydrated, without a page reload.
 *
 * `.svelte.ts` (not `.ts`) because it uses runes outside a component —
 * Svelte 5's supported pattern for state that several components share.
 */

export type ThemePreference = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

const STORAGE_KEY = 'zl-theme';

function resolveSystemTheme(): ResolvedTheme {
	if (typeof window === 'undefined') return 'light';
	return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function readStoredPreference(): ThemePreference {
	if (typeof localStorage === 'undefined') return 'system';
	try {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (stored === 'light' || stored === 'dark' || stored === 'system') return stored;
	} catch {
		// Private browsing / disabled storage — fall back to "system" rather
		// than throwing. Matches the inline pre-paint script's own try/catch.
	}
	return 'system';
}

class ThemeStore {
	preference = $state<ThemePreference>(readStoredPreference());
	#systemTheme = $state<ResolvedTheme>(resolveSystemTheme());

	resolved: ResolvedTheme = $derived(
		this.preference === 'system' ? this.#systemTheme : this.preference
	);

	constructor() {
		if (typeof window !== 'undefined') {
			const media = window.matchMedia('(prefers-color-scheme: dark)');
			const onChange = () => {
				this.#systemTheme = media.matches ? 'dark' : 'light';
			};
			media.addEventListener('change', onChange);
		}
	}

	set(preference: ThemePreference): void {
		this.preference = preference;
		try {
			localStorage.setItem(STORAGE_KEY, preference);
		} catch {
			// Non-fatal: the choice still applies for this session via $state,
			// it just won't survive a reload.
		}
	}
}

export const theme = new ThemeStore();
