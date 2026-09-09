/**
 * The i18n seam.
 *
 * FRONTEND_PLAN.md F1: "no hard-coded user-facing strings ... set up the
 * i18n seam even if there is one locale." This is deliberately the
 * smallest thing that satisfies that: every user-facing string lives in a
 * locale JSON file and is looked up through `t()`, never inlined in a
 * component. There is exactly one locale today (en), so this hand-rolled
 * lookup is enough — no runtime library, no build-time compiler, nothing
 * that could reach for the network. Swapping in a real i18n library
 * (paraglide, svelte-i18n, ...) later is a change to this one file; call
 * sites (`t('some.key')`) do not need to move.
 */
import en from './locales/en.json';

const locales = { en } satisfies Record<string, Record<string, string>>;

export type Locale = keyof typeof locales;
export type MessageKey = keyof typeof en;

export const DEFAULT_LOCALE: Locale = 'en';

export function t(key: MessageKey, locale: Locale = DEFAULT_LOCALE): string {
	return locales[locale]?.[key] ?? locales[DEFAULT_LOCALE][key] ?? key;
}
