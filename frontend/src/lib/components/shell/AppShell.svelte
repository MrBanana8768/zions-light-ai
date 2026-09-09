<script lang="ts">
	/**
	 * The app frame: header (title + theme toggle), conversation rail,
	 * message pane. This is F1's "minimal app shell only" — layout and
	 * theme switching, nothing else. Chat logic, the store, and compactor
	 * calls are explicitly out of scope for this lane.
	 */
	import type { Snippet } from 'svelte';
	import ConversationRail from './ConversationRail.svelte';
	import ThemeToggle from './ThemeToggle.svelte';
	import { t } from '$lib/i18n';
	import { theme } from '$lib/theme.svelte';

	let { children }: { children: Snippet } = $props();

	// Keeps <html data-theme> in sync with the store after hydration. The
	// FIRST paint's theme comes from app.html's inline script, which reads
	// the same localStorage key synchronously before any CSS applies —
	// this effect only takes over for in-session changes (the toggle) and
	// for the OS-level "system" preference changing while the app is open.
	$effect(() => {
		const root = document.documentElement;
		root.setAttribute('data-theme', theme.resolved);
		root.setAttribute('data-theme-pref', theme.preference);
	});
</script>

<div class="shell">
	<a class="skip-link" href="#zl-main">{t('shell.skip_to_content')}</a>
	<header class="shell__header">
		<span class="shell__title">{t('app.title')}</span>
		<ThemeToggle />
	</header>
	<div class="shell__body">
		<aside class="shell__rail" aria-label={t('shell.conversations.heading')}>
			<ConversationRail />
		</aside>
		<main class="shell__main" id="zl-main">
			{@render children()}
		</main>
	</div>
</div>

<style>
	.shell {
		display: flex;
		flex-direction: column;
		height: 100dvh;
	}

	.skip-link {
		position: absolute;
		left: -9999px;
		top: 0;
		z-index: 100;
		padding: var(--zl-space-2) var(--zl-space-3);
		background: var(--zl-surface-raised);
		color: var(--zl-text);
		border: 1px solid var(--zl-border-strong);
		border-radius: var(--zl-radius-sm);
	}

	.skip-link:focus {
		left: var(--zl-space-2);
		top: var(--zl-space-2);
	}

	.shell__header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--zl-space-3) var(--zl-space-4);
		border-bottom: 1px solid var(--zl-border);
		background: var(--zl-surface);
		flex-shrink: 0;
	}

	.shell__title {
		font-weight: var(--zl-font-weight-semibold);
		font-size: var(--zl-font-size-md);
	}

	.shell__body {
		flex: 1;
		display: flex;
		min-height: 0;
	}

	.shell__rail {
		width: 280px;
		flex-shrink: 0;
		border-right: 1px solid var(--zl-border);
		background: var(--zl-bg-inset);
		overflow-y: auto;
	}

	.shell__main {
		flex: 1;
		min-width: 0;
		display: flex;
		flex-direction: column;
	}

	/* One-handed on a phone (FRONTEND_SPEC.md §13): the rail stacks above
	   the message pane instead of sitting beside it. The actual navigation
	   between the two (open rail / open conversation) is L3's — this only
	   scaffolds the breakpoint so it exists before any lane needs it. */
	@media (max-width: 640px) {
		.shell__body {
			flex-direction: column;
		}

		.shell__rail {
			width: 100%;
			max-height: 40dvh;
			border-right: none;
			border-bottom: 1px solid var(--zl-border);
		}
	}
</style>
