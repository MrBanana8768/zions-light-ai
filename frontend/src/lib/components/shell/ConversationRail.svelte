<script lang="ts">
	/**
	 * The empty conversation-list rail (FRONTEND_PLAN.md F1: "an empty
	 * conversation-list rail"). No store, no fetch, no chat logic — this
	 * lane (L0) builds the shell only; the conversation list itself
	 * (loading, rename, search — F17) is L3's, after the store/transport
	 * interface freeze.
	 */
	import { t } from '$lib/i18n';
</script>

<div class="rail">
	<h2 class="rail__heading">{t('shell.conversations.heading')}</h2>
	<p class="rail__empty">{t('shell.conversations.empty')}</p>
</div>

<style>
	.rail {
		padding: var(--zl-space-4);
	}

	.rail__heading {
		margin: 0 0 var(--zl-space-3);
		font-size: var(--zl-font-size-sm);
		font-weight: var(--zl-font-weight-semibold);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--zl-text-muted);
	}

	.rail__empty {
		margin: 0;
		font-size: var(--zl-font-size-sm);
		color: var(--zl-text-subtle);
	}
</style>
