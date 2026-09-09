<script lang="ts">
	/**
	 * The empty message pane (FRONTEND_PLAN.md F1: "an empty message pane
	 * with a placeholder"). No streaming, no compactor calls, no store —
	 * that is F9/F8/F3, other lanes.
	 */
	import { t } from '$lib/i18n';
</script>

<div class="pane" role="region" aria-label={t('shell.message_pane.heading')}>
	<p class="pane__placeholder">{t('shell.message_pane.placeholder')}</p>
</div>

<style>
	.pane {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: var(--zl-space-6);
	}

	.pane__placeholder {
		max-width: 32rem;
		text-align: center;
		color: var(--zl-text-subtle);
		font-size: var(--zl-font-size-md);
		line-height: var(--zl-line-height-relaxed);
	}
</style>
