<script lang="ts">
	import { theme, type ThemePreference } from '$lib/theme.svelte';
	import { t } from '$lib/i18n';

	const options: { value: ThemePreference; labelKey: 'theme.light' | 'theme.dark' | 'theme.system' }[] = [
		{ value: 'system', labelKey: 'theme.system' },
		{ value: 'light', labelKey: 'theme.light' },
		{ value: 'dark', labelKey: 'theme.dark' }
	];
</script>

<div class="toggle" role="group" aria-label={t('theme.toggle.label')}>
	{#each options as option (option.value)}
		<button
			type="button"
			class="toggle__btn"
			class:toggle__btn--active={theme.preference === option.value}
			aria-pressed={theme.preference === option.value}
			onclick={() => theme.set(option.value)}
		>
			{t(option.labelKey)}
		</button>
	{/each}
</div>

<style>
	.toggle {
		display: inline-flex;
		gap: var(--zl-space-1);
		padding: var(--zl-space-1);
		background: var(--zl-bg-inset);
		border: 1px solid var(--zl-border);
		border-radius: var(--zl-radius-md);
	}

	.toggle__btn {
		padding: var(--zl-space-1) var(--zl-space-3);
		font-size: var(--zl-font-size-xs);
		border-radius: var(--zl-radius-sm);
		color: var(--zl-text-muted);
		transition:
			background var(--zl-duration-fast) var(--zl-easing-standard),
			color var(--zl-duration-fast) var(--zl-easing-standard);
	}

	.toggle__btn--active {
		background: var(--zl-surface-raised);
		color: var(--zl-text);
		font-weight: var(--zl-font-weight-medium);
		box-shadow: var(--zl-shadow-sm);
	}
</style>
