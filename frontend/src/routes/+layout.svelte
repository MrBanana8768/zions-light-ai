<script lang="ts">
	import '$lib/styles/global.css';
	import favicon from '$lib/assets/favicon.svg';
	import AppShell from '$lib/components/shell/AppShell.svelte';
	import type { Snippet } from 'svelte';

	let { children }: { children: Snippet } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<AppShell>
	{@render children()}
</AppShell>
