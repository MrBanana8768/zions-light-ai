import adapter from '@sveltejs/adapter-node';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},

			// adapter-node, not adapter-auto: the server component is a hard
			// requirement (FRONTEND_PLAN.md §2.3, §3.4) — it holds the compactor
			// URL / bearer key and is the ONLY auth boundary in the system today
			// (no auth exists on the compactor's own /v1/* routes). This must
			// build to a real Node server we run under supervisord, not whatever
			// adapter-auto happens to detect.
			adapter: adapter()
		})
	]
});
