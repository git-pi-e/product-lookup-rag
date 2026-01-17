<script lang="ts">
  import "../app.css";
  import { theme } from "$lib/stores/theme";
  import { onMount } from "svelte";
  import { Button } from "$lib/components/ui/button";
  import { Moon, Sun, Database } from "lucide-svelte";

  let { children } = $props();

  onMount(() => {
    theme.init();
  });
</script>

<svelte:head>
  <title>Product Lookup RAG</title>
  <meta
    name="description"
    content="Search products using natural language with AI-powered recommendations"
  />
</svelte:head>

<div class="min-h-screen flex flex-col bg-background text-foreground">
  <!-- Header -->
  <header
    class="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur-sm"
  >
    <div
      class="w-full max-w-5xl mx-auto flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8"
    >
      <a href="/" class="flex items-center gap-3">
        <div
          class="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground"
        >
          <Database class="h-5 w-5" />
        </div>
        <div>
          <h1 class="font-semibold leading-none">Product Lookup</h1>
          <p class="text-xs text-muted-foreground">RAG-powered search</p>
        </div>
      </a>
      <Button
        variant="ghost"
        size="icon"
        onclick={() => theme.toggle()}
        aria-label="Toggle theme"
      >
        <Sun
          class="h-5 w-5 rotate-0 scale-100 transition-transform dark:-rotate-90 dark:scale-0"
        />
        <Moon
          class="absolute h-5 w-5 rotate-90 scale-0 transition-transform dark:rotate-0 dark:scale-100"
        />
      </Button>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    {@render children()}
  </main>

  <!-- Footer -->
  <footer class="w-full border-t border-border">
    <div class="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <p class="text-center text-sm text-muted-foreground">
        Powered by Neo4j Graph Database & OpenAI
      </p>
    </div>
  </footer>
</div>
