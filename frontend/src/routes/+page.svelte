<script lang="ts">
  import QueryInput from "$lib/components/QueryInput.svelte";
  import ThinkingProcess from "$lib/components/ThinkingProcess.svelte";
  import ProductCard from "$lib/components/ProductCard.svelte";
  import SimilarItems from "$lib/components/SimilarItems.svelte";
  import ErrorDisplay from "$lib/components/ErrorDisplay.svelte";
  import EmptyState from "$lib/components/EmptyState.svelte";
  import { api, APIError } from "$lib/services/api";
  import type { QueryResponse } from "$lib/types/api";
  import { Package } from "lucide-svelte";

  let query = $state("");
  let loading = $state(false);
  let error = $state<string | null>(null);
  let result = $state<QueryResponse | null>(null);

  async function handleSearch(prompt: string) {
    loading = true;
    error = null;
    result = null;

    try {
      result = await api.query({ prompt, similar_items_limit: 20 });
    } catch (e) {
      if (e instanceof APIError) {
        error =
          e.status === 0
            ? "Unable to connect to the server. Make sure the backend is running."
            : e.message;
      } else {
        error = "An unexpected error occurred. Please try again.";
      }
    } finally {
      loading = false;
    }
  }

  function retry() {
    if (query.trim()) {
      handleSearch(query.trim());
    }
  }

  const usedFallback = $derived(
    result &&
      result.matches.length > 0 &&
      Object.keys(result.extracted_entities).length === 0
  );
</script>

<div class="w-full space-y-8">
  <!-- Hero Section -->
  <section class="text-center py-8">
    <h2 class="text-3xl font-bold mb-3">Find Products with Natural Language</h2>
    <p class="text-muted-foreground max-w-2xl mx-auto">
      Describe what you're looking for and our AI will find matching products
      from the graph database
    </p>
  </section>

  <!-- Search Input -->
  <section class="w-full">
    <QueryInput bind:value={query} {loading} onsubmit={handleSearch} />
  </section>

  <!-- Error Display -->
  {#if error}
    <ErrorDisplay message={error} onretry={retry} />
  {/if}

  <!-- Loading State -->
  {#if loading}
    <ThinkingProcess entities={{}} matchCount={0} loading={true} />
  {/if}

  <!-- Results -->
  {#if result && !loading}
    <!-- Thinking Process -->
    <ThinkingProcess
      entities={result.extracted_entities}
      matchCount={result.matches.length}
      usedFallback={usedFallback ?? false}
    />

    <!-- Main Matches -->
    {#if result.matches.length > 0}
      <section class="w-full">
        <div class="flex items-center gap-2 mb-4">
          <Package class="h-5 w-5 text-primary" />
          <h2 class="text-lg font-semibold">Matching Products</h2>
          <span class="text-sm text-muted-foreground"
            >({result.matches.length})</span
          >
        </div>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {#each result.matches as product}
            <ProductCard {product} />
          {/each}
        </div>
      </section>

      <!-- Similar Items -->
      <SimilarItems items={result.similar_items} />
    {:else}
      <EmptyState query={result.query} />
    {/if}
  {/if}

  <!-- Initial Empty State -->
  {#if !result && !loading && !error}
    <EmptyState />
  {/if}
</div>
