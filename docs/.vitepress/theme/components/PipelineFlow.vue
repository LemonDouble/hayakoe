<script setup lang="ts">
defineProps<{
  steps: Array<{
    num: string
    title: string
    tool: string
    content: string | string[]
    chips?: string[]
    gpu?: string
  }>
}>()

function toLines(content: string | string[]): string[] {
  return Array.isArray(content) ? content : [content]
}

function gpuClass(gpu: string): string {
  if (gpu.includes('필수')) return 'gpu-required'
  if (gpu.includes('선택')) return 'gpu-optional'
  return 'gpu-none'
}
</script>

<template>
  <div class="pipeline-flow">
    <div class="pipeline-grid">
      <template v-for="(step, i) in steps" :key="step.num">
        <div class="step-card">
          <div class="step-head">
            <span class="step-num">{{ step.num }}</span>
            <div class="step-badges">
              <span v-if="step.tool" class="badge tool">{{ step.tool }}</span>
              <span v-if="step.gpu" class="badge gpu" :class="gpuClass(step.gpu)">GPU {{ step.gpu }}</span>
            </div>
          </div>
          <div class="step-title">{{ step.title }}</div>
          <div class="step-content">
            <p v-for="(line, li) in toLines(step.content)" :key="li">{{ line }}</p>
          </div>
          <div v-if="step.chips && step.chips.length" class="step-chips">
            <span v-for="chip in step.chips" :key="chip" class="chip">{{ chip }}</span>
          </div>
        </div>
        <div v-if="i < steps.length - 1" class="step-arrow" aria-hidden="true">→</div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.pipeline-flow {
  margin: 24px 0;
}

.pipeline-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  align-items: stretch;
}

.step-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 16px 18px;
  border-radius: 14px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
}

.step-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.step-num {
  font-family: 'Galmuri11', var(--vp-font-family-mono), monospace;
  font-size: 22px;
  line-height: 1;
  color: var(--vp-c-brand-1);
  flex-shrink: 0;
}

.step-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  justify-content: flex-end;
}

.badge {
  display: inline-flex;
  align-items: center;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 9999px;
  line-height: 1.6;
  white-space: nowrap;
}

.badge.tool {
  background: rgba(240, 185, 11, 0.15);
  color: var(--vp-c-brand-1);
  border: 1px solid rgba(240, 185, 11, 0.35);
}

.badge.gpu {
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  background: transparent;
}

.badge.gpu.gpu-required {
  border-color: rgba(255, 99, 99, 0.45);
  color: #ff8585;
  background: rgba(255, 99, 99, 0.08);
}

.badge.gpu.gpu-optional {
  border-color: var(--vp-c-divider);
  color: var(--vp-c-text-2);
}

.step-title {
  font-weight: 700;
  font-size: 15px;
  color: var(--vp-c-text-1);
  line-height: 1.35;
}

.step-content {
  font-size: 13px;
  line-height: 1.7;
  color: var(--vp-c-text-2);
}

.step-content p {
  margin: 0 0 8px;
}

.step-content p:last-child {
  margin-bottom: 0;
}

.step-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 2px;
}

.chip {
  font-size: 10.5px;
  line-height: 1.6;
  padding: 1px 7px;
  border-radius: 6px;
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
  white-space: nowrap;
}

.step-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--vp-c-brand-1);
  font-size: 22px;
  font-weight: 700;
  user-select: none;
  transform: rotate(90deg);
  padding: 4px 0;
}
</style>
