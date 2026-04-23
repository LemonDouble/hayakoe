<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'

interface Sample {
  value: string
  caption?: string
  src: string
}

const props = withDefaults(
  defineProps<{
    label: string
    samples: Sample[]
    defaultIndex?: number
    badge?: string
    badgeIcon?: string
  }>(),
  {
    defaultIndex: 0,
  },
)

const selectedIndex = ref(props.defaultIndex)
const audioEl = ref<HTMLAudioElement | null>(null)
const progressEl = ref<HTMLDivElement | null>(null)
const playing = ref(false)
const currentTime = ref(0)
const duration = ref(0)

const current = computed(() => props.samples[selectedIndex.value])

function toggle() {
  const el = audioEl.value
  if (!el) return
  if (el.paused) el.play()
  else el.pause()
}

async function select(i: number) {
  if (i === selectedIndex.value) return
  const wasPlaying = playing.value
  selectedIndex.value = i
  currentTime.value = 0
  duration.value = 0
  await nextTick()
  const el = audioEl.value
  if (el && wasPlaying) {
    el.play().catch(() => {})
  }
}

function onPlay() {
  playing.value = true
}
function onPause() {
  playing.value = false
}
function onEnd() {
  playing.value = false
  currentTime.value = 0
}
function onTimeUpdate() {
  currentTime.value = audioEl.value?.currentTime ?? 0
}
function onLoaded() {
  duration.value = audioEl.value?.duration ?? 0
}

function fmt(t: number): string {
  if (!isFinite(t) || t < 0) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function seek(e: MouseEvent) {
  const bar = progressEl.value
  const audio = audioEl.value
  if (!bar || !audio || !duration.value) return
  const rect = bar.getBoundingClientRect()
  const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
  audio.currentTime = ratio * duration.value
}
</script>

<template>
  <div class="speaker-sample-group">
    <div class="player">
      <img v-if="badgeIcon" :src="badgeIcon" class="avatar" alt="" />
      <button
        class="play-btn"
        type="button"
        :aria-label="playing ? 'Pause' : 'Play'"
        @click="toggle"
      >
        <svg v-if="!playing" viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
          <path d="M8 5v14l11-7z" />
        </svg>
        <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
          <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
        </svg>
      </button>
      <div class="info">
        <div class="label-row">
          <span v-if="badge" class="badge">{{ badge }}</span>
          <span class="param">{{ label }} = {{ current.value }}</span>
          <span v-if="current.caption" class="caption">{{ current.caption }}</span>
        </div>
        <div class="bar-row">
          <div
            ref="progressEl"
            class="progress"
            role="slider"
            :aria-valuenow="currentTime"
            :aria-valuemin="0"
            :aria-valuemax="duration"
            @click="seek"
          >
            <div
              class="fill"
              :style="{
                width: duration ? `${(currentTime / duration) * 100}%` : '0%',
              }"
            />
          </div>
          <div class="time">{{ fmt(currentTime) }} / {{ fmt(duration) }}</div>
        </div>
      </div>
      <audio
        ref="audioEl"
        :src="current.src"
        preload="metadata"
        @play="onPlay"
        @pause="onPause"
        @ended="onEnd"
        @timeupdate="onTimeUpdate"
        @loadedmetadata="onLoaded"
      />
    </div>
    <div class="chip-row" role="tablist" :aria-label="`${label} 값 선택`">
      <button
        v-for="(s, i) in samples"
        :key="i"
        type="button"
        role="tab"
        class="chip"
        :class="{ active: i === selectedIndex }"
        :aria-selected="i === selectedIndex"
        @click="select(i)"
      >
        {{ s.value }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.speaker-sample-group {
  margin: 16px 0;
  padding: 14px 18px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  transition: border-color 0.2s ease;
}

.speaker-sample-group:hover {
  border-color: rgba(240, 185, 11, 0.25);
}

.player {
  display: flex;
  align-items: center;
  gap: 12px;
}

.avatar {
  flex-shrink: 0;
  width: 54px;
  height: 54px;
  border-radius: 9999px;
  object-fit: cover;
  border: 2px solid rgba(240, 185, 11, 0.3);
}

.badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 9999px;
  background: rgba(240, 185, 11, 0.15);
  color: var(--vp-c-brand-1);
  white-space: nowrap;
}

@media (max-width: 640px) {
  .avatar {
    width: 40px;
    height: 40px;
  }
}

.play-btn {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 9999px;
  border: none;
  background: var(--vp-c-brand-1);
  color: #12100e;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s ease, transform 0.15s ease;
}

.play-btn:hover {
  background: var(--vp-c-brand-2);
}

.play-btn:active {
  transform: scale(0.96);
}

.play-btn:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.label-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
  min-height: 1.2em;
}

.param {
  font-family: 'Galmuri11', var(--vp-font-family-mono), monospace;
  font-weight: 700;
  font-size: 14px;
  color: var(--vp-c-text-1);
  line-height: 1.2;
}

.caption {
  font-size: 12px;
  color: var(--vp-c-text-2);
  line-height: 1.2;
}

.bar-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress {
  flex: 1;
  height: 4px;
  border-radius: 9999px;
  background: var(--vp-c-default-1);
  cursor: pointer;
  overflow: hidden;
  position: relative;
}

.fill {
  height: 100%;
  background: var(--vp-c-brand-1);
  transition: width 0.08s linear;
}

.time {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
  min-width: 60px;
  text-align: right;
}

.chip-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed var(--vp-c-divider);
}

.chip {
  padding: 4px 12px;
  border-radius: 9999px;
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-family: 'Galmuri11', var(--vp-font-family-mono), monospace;
  font-size: 12px;
  font-weight: 600;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.chip:hover {
  border-color: rgba(240, 185, 11, 0.5);
  color: var(--vp-c-text-1);
}

.chip.active {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #12100e;
}

.chip:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
</style>
