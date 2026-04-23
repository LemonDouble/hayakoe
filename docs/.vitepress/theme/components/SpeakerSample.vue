<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'

defineProps<{
  name: string
  src: string
  badge?: string
  badgeIcon?: string
}>()

const audioEl = ref<HTMLAudioElement | null>(null)
const progressEl = ref<HTMLDivElement | null>(null)
const playing = ref(false)
const currentTime = ref(0)
const duration = ref(0)

let rafId = 0

function tick() {
  const el = audioEl.value
  if (!el) return
  currentTime.value = el.currentTime
  rafId = requestAnimationFrame(tick)
}

function startTicking() {
  if (rafId) return
  rafId = requestAnimationFrame(tick)
}

function stopTicking() {
  if (!rafId) return
  cancelAnimationFrame(rafId)
  rafId = 0
}

function toggle() {
  const el = audioEl.value
  if (!el) return
  if (el.paused) el.play()
  else el.pause()
}

function onPlay() {
  playing.value = true
  startTicking()
}
function onPause() {
  playing.value = false
  stopTicking()
  currentTime.value = audioEl.value?.currentTime ?? 0
}
function onEnd() {
  playing.value = false
  stopTicking()
  currentTime.value = 0
}
function onSeeked() {
  currentTime.value = audioEl.value?.currentTime ?? 0
}
function onLoaded() {
  duration.value = audioEl.value?.duration ?? 0
}

onBeforeUnmount(stopTicking)

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
  <div class="speaker-sample">
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
      <div class="label">
        <span v-if="badge" class="badge">{{ badge }}</span>
        {{ name }}
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
      :src="src"
      preload="metadata"
      @play="onPlay"
      @pause="onPause"
      @ended="onEnd"
      @seeked="onSeeked"
      @loadedmetadata="onLoaded"
    />
  </div>
</template>

<style scoped>
.speaker-sample {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  margin: 10px 0;
  transition: border-color 0.2s ease, background 0.2s ease;
}

@media (max-width: 640px) {
  .speaker-sample {
    gap: 10px;
    padding: 12px 14px;
  }
}

.speaker-sample:hover {
  border-color: rgba(240, 185, 11, 0.25);
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

.label {
  font-family: 'Galmuri11', var(--vp-font-family-mono), monospace;
  font-weight: 700;
  font-size: 14px;
  color: var(--vp-c-text-1);
  line-height: 1.4;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

@media (max-width: 640px) {
  .label {
    font-size: 13px;
  }
}

.avatar {
  flex-shrink: 0;
  width: 54px;
  height: 54px;
  border-radius: 9999px;
  object-fit: cover;
  border: 2px solid rgba(240, 185, 11, 0.3);
}

@media (max-width: 640px) {
  .avatar {
    width: 40px;
    height: 40px;
  }
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
}

.time {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
  min-width: 60px;
  text-align: right;
}
</style>
