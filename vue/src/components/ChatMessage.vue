<template>
  <article class="message" :class="[role, statusClass]">
    <div class="avatar" aria-hidden="true">{{ role === 'user' ? '你' : 'AI' }}</div>
    <div class="message-body">
      <div class="message-head" v-if="role === 'agent'">
        <span class="speaker">Job Copilot</span>
        <span class="state-badge" :class="statusClass">{{ statusText }}</span>
      </div>

      <div class="bubble">
        <template v-if="role === 'agent'">
          <div v-if="showPlaceholder" class="stream-placeholder">
            <span class="pulse-dot"></span>
            <span>{{ activity?.latestStatus || '处理中' }}</span>
          </div>
          <div v-if="content" class="md-content" v-html="renderedContent"></div>
        </template>
        <template v-else>{{ content }}</template>
      </div>

      <section v-if="role === 'agent' && hasActivity" class="activity-panel">
        <div class="activity-status" :class="statusClass">
          <span class="activity-line"></span>
          <span>{{ activity.latestStatus || statusText }}</span>
        </div>

        <div v-if="activity?.todos?.length" class="activity-block">
          <h4>当前任务</h4>
          <ul class="todo-list">
            <li v-for="item in activity.todos" :key="`${item.content}-${item.status}`" class="todo-item">
              <span class="tag" :class="todoTagClass(item.status)">{{ todoStatusText(item.status) }}</span>
              <span>{{ item.content }}</span>
            </li>
          </ul>
        </div>

        <div v-if="activity?.subagents?.length" class="activity-block">
          <h4>协作 Agent</h4>
          <div class="meta-list">
            <span v-for="item in activity.subagents" :key="item" class="tag">{{ item }}</span>
          </div>
        </div>

        <div v-if="activity?.toolDetails?.length || activity?.tools?.length" class="activity-block">
          <h4>工具进度</h4>
          <div v-if="activity?.toolDetails?.length" class="tool-detail-list">
            <div v-for="item in activity.toolDetails" :key="`${item.name}-${item.status || 'unknown'}`" class="tool-detail-item">
              <span class="tool-name">{{ item.display_name || item.name }}</span>
              <span v-if="item.category" class="tag tag-gray">{{ categoryText(item.category) }}</span>
              <span v-if="item.requires_network" class="tag tag-gray">联网</span>
              <span v-if="item.status" class="tag" :class="item.status === 'completed' ? 'tag-success' : 'tag-warning'">
                {{ item.status === 'completed' ? '完成' : '进行中' }}
              </span>
            </div>
          </div>
          <div v-else class="meta-list">
            <span v-for="item in activity.tools" :key="item" class="tag tag-gray">{{ item }}</span>
          </div>
        </div>

        <div v-if="activity?.errorMessage" class="activity-error">
          {{ activity.errorMessage }}
        </div>
      </section>

      <section v-if="role === 'agent' && hasTrace" class="trace-panel">
        <details>
          <summary>
            <span>运行追踪</span>
            <span class="trace-summary">{{ traceSummary }}</span>
          </summary>

          <div class="trace-metrics">
            <div class="trace-metric">
              <span>耗时</span>
              <strong>{{ formatLatency(trace.duration_ms || meta?.latency_ms || 0) }}</strong>
            </div>
            <div class="trace-metric">
              <span>工具</span>
              <strong>{{ trace.metrics?.tool_call_count || 0 }}</strong>
            </div>
            <div class="trace-metric">
              <span>联网</span>
              <strong>{{ trace.metrics?.network_tool_call_count || 0 }}</strong>
            </div>
            <div class="trace-metric" :class="{ danger: trace.metrics?.failed_tool_call_count }">
              <span>失败</span>
              <strong>{{ trace.metrics?.failed_tool_call_count || 0 }}</strong>
            </div>
          </div>

          <ol v-if="traceTimeline.length" class="trace-timeline">
            <li v-for="(item, index) in traceTimeline" :key="traceItemKey(item, index)" class="trace-item" :class="traceItemClass(item)">
              <span class="trace-node"></span>
              <div class="trace-item-body">
                <div class="trace-item-head">
                  <strong>{{ traceItemLabel(item) }}</strong>
                  <span class="tag tag-gray">{{ traceTypeText(item.type) }}</span>
                  <span v-if="item.requires_network" class="tag tag-gray">联网</span>
                  <span v-if="item.status" class="tag" :class="traceStatusClass(item.status)">{{ traceStatusText(item.status) }}</span>
                </div>
                <div class="trace-item-meta">
                  <span v-if="item.category">{{ categoryText(item.category) }}</span>
                  <span v-if="item.source_name">{{ item.source_name }}</span>
                  <span v-if="item.latency_ms">{{ formatLatency(item.latency_ms) }}</span>
                  <span v-if="item.error" class="trace-error">{{ item.error }}</span>
                </div>
              </div>
            </li>
          </ol>
        </details>
      </section>

      <div v-if="role === 'agent' && meta && hasMeta" class="meta-list compact">
        <span v-if="meta.latency_ms" class="tag tag-gray">{{ formatLatency(meta.latency_ms) }}</span>
        <span v-for="s in meta.used_subagents" :key="s" class="tag">{{ s }}</span>
        <template v-if="meta.tool_calls?.length">
          <span v-for="item in meta.tool_calls" :key="`${item.name}-${item.status || 'meta'}`" class="tag tag-gray">
            {{ toolMetaText(item) }}
          </span>
        </template>
        <template v-else>
          <span v-for="t in meta.tool_calls_summary" :key="t" class="tag tag-gray">{{ t }}</span>
        </template>
        <a v-for="src in meta.sources" :key="src" :href="src" target="_blank" rel="noreferrer" class="tag tag-gray source-link">来源</a>
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, required: true },
  meta: { type: Object, default: null },
  status: { type: String, default: 'done' },
  activity: { type: Object, default: null },
})

marked.setOptions({ breaks: true })

const renderedContent = computed(() => marked.parse(props.content))
const showPlaceholder = computed(() => props.role === 'agent' && props.status === 'streaming' && !props.content)
const statusClass = computed(() => props.status || 'done')
const trace = computed(() => props.meta?.trace || props.activity?.trace || {})
const traceTimeline = computed(() => (Array.isArray(trace.value?.timeline) ? trace.value.timeline : []))
const hasTrace = computed(() => {
  const value = trace.value
  return Boolean(value?.trace_id || value?.run_id || traceTimeline.value.length || value?.metrics?.tool_call_count)
})
const traceSummary = computed(() => {
  const metrics = trace.value?.metrics || {}
  const pieces = []
  if (trace.value?.duration_ms || props.meta?.latency_ms) {
    pieces.push(formatLatency(trace.value.duration_ms || props.meta.latency_ms))
  }
  pieces.push(`${metrics.tool_call_count || 0} 工具`)
  if (metrics.network_tool_call_count) pieces.push(`${metrics.network_tool_call_count} 联网`)
  if (metrics.failed_tool_call_count) pieces.push(`${metrics.failed_tool_call_count} 失败`)
  return pieces.join(' · ')
})
const statusText = computed(() => {
  if (props.status === 'streaming') return '思考中'
  if (props.status === 'error') return '失败'
  if (props.status === 'stopped') return '已停止'
  return '已完成'
})
const hasActivity = computed(() => {
  const activity = props.activity
  if (!activity) return false
  return Boolean(
    activity.latestStatus ||
      activity.errorMessage ||
      activity.todos?.length ||
      activity.subagents?.length ||
      activity.toolDetails?.length ||
      activity.tools?.length,
  )
})
const hasMeta = computed(() => {
  const meta = props.meta
  if (!meta) return false
  return Boolean(
    meta.latency_ms ||
      meta.used_subagents?.length ||
      meta.tool_calls?.length ||
      meta.tool_calls_summary?.length ||
      meta.sources?.length,
  )
})

function todoStatusText(status) {
  if (status === 'completed') return '已完成'
  if (status === 'in_progress') return '进行中'
  return '待处理'
}

function todoTagClass(status) {
  if (status === 'completed') return 'tag-success'
  if (status === 'in_progress') return 'tag-warning'
  return 'tag-gray'
}

function categoryText(category) {
  if (category === 'job_db') return '岗位库'
  if (category === 'web_search') return '联网搜索'
  if (category === 'web_extract') return '网页抽取'
  if (category === 'external_mcp') return '外部 MCP'
  return '其他'
}

function toolMetaText(item) {
  const segments = [item.display_name || item.name]
  if (item.category) segments.push(categoryText(item.category))
  if (item.requires_network) segments.push('联网')
  return segments.filter(Boolean).join(' · ')
}

function formatLatency(ms) {
  if (!ms) return '0ms'
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  return `${ms}ms`
}

function traceTypeText(type) {
  if (type === 'run') return '运行'
  if (type === 'subagent') return 'Agent'
  if (type === 'tool') return '工具'
  return '事件'
}

function traceStatusText(status) {
  if (status === 'completed') return '完成'
  if (status === 'started') return '开始'
  if (status === 'error') return '失败'
  return status
}

function traceStatusClass(status) {
  if (status === 'completed') return 'tag-success'
  if (status === 'error') return 'tag-warning'
  return 'tag-gray'
}

function traceItemLabel(item) {
  return item.label || item.name || traceTypeText(item.type)
}

function traceItemClass(item) {
  return {
    error: item.status === 'error' || item.error,
    tool: item.type === 'tool',
    subagent: item.type === 'subagent',
  }
}

function traceItemKey(item, index) {
  return `${item.type || 'event'}-${item.name || item.label || index}-${index}`
}
</script>

<style scoped>
.message {
  display: flex;
  gap: 12px;
  width: 100%;
  margin: 0 0 18px;
  align-items: flex-start;
}

.message.agent {
  justify-content: flex-start;
  padding-right: clamp(160px, 22vw, 460px);
}

.message.user {
  flex-direction: row-reverse;
  justify-content: flex-start;
  padding-left: clamp(220px, 34vw, 720px);
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: #eef8f3;
  color: #167452;
  font-size: 12px;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: inset 0 0 0 1px rgba(22, 116, 82, 0.1);
}

.message.user .avatar {
  background: var(--primary);
  color: #fff;
  box-shadow: none;
}

.message-body {
  max-width: min(1040px, 100%);
  min-width: 0;
}

.message.user .message-body {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  max-width: min(560px, 100%);
}

.message-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.speaker {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
}

.state-badge {
  min-height: 22px;
  border-radius: 7px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
  background: #eef8f3;
  color: #167452;
}

.state-badge.streaming {
  background: #fff4df;
  color: #935f00;
}

.state-badge.error {
  background: #fff0ee;
  color: #b42318;
}

.state-badge.stopped {
  background: #f4f4f5;
  color: #71717a;
}

.bubble {
  padding: 13px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.68;
  word-break: break-word;
  box-shadow: 0 12px 28px rgba(31, 41, 55, 0.07);
}

.message.agent .bubble {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(28, 37, 65, 0.08);
}

.message.user .bubble {
  background: linear-gradient(135deg, #4f7ef8, #426de8);
  color: #fff;
  text-align: left;
}

.activity-panel {
  margin-top: 9px;
  padding: 12px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(28, 37, 65, 0.08);
}

.trace-panel {
  margin-top: 9px;
  border: 1px solid rgba(28, 37, 65, 0.08);
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.78);
}

.trace-panel details {
  padding: 0;
}

.trace-panel summary {
  min-height: 42px;
  padding: 10px 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--text);
  font-size: 13px;
  font-weight: 750;
}

.trace-summary {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 650;
  text-align: right;
}

.trace-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  padding: 0 12px 12px;
}

.trace-metric {
  min-width: 0;
  padding: 9px 10px;
  border-radius: 8px;
  background: #fff;
  border: 1px solid rgba(28, 37, 65, 0.06);
}

.trace-metric span {
  display: block;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 700;
}

.trace-metric strong {
  display: block;
  margin-top: 2px;
  color: var(--text);
  font-size: 14px;
  line-height: 1.2;
}

.trace-metric.danger strong {
  color: #b42318;
}

.trace-timeline {
  list-style: none;
  padding: 0 12px 12px;
}

.trace-item {
  position: relative;
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr);
  gap: 8px;
  padding: 8px 0;
}

.trace-item::before {
  content: '';
  position: absolute;
  left: 7px;
  top: 24px;
  bottom: -10px;
  width: 1px;
  background: #dbe3ef;
}

.trace-item:last-child::before {
  display: none;
}

.trace-node {
  width: 9px;
  height: 9px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 0 0 4px rgba(79, 126, 248, 0.12);
}

.trace-item.tool .trace-node {
  background: var(--accent);
  box-shadow: 0 0 0 4px rgba(26, 169, 130, 0.12);
}

.trace-item.error .trace-node {
  background: var(--danger);
  box-shadow: 0 0 0 4px rgba(217, 45, 32, 0.12);
}

.trace-item-body {
  min-width: 0;
}

.trace-item-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
}

.trace-item-head strong {
  font-size: 13px;
  line-height: 1.4;
}

.trace-item-meta {
  margin-top: 3px;
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  color: var(--text-secondary);
  font-size: 12px;
}

.trace-error {
  color: #b42318;
}

.activity-status {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 650;
}

.activity-status.error {
  color: #b42318;
}

.activity-status.stopped {
  color: #71717a;
}

.activity-line {
  width: 3px;
  height: 18px;
  border-radius: 999px;
  background: var(--primary);
}

.activity-status.streaming .activity-line {
  background: var(--warning);
}

.activity-status.error .activity-line {
  background: #d92d20;
}

.activity-block {
  margin-top: 12px;
}

.activity-block h4 {
  margin-bottom: 7px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 750;
}

.todo-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.todo-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
}

.tool-detail-list {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.tool-detail-item {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.tool-name {
  font-size: 13px;
  font-weight: 700;
}

.meta-list {
  margin-top: 7px;
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.meta-list.compact {
  margin-top: 7px;
}

.source-link {
  text-decoration: none;
}

.activity-error {
  margin-top: 10px;
  color: #b42318;
  font-size: 13px;
  line-height: 1.5;
}

.stream-placeholder {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  min-height: 24px;
  font-weight: 650;
}

.pulse-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--warning);
  animation: pulse 1.15s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(0.72); opacity: 0.5; }
  50% { transform: scale(1); opacity: 1; }
}

@media (max-width: 640px) {
  .message.agent,
  .message.user {
    padding-left: 0;
    padding-right: 0;
  }

  .message-body {
    max-width: calc(100vw - 84px);
  }

  .trace-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
