<template>
  <div class="chat-layout">
    <aside class="profile-panel" :class="{ collapsed: sidebarCollapsed }">
      <div class="panel-rail" v-if="sidebarCollapsed">
        <button class="icon-button" type="button" title="展开档案" @click="sidebarCollapsed = false">›</button>
        <span class="rail-label">档案</span>
      </div>

      <template v-else>
        <header class="profile-head">
          <div>
            <p class="eyebrow">Job Copilot</p>
            <h1>求职档案</h1>
          </div>
          <button class="icon-button" type="button" title="收起档案" @click="sidebarCollapsed = true">‹</button>
        </header>

        <section class="conversation-section">
          <div class="section-head">
            <span>会话</span>
            <button class="mini-action" type="button" :disabled="loading" @click="newConversation">新建</button>
          </div>

          <div class="conversation-list">
            <div
              v-for="session in visibleSessions"
              :key="session.session_id"
              class="conversation-item"
              :class="{ active: session.session_id === sessionId }"
              role="button"
              tabindex="0"
              @click="selectConversation(session)"
              @keydown.enter.prevent="selectConversation(session)"
            >
              <div class="conversation-copy">
                <strong>{{ session.title || '未命名会话' }}</strong>
                <small>{{ session.draft ? '当前草稿' : formatSessionTime(session.updated_at) }} · {{ session.turn_count || 0 }} 轮</small>
              </div>
              <button
                v-if="!session.draft"
                class="conversation-delete"
                type="button"
                title="删除会话"
                @click="deleteConversation(session, $event)"
              >
                ×
              </button>
            </div>

            <div v-if="!visibleSessions.length" class="conversation-empty">
              {{ sessionsLoading ? '加载会话中' : '还没有历史会话' }}
            </div>
          </div>
        </section>

        <section class="model-strip">
          <label class="field compact">
            <span>Agent 模型</span>
            <select v-model="selectedModelProvider" :disabled="loading">
              <option
                v-for="option in modelOptions"
                :key="option.provider"
                :value="option.provider"
                :disabled="!option.configured"
              >
                {{ option.label }} · {{ option.model }}{{ option.configured ? '' : '（未配置）' }}
              </option>
            </select>
          </label>
        </section>

        <section class="memory-strip">
          <div><span class="memory-dot"></span><strong>长期记忆</strong></div>
          <span>{{ memoryLabel }}</span>
        </section>

        <section class="profile-score">
          <div class="score-ring" :style="{ '--score': profileScore }"><span>{{ profileScore }}</span></div>
          <div>
            <strong>上下文完整度</strong>
            <p>{{ profileHint }}</p>
          </div>
        </section>

        <div class="context-form">
          <label class="field">
            <span>目标城市</span>
            <input v-model="ctx.target_city" placeholder="上海 / 北京 / 远程" />
          </label>
          <label class="field">
            <span>岗位方向</span>
            <input v-model="ctx.job_direction" placeholder="数据分析 / Java 后端 / 产品" />
          </label>
          <label class="field">
            <span>个人背景</span>
            <textarea v-model="ctx.user_profile" rows="4" placeholder="学历、专业、经验、技能栈、求职偏好"></textarea>
          </label>
          <label class="field">
            <span>简历文本</span>
            <textarea v-model="ctx.resume_text" rows="6" placeholder="粘贴简历要点，聊天时会一起发送给后端"></textarea>
          </label>
        </div>

        <div class="side-actions">
          <button class="btn btn-soft" type="button" :disabled="loading" @click="loadSessions">刷新列表</button>
          <button class="btn btn-outline danger" type="button" @click="clearMessages">清空当前</button>
        </div>
      </template>
    </aside>

    <main class="chat-main">
      <header class="chat-topbar">
        <div>
          <p class="eyebrow">智能求职助手</p>
          <h2>{{ chatTitle }}</h2>
        </div>
        <div class="topbar-meta">
          <span class="status-pill" :class="loading ? 'live' : 'ready'">{{ loading ? '生成中' : '就绪' }}</span>
          <span class="session-pill">{{ selectedModelLabel }}</span>
          <span class="session-pill">{{ activeSessionTitle }}</span>
        </div>
      </header>

      <div class="context-chips" v-if="contextChips.length">
        <span v-for="chip in contextChips" :key="chip" class="context-chip">{{ chip }}</span>
      </div>

      <div class="messages" ref="messagesEl">
        <section v-if="messages.length === 0" class="empty-state">
          <div class="empty-copy">
            <p class="eyebrow">从一个具体目标开始</p>
            <h3>把求职问题拆成可执行的下一步。</h3>
            <p>我会结合你的档案、简历、岗位库、联网工具和这次会话的长期记忆来回答。</p>
          </div>
          <div class="prompt-grid">
            <button
              v-for="item in quickPrompts"
              :key="item.title"
              class="prompt-card"
              type="button"
              @click="applyPrompt(item.prompt)"
            >
              <span>{{ item.title }}</span>
              <small>{{ item.text }}</small>
            </button>
          </div>
        </section>

        <ChatMessage
          v-for="(msg, i) in messages"
          :key="i"
          :role="msg.role"
          :content="msg.content"
          :meta="msg.meta"
          :status="msg.status"
          :activity="msg.activity"
        />
      </div>

      <footer class="composer-panel">
        <div class="attachment-row">
          <label class="upload-trigger" :class="{ disabled: loading }">
            <input
              ref="resumeFileInputEl"
              class="hidden-file-input"
              type="file"
              accept=".pdf,application/pdf"
              :disabled="loading"
              @change="handleResumeFileChange"
            />
            <span class="upload-icon">PDF</span>
            <span>{{ selectedResumeFile ? '更换简历' : '上传简历' }}</span>
          </label>
          <div v-if="selectedResumeFile" class="upload-chip">
            <span class="upload-name" :title="selectedResumeFile.name">{{ selectedResumeFile.name }}</span>
            <button class="icon-button small" type="button" @click="removeSelectedResumeFile" :disabled="loading">×</button>
          </div>
          <span v-if="activeRunId" class="run-chip">Run {{ activeRunId.slice(0, 8) }}</span>
          <span class="history-state">{{ historyState }}</span>
        </div>

        <div class="input-bar">
          <textarea
            v-model="inputText"
            class="chat-input"
            placeholder="例如：帮我判断这份简历更适合哪些数据岗位，并给出修改优先级"
            rows="1"
            @keydown.enter.exact.prevent="send"
            @input="autoResize"
            ref="inputEl"
            :disabled="loading"
          ></textarea>
          <button class="send-button" type="button" @click="loading ? stopCurrentRun() : send()" :disabled="!loading && !canSend">
            {{ loading ? '停止' : '发送' }}
          </button>
        </div>
      </footer>
    </main>
  </div>
</template>

<script setup>
import { computed, ref, reactive, nextTick, watch, onMounted } from 'vue'
import ChatMessage from '../components/ChatMessage.vue'
import {
  clearChatSession,
  deleteChatSession,
  getChatHistory,
  getChatModelOptions,
  getChatSessions,
  stopChatRun,
  streamChatWithAgent,
} from '../api/index.js'

const CHAT_STORAGE_KEY = 'job-copilot-chat-state'

const messages = ref([])
const inputText = ref('')
const loading = ref(false)
const sessions = ref([])
const sessionsLoading = ref(false)
const sessionId = ref(createSessionId())
const modelOptions = ref([])
const selectedModelProvider = ref('minimax')
const sidebarCollapsed = ref(false)
const messagesEl = ref(null)
const inputEl = ref(null)
const resumeFileInputEl = ref(null)
const selectedResumeFile = ref(null)
const activeRunId = ref('')
const historyState = ref('历史已同步')
const canSend = computed(() => Boolean(inputText.value.trim()))
let activeAbortController = null
let activeAgentMessage = null

const ctx = reactive({
  target_city: '',
  job_direction: '',
  user_profile: '',
  resume_text: '',
})

const quickPrompts = [
  {
    title: '岗位定位',
    text: '根据背景找方向',
    prompt: '请根据我的背景和目标城市，推荐 3 个最值得优先投递的岗位方向，并说明原因。',
  },
  {
    title: '简历诊断',
    text: '找出修改优先级',
    prompt: '请从招聘方视角分析我的简历，列出最影响通过率的 5 个问题和修改示例。',
  },
  {
    title: '投递策略',
    text: '制定一周计划',
    prompt: '请为我制定一个 7 天求职推进计划，包括岗位搜索、简历优化和面试准备。',
  },
  {
    title: '面试准备',
    text: '生成追问清单',
    prompt: '请围绕我的目标岗位，生成一套高频面试题、回答要点和需要补强的知识点。',
  },
]

const profileScore = computed(() => {
  const fields = [ctx.target_city, ctx.job_direction, ctx.user_profile, ctx.resume_text]
  return Math.round((fields.filter((item) => item.trim()).length / fields.length) * 100)
})

const profileHint = computed(() => {
  if (profileScore.value >= 100) return '信息充足，回答会更贴近你的真实目标。'
  if (!ctx.job_direction.trim()) return '补充岗位方向后，建议会更聚焦。'
  if (!ctx.resume_text.trim()) return '加入简历文本后，可以做更细的匹配分析。'
  return '继续补充背景，长期记忆会沉淀更多偏好。'
})

const contextChips = computed(() => {
  const chips = []
  if (ctx.target_city.trim()) chips.push(`城市：${ctx.target_city.trim()}`)
  if (ctx.job_direction.trim()) chips.push(`方向：${ctx.job_direction.trim()}`)
  if (ctx.user_profile.trim()) chips.push('已提供背景')
  if (ctx.resume_text.trim() || selectedResumeFile.value) chips.push('已接入简历')
  if (messages.value.length) chips.push(`${Math.ceil(messages.value.length / 2)} 轮上下文`)
  return chips
})

const memoryLabel = computed(() => (messages.value.length ? '已沉淀会话' : '等待第一轮'))
const chatTitle = computed(() => ctx.job_direction.trim() || '职业选择、简历与面试')
const activeSession = computed(() => sessions.value.find((item) => item.session_id === sessionId.value) || null)
const activeSessionTitle = computed(() => activeSession.value?.title || (messages.value.length ? '当前会话' : '新会话'))
const selectedModelLabel = computed(() => {
  const option = modelOptions.value.find((item) => item.provider === selectedModelProvider.value)
  return option ? `${option.label} · ${option.model}` : selectedModelProvider.value
})
const visibleSessions = computed(() => {
  if (activeSession.value) return sessions.value
  return [
    {
      session_id: sessionId.value,
      title: messages.value[0]?.content || '新会话',
      turn_count: Math.ceil(messages.value.length / 2),
      updated_at: '',
      draft: true,
    },
    ...sessions.value,
  ]
})

function createSessionId() {
  const time = Date.now().toString(36)
  const random = Math.random().toString(36).slice(2, 8)
  return `chat_${time}_${random}`
}

function normalizeSessionSummary(session) {
  return {
    session_id: session?.session_id || createSessionId(),
    title: session?.title || '未命名会话',
    turn_count: Number(session?.turn_count || 0),
    message_count: Number(session?.message_count || 0),
    created_at: session?.created_at || '',
    updated_at: session?.updated_at || '',
  }
}

function normalizeModelOption(option) {
  return {
    provider: option?.provider || 'minimax',
    label: option?.label || option?.provider || 'Agent',
    model: option?.model || '',
    base_url: option?.base_url || '',
    configured: Boolean(option?.configured),
    default: Boolean(option?.default),
  }
}

function formatSessionTime(value) {
  if (!value) return '刚刚'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '刚刚'
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function createAgentPlaceholder() {
  return {
    role: 'agent',
    content: '',
    status: 'streaming',
    meta: {
      latency_ms: null,
      run_id: null,
      used_subagents: [],
      tool_calls_summary: [],
      tool_calls: [],
      sources: [],
      trace: {},
    },
    activity: {
      latestStatus: '正在理解你的问题',
      todos: [],
      subagents: [],
      tools: [],
      toolDetails: [],
      errorMessage: '',
    },
  }
}

function createPersistedMessage(message) {
  if (message?.role === 'agent') {
    const wasStreaming = message.status === 'streaming'
    const wasStopped = message.status === 'stopped'
    return reactive({
      role: 'agent',
      content: wasStreaming
        ? '上次回答在页面刷新或离开时中断了。'
        : message.content || (wasStopped ? '本次回答已停止。' : ''),
      status: wasStreaming ? 'error' : message.status || 'done',
      meta: {
        run_id: message.meta?.run_id ?? null,
        latency_ms: message.meta?.latency_ms ?? null,
        used_subagents: Array.isArray(message.meta?.used_subagents) ? message.meta.used_subagents : [],
        tool_calls_summary: Array.isArray(message.meta?.tool_calls_summary) ? message.meta.tool_calls_summary : [],
        tool_calls: Array.isArray(message.meta?.tool_calls) ? message.meta.tool_calls : [],
        sources: Array.isArray(message.meta?.sources) ? message.meta.sources : [],
        trace: message.meta?.trace || {},
      },
      activity: {
        latestStatus: wasStreaming ? '上次会话已中断' : message.activity?.latestStatus || (wasStopped ? '已停止生成' : ''),
        todos: Array.isArray(message.activity?.todos) ? message.activity.todos : [],
        subagents: Array.isArray(message.activity?.subagents) ? message.activity.subagents : [],
        tools: Array.isArray(message.activity?.tools) ? message.activity.tools : [],
        toolDetails: Array.isArray(message.activity?.toolDetails) ? message.activity.toolDetails : [],
        trace: message.activity?.trace || message.meta?.trace || {},
        errorMessage: wasStreaming ? '页面刷新、关闭或切换期间，中断了这次生成。' : message.activity?.errorMessage || '',
      },
    })
  }

  return {
    role: message?.role || 'user',
    content: message?.content || '',
  }
}

function normalizeServerMessage(message) {
  return createPersistedMessage(message)
}

async function loadSessions() {
  sessionsLoading.value = true
  try {
    const { data } = await getChatSessions()
    sessions.value = Array.isArray(data?.sessions)
      ? data.sessions.map((session) => normalizeSessionSummary(session))
      : []
  } catch (error) {
    console.warn('[chat-view] load sessions failed', error)
  } finally {
    sessionsLoading.value = false
  }
}

async function loadModelOptions() {
  try {
    const { data } = await getChatModelOptions()
    const options = Array.isArray(data?.models)
      ? data.models.map((option) => normalizeModelOption(option))
      : []
    modelOptions.value = options.length
      ? options
      : [{ provider: 'minimax', label: 'MiniMax Agent', model: 'MiniMax-M2.7', configured: true, default: true }]

    const savedOrCurrent = modelOptions.value.find((option) => option.provider === selectedModelProvider.value && option.configured)
    const defaultOption = modelOptions.value.find((option) => option.default && option.configured)
    const firstConfigured = modelOptions.value.find((option) => option.configured)
    selectedModelProvider.value = (savedOrCurrent || defaultOption || firstConfigured || modelOptions.value[0]).provider
  } catch (error) {
    console.warn('[chat-view] load model options failed', error)
    modelOptions.value = [{ provider: 'minimax', label: 'MiniMax Agent', model: 'MiniMax-M2.7', configured: true, default: true }]
  }
}

async function syncMessagesFromServer() {
  if (loading.value) return

  historyState.value = '同步中'
  try {
    const { data } = await getChatHistory(sessionId.value)
    const serverMessages = Array.isArray(data?.messages)
      ? data.messages.map((message) => normalizeServerMessage(message))
      : []

    messages.value = serverMessages
    historyState.value = serverMessages.length ? '历史已同步' : '暂无历史'
    await nextTick()
    scrollToBottom()
  } catch (error) {
    historyState.value = '历史同步失败'
    console.warn('[chat-view] load history failed', error)
  }
}

async function newConversation() {
  if (loading.value) {
    await stopCurrentRun()
  }

  sessionId.value = createSessionId()
  messages.value = []
  inputText.value = ''
  activeRunId.value = ''
  historyState.value = '新会话'
  removeSelectedResumeFile()
  await nextTick()
  inputEl.value?.focus()
}

async function selectConversation(session) {
  if (!session?.session_id || session.session_id === sessionId.value) return

  if (loading.value) {
    await stopCurrentRun()
  }

  sessionId.value = session.session_id
  await syncMessagesFromServer()
}

async function deleteConversation(session, event) {
  event?.stopPropagation()
  if (!session?.session_id) return

  if (!window.confirm(`删除「${session.title || '未命名会话'}」及其历史上下文？`)) {
    return
  }

  if (loading.value && session.session_id === sessionId.value) {
    await stopCurrentRun()
  }

  try {
    await deleteChatSession(session.session_id)
  } catch (error) {
    console.warn('[chat-view] delete session failed', error)
    return
  }

  sessions.value = sessions.value.filter((item) => item.session_id !== session.session_id)
  if (session.session_id === sessionId.value) {
    const nextSession = sessions.value[0]
    if (nextSession) {
      sessionId.value = nextSession.session_id
      await syncMessagesFromServer()
    } else {
      await newConversation()
    }
  }
}

function buildPayload(text) {
  return {
    message: text,
    session_id: sessionId.value,
    agent_model_provider: selectedModelProvider.value,
    user_profile: ctx.user_profile || null,
    target_city: ctx.target_city || null,
    job_direction: ctx.job_direction || null,
    resume_text: ctx.resume_text || null,
  }
}

function buildUploadFormData(text) {
  const formData = new FormData()
  formData.append('resume_file', selectedResumeFile.value)
  formData.append('message', text)
  formData.append('session_id', sessionId.value)
  formData.append('agent_model_provider', selectedModelProvider.value)
  formData.append('user_profile', ctx.user_profile || '')
  formData.append('target_city', ctx.target_city || '')
  formData.append('job_direction', ctx.job_direction || '')
  return formData
}

function resetResumeFileInput() {
  if (resumeFileInputEl.value) {
    resumeFileInputEl.value.value = ''
  }
}

function removeSelectedResumeFile() {
  selectedResumeFile.value = null
  resetResumeFileInput()
}

function handleResumeFileChange(event) {
  const file = event.target.files?.[0] || null
  if (!file) {
    removeSelectedResumeFile()
    return
  }

  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
  if (!isPdf) {
    removeSelectedResumeFile()
    window.alert('目前只支持上传 PDF 简历。')
    return
  }

  selectedResumeFile.value = file
}

function dedupePush(list, value) {
  if (!value || list.includes(value)) return
  list.push(value)
}

function currentTodoText(todos = []) {
  const activeTodo = todos.find((item) => item.status === 'in_progress')
  if (activeTodo?.content) return `正在处理：${activeTodo.content}`

  const pendingTodo = todos.find((item) => item.status === 'pending')
  if (pendingTodo?.content) return `即将处理：${pendingTodo.content}`

  return ''
}

function markStopped(target, message = '已停止生成') {
  if (!target) return
  target.status = 'stopped'
  target.activity.latestStatus = message
  target.activity.errorMessage = ''
  if (!target.content) {
    target.content = '本次回答已停止。'
  }
}

function updateStreamingPreview(target) {
  if (target.status !== 'streaming') return

  if (target.activity.errorMessage) {
    target.content = `出错了：${target.activity.errorMessage}`
    return
  }

  const todoText = currentTodoText(target.activity.todos)
  if (todoText) {
    target.content = todoText
    return
  }

  const latestTool = target.activity.toolDetails?.at(-1)?.display_name || target.activity.tools.at(-1)
  if (latestTool) {
    target.content = `工具进度：${latestTool}`
    return
  }

  const latestSubagent = target.activity.subagents.at(-1)
  if (latestSubagent) {
    target.content = `正在使用 ${latestSubagent} 处理你的问题…`
    return
  }

  target.content = target.activity.latestStatus || ''
}

function handleStreamEvent(target, event) {
  const payload = event?.data?.payload || {}
  const type = event?.data?.type || event?.event
  console.log('[chat-view] handleStreamEvent', type, payload)

  if (type === 'status') {
    target.activity.latestStatus = payload.message || payload.stage || '处理中'
    updateStreamingPreview(target)
    return
  }

  if (type === 'todo') {
    target.activity.todos = Array.isArray(payload.items) ? payload.items : []
    updateStreamingPreview(target)
    return
  }

  if (type === 'subagent') {
    const name = payload.name
    dedupePush(target.activity.subagents, name)
    dedupePush(target.meta.used_subagents, name)
    updateStreamingPreview(target)
    return
  }

  if (type === 'tool') {
    const displayName = payload.display_name || payload.name || ''
    const label = displayName ? `${displayName}${payload.status === 'completed' ? ' · 完成' : ' · 进行中'}` : ''
    dedupePush(target.activity.tools, label)
    if (payload.name && payload.status === 'completed') {
      dedupePush(target.meta.tool_calls_summary, payload.name)
    }
    if (payload.name) {
      const existingIndex = target.activity.toolDetails.findIndex((item) => item.name === payload.name)
      if (existingIndex >= 0) {
        target.activity.toolDetails[existingIndex] = payload
      } else {
        target.activity.toolDetails.push(payload)
      }
    }
    updateStreamingPreview(target)
    return
  }

  if (type === 'error') {
    target.status = 'error'
    target.activity.errorMessage = payload.message || '流式请求失败'
    target.activity.latestStatus = '处理失败'
    if (!target.content) {
      target.content = `出错了：${target.activity.errorMessage}`
    }
    updateStreamingPreview(target)
    return
  }

  if (type === 'stopped') {
    markStopped(target, payload.message || '已停止生成')
    return
  }

  if (type === 'final') {
    target.content = payload.reply || target.content || '已完成，但未返回内容。'
    target.meta = {
      run_id: payload.run_id || null,
      latency_ms: payload.latency_ms,
      used_subagents: payload.used_subagents || target.meta.used_subagents || [],
      tool_calls_summary: payload.tool_calls_summary || target.meta.tool_calls_summary || [],
      tool_calls: payload.tool_calls || target.meta.tool_calls || [],
      sources: payload.sources || [],
      trace: payload.trace || {},
    }
    target.activity.toolDetails = payload.tool_calls || target.activity.toolDetails || []
    target.activity.trace = payload.trace || {}
    target.status = payload.error ? 'error' : 'done'
    target.activity.latestStatus = payload.error ? '处理失败' : '回答已生成'
    if (payload.error) {
      target.activity.errorMessage = payload.error
    }
  }
}

function restoreState() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY)
    if (!raw) return

    const saved = JSON.parse(raw)
    sessionId.value = saved.sessionId || createSessionId()
    selectedModelProvider.value = saved.selectedModelProvider || 'minimax'
    inputText.value = saved.inputText || ''
    sidebarCollapsed.value = Boolean(saved.sidebarCollapsed)
    ctx.target_city = saved.ctx?.target_city || ''
    ctx.job_direction = saved.ctx?.job_direction || ''
    ctx.user_profile = saved.ctx?.user_profile || ''
    ctx.resume_text = saved.ctx?.resume_text || ''
    messages.value = Array.isArray(saved.messages)
      ? saved.messages.map((message) => createPersistedMessage(message))
      : []
  } catch (error) {
    console.warn('[chat-view] restore state failed', error)
  }
}

function persistState() {
  try {
    const snapshot = {
      sessionId: sessionId.value,
      selectedModelProvider: selectedModelProvider.value,
      inputText: inputText.value,
      sidebarCollapsed: sidebarCollapsed.value,
      ctx: {
        target_city: ctx.target_city,
        job_direction: ctx.job_direction,
        user_profile: ctx.user_profile,
        resume_text: ctx.resume_text,
      },
      messages: messages.value,
    }
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(snapshot))
  } catch (error) {
    console.warn('[chat-view] persist state failed', error)
  }
}

function applyPrompt(prompt) {
  inputText.value = prompt
  nextTick(() => inputEl.value?.focus())
}

restoreState()
loadModelOptions()
loadSessions()
syncMessagesFromServer()

onMounted(async () => {
  await nextTick()
  scrollToBottom()
})

watch(
  [
    messages,
    inputText,
    sessionId,
    selectedModelProvider,
    sidebarCollapsed,
    () => ctx.target_city,
    () => ctx.job_direction,
    () => ctx.user_profile,
    () => ctx.resume_text,
  ],
  persistState,
  { deep: true },
)

async function stopCurrentRun() {
  const runId = activeRunId.value
  const target = activeAgentMessage

  activeAbortController?.abort()
  activeAbortController = null
  activeRunId.value = ''
  activeAgentMessage = null
  loading.value = false
  markStopped(target)

  if (runId) {
    try {
      await stopChatRun({ run_id: runId })
    } catch (error) {
      console.warn('[chat-view] stop run failed', error)
    }
  }

  await nextTick()
  scrollToBottom()
}

async function send() {
  const text = inputText.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  const agentMessage = reactive(createAgentPlaceholder())
  messages.value.push(agentMessage)
  inputText.value = ''
  loading.value = true
  historyState.value = '等待保存'
  activeAbortController = new AbortController()
  activeRunId.value = ''
  activeAgentMessage = agentMessage
  await nextTick()
  scrollToBottom()

  try {
    const requestPayload = selectedResumeFile.value ? buildUploadFormData(text) : buildPayload(text)

    await streamChatWithAgent(
      requestPayload,
      {
        onEvent: async (event) => {
          if (!activeRunId.value && event?.data?.run_id) {
            activeRunId.value = event.data.run_id
          }
          handleStreamEvent(agentMessage, event)
          await nextTick()
          scrollToBottom()
        },
      },
      { signal: activeAbortController.signal },
    )

    if (agentMessage.status === 'streaming') {
      agentMessage.status = 'done'
      agentMessage.activity.latestStatus = '回答已生成'
    }
    historyState.value = '历史已保存'
    await loadSessions()
  } catch (err) {
    if (err?.name === 'AbortError') {
      markStopped(agentMessage)
      historyState.value = '生成已停止'
    } else {
      const msg = err.message || '请求失败，请检查后端是否启动'
      agentMessage.status = 'error'
      agentMessage.activity.latestStatus = '处理失败'
      agentMessage.activity.errorMessage = msg
      agentMessage.content = agentMessage.content || `出错了：${msg}`
      historyState.value = '请求失败'
    }
  } finally {
    loading.value = false
    activeAbortController = null
    activeRunId.value = ''
    activeAgentMessage = null
    await nextTick()
    scrollToBottom()
  }
}

function scrollToBottom() {
  const scroll = () => {
    const el = messagesEl.value
    if (!el) return
    el.scrollTop = el.scrollHeight
  }

  scroll()
  requestAnimationFrame(() => {
    scroll()
    requestAnimationFrame(scroll)
  })
}

async function clearMessages() {
  const currentSessionId = sessionId.value

  if (loading.value) {
    await stopCurrentRun()
  }

  try {
    await clearChatSession({ session_id: currentSessionId })
  } catch (error) {
    console.warn('[chat-view] clear session failed', error)
  }

  messages.value = []
  inputText.value = ''
  sessionId.value = createSessionId()
  sessions.value = sessions.value.filter((item) => item.session_id !== currentSessionId)
  historyState.value = '新会话'
  removeSelectedResumeFile()
  await loadSessions()
}

function autoResize(e) {
  const el = e.target
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 168) + 'px'
}
</script>

<style scoped>
.chat-layout {
  display: flex;
  height: calc(100vh - 56px);
  background:
    radial-gradient(circle at top left, rgba(79, 126, 248, 0.12), transparent 32rem),
    linear-gradient(135deg, #f7f8fc 0%, #eef4f1 100%);
  overflow: hidden;
}

.profile-panel {
  width: 332px;
  flex-shrink: 0;
  border-right: 1px solid rgba(28, 37, 65, 0.08);
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(18px);
  display: flex;
  flex-direction: column;
  transition: width 0.22s ease;
  overflow: hidden;
}

.profile-panel.collapsed {
  width: 54px;
}

.panel-rail {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  padding: 14px 8px;
}

.rail-label {
  writing-mode: vertical-rl;
  color: var(--text-secondary);
  font-size: 12px;
  letter-spacing: 0;
}

.profile-head,
.chat-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.profile-head {
  padding: 22px 20px 16px;
}

.eyebrow {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.profile-head h1,
.chat-topbar h2 {
  margin-top: 2px;
  font-size: 22px;
  line-height: 1.25;
}

.icon-button {
  width: 32px;
  height: 32px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text-secondary);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 0.18s ease;
}

.icon-button:hover:not(:disabled) {
  color: var(--text);
  border-color: rgba(79, 126, 248, 0.35);
  box-shadow: 0 8px 20px rgba(31, 41, 55, 0.08);
}

.icon-button.small {
  width: 24px;
  height: 24px;
  font-size: 16px;
}

.memory-strip,
.conversation-section,
.model-strip,
.profile-score,
.context-form,
.side-actions {
  margin: 0 18px 14px;
}

.conversation-section {
  min-height: 0;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 700;
}

.mini-action {
  min-height: 28px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--primary);
  font-size: 12px;
  font-weight: 750;
  padding: 4px 10px;
  cursor: pointer;
}

.mini-action:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.conversation-list {
  display: grid;
  gap: 6px;
  max-height: 220px;
  overflow-y: auto;
  padding-right: 2px;
}

.conversation-item {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 28px;
  align-items: center;
  gap: 8px;
  min-height: 58px;
  padding: 8px 8px 8px 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--text);
  cursor: pointer;
  transition: background 0.18s ease, border-color 0.18s ease;
}

.conversation-item:hover,
.conversation-item.active {
  background: #f6f8fc;
  border-color: rgba(28, 37, 65, 0.08);
}

.conversation-copy {
  min-width: 0;
}

.conversation-copy strong,
.conversation-copy small {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-copy strong {
  font-size: 13px;
  line-height: 1.35;
}

.conversation-copy small {
  margin-top: 4px;
  color: var(--text-muted);
  font-size: 12px;
}

.conversation-delete {
  width: 28px;
  height: 28px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.18s ease, background 0.18s ease, color 0.18s ease;
}

.conversation-item:hover .conversation-delete,
.conversation-item.active .conversation-delete {
  opacity: 1;
}

.conversation-delete:hover {
  background: #fff1f0;
  color: #b42318;
}

.conversation-empty {
  min-height: 46px;
  display: grid;
  place-items: center;
  border: 1px dashed var(--border);
  border-radius: 8px;
  color: var(--text-muted);
  font-size: 13px;
}

.memory-strip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 8px;
  background: #f0f7f4;
  color: #196650;
  font-size: 13px;
}

.memory-strip div {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.memory-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #1aa982;
  box-shadow: 0 0 0 4px rgba(26, 169, 130, 0.14);
}

.profile-score {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px;
  border-radius: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
}

.score-ring {
  --score: 0;
  width: 58px;
  height: 58px;
  border-radius: 50%;
  background: conic-gradient(#4f7ef8 calc(var(--score) * 1%), #dfe8fb 0);
  display: grid;
  place-items: center;
  flex-shrink: 0;
  position: relative;
}

.score-ring::after {
  content: '';
  position: absolute;
  inset: 7px;
  border-radius: 50%;
  background: white;
}

.score-ring span {
  position: relative;
  z-index: 1;
  font-weight: 800;
  color: var(--primary);
}

.profile-score strong {
  display: block;
  font-size: 14px;
}

.profile-score p {
  margin-top: 4px;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.context-form {
  flex: 1;
  overflow-y: auto;
  padding-right: 2px;
}

.field {
  display: block;
  margin-bottom: 13px;
}

.field span {
  display: block;
  margin-bottom: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 650;
}

.field.compact {
  margin-bottom: 0;
}

.field select {
  width: 100%;
  min-height: 38px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text);
  padding: 0 10px;
  font-size: 13px;
}

.side-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  padding-bottom: 18px;
}

.btn-soft {
  background: #edf5ff;
  color: var(--primary);
}

.btn-outline.danger {
  color: #b42318;
  border-color: #f4c7c3;
}

.chat-main {
  min-width: 0;
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-topbar {
  padding: 22px 28px 12px;
}

.topbar-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.status-pill,
.session-pill,
.run-chip,
.history-state,
.context-chip {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  border-radius: 8px;
  padding: 4px 10px;
  font-size: 12px;
  font-weight: 650;
  white-space: nowrap;
}

.status-pill.ready {
  background: #eef8f3;
  color: #167452;
}

.status-pill.live {
  background: #fff4df;
  color: #935f00;
}

.session-pill,
.run-chip,
.history-state,
.context-chip {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(28, 37, 65, 0.08);
  color: var(--text-secondary);
}

.context-chips {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 0 28px 12px;
}

.messages {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 8px 28px 24px;
}

.empty-state {
  max-width: 920px;
  margin: 42px auto 0;
}

.empty-copy {
  max-width: 650px;
  margin-bottom: 18px;
}

.empty-copy h3 {
  margin: 4px 0 8px;
  font-size: 30px;
  line-height: 1.18;
}

.empty-copy p:last-child {
  color: var(--text-secondary);
}

.prompt-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.prompt-card {
  min-height: 112px;
  text-align: left;
  border: 1px solid rgba(28, 37, 65, 0.08);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
  padding: 14px;
  cursor: pointer;
  color: var(--text);
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.prompt-card:hover {
  transform: translateY(-2px);
  border-color: rgba(79, 126, 248, 0.28);
  box-shadow: 0 16px 34px rgba(31, 41, 55, 0.09);
}

.prompt-card span {
  display: block;
  font-size: 15px;
  font-weight: 750;
}

.prompt-card small {
  display: block;
  margin-top: 8px;
  color: var(--text-secondary);
  line-height: 1.45;
}

.composer-panel {
  margin: 0 24px 20px;
  padding: 12px;
  border: 1px solid rgba(28, 37, 65, 0.1);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 18px 40px rgba(31, 41, 55, 0.1);
}

.attachment-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.upload-trigger {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  padding: 5px 10px 5px 6px;
  border: 1px dashed rgba(79, 126, 248, 0.38);
  border-radius: 8px;
  cursor: pointer;
  color: var(--text-secondary);
  background: #f7faff;
  font-size: 13px;
  font-weight: 650;
}

.upload-trigger.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.upload-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 22px;
  border-radius: 6px;
  background: #dfe9ff;
  color: var(--primary);
  font-size: 11px;
  font-weight: 800;
}

.hidden-file-input {
  display: none;
}

.upload-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  max-width: min(100%, 420px);
  min-height: 32px;
  padding: 4px 6px 4px 10px;
  border-radius: 8px;
  background: #f6f7fb;
  border: 1px solid var(--border);
}

.upload-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}

.history-state {
  margin-left: auto;
}

.input-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 84px;
  gap: 10px;
  align-items: end;
}

.chat-input {
  min-height: 46px;
  max-height: 168px;
  resize: none;
  border-radius: 8px;
  border-color: transparent;
  background: #f7f8fb;
  font-size: 14px;
  line-height: 1.55;
  overflow-y: auto;
}

.send-button {
  height: 46px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #4f7ef8, #1aa982);
  color: #fff;
  font-weight: 750;
  cursor: pointer;
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-1px);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

@media (max-width: 960px) {
  .chat-layout {
    flex-direction: column;
    overflow: auto;
  }

  .profile-panel,
  .profile-panel.collapsed {
    width: 100%;
    max-height: none;
  }

  .panel-rail {
    height: 54px;
    flex-direction: row;
  }

  .rail-label {
    writing-mode: horizontal-tb;
  }

  .prompt-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 640px) {
  .chat-topbar,
  .messages,
  .context-chips {
    padding-left: 16px;
    padding-right: 16px;
  }

  .empty-copy h3 {
    font-size: 24px;
  }

  .prompt-grid {
    grid-template-columns: 1fr;
  }

  .composer-panel {
    margin: 0 12px 12px;
  }

  .input-bar {
    grid-template-columns: 1fr;
  }

  .send-button {
    width: 100%;
  }
}
</style>
