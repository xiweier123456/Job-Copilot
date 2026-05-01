import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 1200000,
})

function parseSSEBlock(block) {
  const normalizedBlock = block.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const lines = normalizedBlock.split('\n')
  let event = 'message'
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (!dataLines.length) {
    return null
  }

  const raw = dataLines.join('\n')
  return {
    event,
    data: JSON.parse(raw),
  }
}

function buildStreamRequestInit(payload, options = {}) {
  const isFormData = payload instanceof FormData

  if (isFormData) {
    return {
      endpoint: '/api/chat/stream/upload',
      init: {
        method: 'POST',
        headers: {
          Accept: 'text/event-stream',
        },
        body: payload,
        signal: options.signal,
      },
    }
  }

  return {
    endpoint: '/api/chat/stream',
    init: {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal: options.signal,
    },
  }
}

async function consumeSSEStream(response, handlers = {}) {
  if (!response.ok) {
    throw new Error(`请求失败：${response.status}`)
  }

  if (!response.body) {
    throw new Error('当前浏览器不支持流式响应')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  const start = performance.now()

  console.log('[stream] opened', response.status, response.headers.get('content-type'))
  handlers.onOpen?.(response)

  while (true) {
    const { value, done } = await reader.read()
    console.log('[stream] reader.read', {
      done,
      byteLength: value?.byteLength || 0,
      elapsedMs: Math.round(performance.now() - start),
    })
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const normalizedBuffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
    const blocks = normalizedBuffer.split('\n\n')
    buffer = blocks.pop() || ''

    for (const block of blocks) {
      const parsed = parseSSEBlock(block.trim())
      if (!parsed) continue
      console.log('[stream] parsed event', parsed.event, parsed.data?.type, Math.round(performance.now() - start))
      handlers.onEvent?.(parsed)
    }
  }

  const tail = decoder.decode()
  if (tail) {
    buffer += tail
  }

  if (buffer.trim()) {
    const parsed = parseSSEBlock(buffer.trim())
    if (parsed) {
      console.log('[stream] parsed tail event', parsed.event, parsed.data?.type, Math.round(performance.now() - start))
      handlers.onEvent?.(parsed)
    }
  }

  console.log('[stream] complete', Math.round(performance.now() - start))
  handlers.onComplete?.()
}

export async function streamChatWithAgent(payload, handlers = {}, options = {}) {
  const { endpoint, init } = buildStreamRequestInit(payload, options)
  const response = await fetch(endpoint, init)
  return consumeSSEStream(response, handlers)
}

export async function stopChatRun(payload) {
  return fetch('/api/chat/stop', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  })
}

export function clearChatSession(payload) {
  return http.post('/chat/clear', payload)
}

export function getChatHistory(sessionId) {
  return http.get('/chat/history', {
    params: { session_id: sessionId },
  })
}

export function getChatSessions() {
  return http.get('/chat/sessions')
}

export function deleteChatSession(sessionId) {
  return http.delete(`/chat/sessions/${encodeURIComponent(sessionId)}`)
}

export function getChatModelOptions() {
  return http.get('/chat/model-options')
}

export function searchJobs(params) {
  return http.get('/jobs/search', { params })
}

export function matchResume(payload) {
  return http.post('/resume/match', payload)
}

export function checkHealth() {
  return http.get('/health')
}
