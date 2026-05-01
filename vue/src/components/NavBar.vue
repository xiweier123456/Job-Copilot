<template>
  <nav class="navbar">
    <div class="navbar-inner">
      <span class="navbar-brand">Job Copilot</span>
      <div class="navbar-links">
        <router-link to="/" class="nav-link" active-class="active" exact>
          对话
        </router-link>
        <router-link to="/jobs" class="nav-link" active-class="active">
          岗位检索
        </router-link>
        <router-link to="/resume" class="nav-link" active-class="active">
          简历匹配
        </router-link>
      </div>
      <div class="health-dot" :class="healthStatus" :title="healthTitle"></div>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { checkHealth } from '../api/index.js'

const healthStatus = ref('unknown')
const healthTitle = ref('检查中...')

onMounted(async () => {
  try {
    const res = await checkHealth()
    healthStatus.value = res.data.status === 'ok' ? 'ok' : 'degraded'
    healthTitle.value = res.data.status === 'ok' ? '后端连接正常' : '后端部分异常'
  } catch {
    healthStatus.value = 'error'
    healthTitle.value = '后端无法连接'
  }
})
</script>

<style scoped>
.navbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  position: sticky;
  top: 0;
  z-index: 100;
}
.navbar-inner {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 16px;
  height: 56px;
  display: flex;
  align-items: center;
  gap: 32px;
}
.navbar-brand {
  font-size: 18px;
  font-weight: 700;
  color: var(--primary);
  flex-shrink: 0;
}
.navbar-links {
  display: flex;
  gap: 4px;
  flex: 1;
}
.nav-link {
  padding: 6px 14px;
  border-radius: 8px;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-secondary);
  transition: all 0.2s;
}
.nav-link:hover {
  background: var(--bg);
  color: var(--text);
}
.nav-link.active {
  background: var(--primary-light);
  color: var(--primary);
}
.health-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
  cursor: help;
}
.health-dot.ok { background: #22c55e; }
.health-dot.degraded { background: #f59e0b; }
.health-dot.error { background: #ef4444; }
.health-dot.unknown { background: #d1d5db; }
</style>
