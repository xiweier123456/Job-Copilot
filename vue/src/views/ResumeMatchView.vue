<template>
  <div class="page">
    <h2 class="page-title">简历匹配分析</h2>

    <div class="card">
      <div class="form-row-2">
        <div class="form-group">
          <label>目标岗位 *</label>
          <input v-model="form.job_query" placeholder="如：数据分析师、算法工程师" />
        </div>
        <div class="form-group">
          <label>目标城市</label>
          <input v-model="form.city" placeholder="如：上海" />
        </div>
        <div class="form-group">
          <label>参考岗位数</label>
          <select v-model.number="form.top_k">
            <option :value="3">3</option>
            <option :value="5">5</option>
            <option :value="8">8</option>
            <option :value="10">10</option>
          </select>
        </div>
      </div>
      <div class="form-group">
        <label>简历文本 *</label>
        <textarea
          v-model="form.resume_text"
          rows="10"
          placeholder="粘贴你的简历文本，越详细越准确（支持纯文本格式）"
        ></textarea>
      </div>
      <button
        class="btn btn-primary"
        @click="analyze"
        :disabled="loading || !form.job_query.trim() || !form.resume_text.trim()"
      >
        <span v-if="loading" class="spinner"></span>
        <span v-else>开始分析</span>
      </button>
      <span v-if="loading" style="margin-left:12px;font-size:13px;color:var(--text-secondary)">分析中，可能需要 20-40 秒...</span>
    </div>

    <div v-if="error" class="error-msg">{{ error }}</div>

    <!-- Results -->
    <div v-if="result" class="results">
      <!-- Score -->
      <div class="card score-card">
        <div class="score-circle" :style="{ '--pct': result.match_score }">
          <span class="score-num">{{ result.match_score }}<small>分</small></span>
        </div>
        <div class="score-summary">
          <h3>整体评价</h3>
          <p>{{ result.summary }}</p>
        </div>
      </div>

      <!-- Skill gap -->
      <div class="skill-grid">
        <div class="card">
          <h3 class="section-title matched">匹配项</h3>
          <div class="tag-list">
            <span v-for="s in result.skill_gap.matched" :key="s" class="tag tag-success">{{ s }}</span>
            <span v-if="!result.skill_gap.matched.length" class="empty-list">暂无</span>
          </div>
        </div>
        <div class="card">
          <h3 class="section-title missing">欠缺项</h3>
          <div class="tag-list">
            <span v-for="s in result.skill_gap.missing" :key="s" class="tag tag-warning">{{ s }}</span>
            <span v-if="!result.skill_gap.missing.length" class="empty-list">无明显欠缺</span>
          </div>
        </div>
      </div>

      <!-- Suggestions -->
      <div class="card">
        <h3 class="section-title">改进建议</h3>
        <ol class="suggestions">
          <li v-for="s in result.skill_gap.suggestions" :key="s">{{ s }}</li>
          <li v-if="!result.skill_gap.suggestions.length" class="empty-list">暂无建议</li>
        </ol>
      </div>

      <!-- Reference jobs -->
      <div v-if="result.reference_jobs.length > 0" class="card">
        <h3 class="section-title">参考岗位</h3>
        <div class="ref-jobs">
          <div v-for="(job, i) in result.reference_jobs" :key="i" class="ref-job">
            <span class="ref-title">{{ job.job_title }}</span>
            <span class="tag tag-gray">{{ job.company }}</span>
            <span class="tag tag-gray">{{ job.city }}</span>
            <span class="tag">{{ job.education }}</span>
            <span class="tag">{{ job.experience }}</span>
            <span v-if="job.min_salary" class="salary">{{ formatSalary(job.min_salary, job.max_salary) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { matchResume } from '../api/index.js'

const form = reactive({
  resume_text: '',
  job_query: '',
  city: '',
  top_k: 3,
})

const result = ref(null)
const loading = ref(false)
const error = ref('')

async function analyze() {
  loading.value = true
  error.value = ''
  result.value = null
  try {
    const payload = {
      resume_text: form.resume_text,
      job_query: form.job_query,
      top_k: form.top_k,
    }
    if (form.city) payload.city = form.city
    const res = await matchResume(payload)
    result.value = res.data
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '请求失败'
  } finally {
    loading.value = false
  }
}

function formatSalary(min, max) {
  if (min && max) return `${(min / 1000).toFixed(0)}-${(max / 1000).toFixed(0)}K`
  if (min) return `${(min / 1000).toFixed(0)}K+`
  return ''
}
</script>

<style scoped>
.form-row-2 {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}
@media (max-width: 600px) {
  .form-row-2 { grid-template-columns: 1fr; }
}
.results {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.error-msg {
  color: #ef4444;
  margin-top: 12px;
  font-size: 14px;
}

/* Score card */
.score-card {
  display: flex;
  gap: 24px;
  align-items: center;
}
.score-circle {
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background: conic-gradient(var(--primary) calc(var(--pct) * 1%), var(--primary-light) 0);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  position: relative;
}
.score-circle::before {
  content: '';
  position: absolute;
  inset: 8px;
  background: var(--surface);
  border-radius: 50%;
}
.score-num {
  position: relative;
  font-size: 22px;
  font-weight: 700;
  color: var(--primary);
  z-index: 1;
}
.score-num small {
  font-size: 13px;
  font-weight: 400;
}
.score-summary h3 {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 6px;
}
.score-summary p {
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.65;
}

/* Skill grid */
.skill-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 600px) {
  .skill-grid { grid-template-columns: 1fr; }
}
.section-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
}
.section-title.matched { color: #16a34a; }
.section-title.missing { color: #d97706; }
.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.empty-list {
  font-size: 13px;
  color: var(--text-secondary);
}

/* Suggestions */
.suggestions {
  padding-left: 18px;
  font-size: 14px;
  line-height: 2;
  color: var(--text);
}

/* Reference jobs */
.ref-jobs {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ref-job {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
}
.ref-job:last-child { border-bottom: none; }
.ref-title {
  font-weight: 600;
  font-size: 14px;
  margin-right: 4px;
}
.salary {
  font-size: 13px;
  font-weight: 600;
  color: #e04040;
}
</style>
