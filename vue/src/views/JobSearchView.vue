<template>
  <div class="page">
    <h2 class="page-title">岗位检索</h2>

    <div class="card search-form">
      <div class="form-row">
        <div class="form-group">
          <label>关键词 *</label>
          <input v-model="params.query" placeholder="如：数据分析师 Python SQL" @keydown.enter="search" />
        </div>
        <div class="form-group">
          <label>城市</label>
          <input v-model="params.city" placeholder="如：北京" />
        </div>
        <div class="form-group">
          <label>行业</label>
          <input v-model="params.industry" placeholder="如：银行" />
        </div>
        <div class="form-group">
          <label>学历要求</label>
          <select v-model="params.education">
            <option value="">不限</option>
            <option value="本科">本科</option>
            <option value="硕士">硕士</option>
            <option value="博士">博士</option>
            <option value="大专">大专</option>
          </select>
        </div>
        <div class="form-group">
          <label>返回数量</label>
          <select v-model.number="params.top_k">
            <option :value="5">5</option>
            <option :value="10">10</option>
            <option :value="15">15</option>
            <option :value="20">20</option>
          </select>
        </div>
      </div>
      <button class="btn btn-primary" @click="search" :disabled="loading || !params.query.trim()">
        <span v-if="loading" class="spinner"></span>
        <span v-else>搜索</span>
      </button>
    </div>

    <div v-if="error" class="error-msg">{{ error }}</div>

    <div v-if="results.length > 0">
      <p class="result-count">共找到 {{ results.length }} 个相关岗位</p>
      <JobCard v-for="job in results" :key="job.chunk_id" :job="job" />
    </div>

    <div v-else-if="searched && !loading" class="empty-result">
      未找到相关岗位，请尝试其他关键词。
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import JobCard from '../components/JobCard.vue'
import { searchJobs } from '../api/index.js'

const params = reactive({
  query: '',
  city: '',
  industry: '',
  education: '',
  top_k: 5,
})

const results = ref([])
const loading = ref(false)
const error = ref('')
const searched = ref(false)

async function search() {
  if (!params.query.trim()) return
  loading.value = true
  error.value = ''
  searched.value = false

  const p = { query: params.query, top_k: params.top_k }
  if (params.city) p.city = params.city
  if (params.industry) p.industry = params.industry
  if (params.education) p.education = params.education

  try {
    const res = await searchJobs(p)
    results.value = res.data.results
  } catch (err) {
    error.value = err.response?.data?.detail || err.message || '请求失败'
    results.value = []
  } finally {
    loading.value = false
    searched.value = true
  }
}
</script>

<style scoped>
.search-form {
  margin-bottom: 20px;
}
.form-row {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}
@media (max-width: 800px) {
  .form-row {
    grid-template-columns: 1fr 1fr;
  }
}
@media (max-width: 500px) {
  .form-row {
    grid-template-columns: 1fr;
  }
}
.result-count {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 12px;
}
.error-msg {
  color: #ef4444;
  margin-bottom: 12px;
  font-size: 14px;
}
.empty-result {
  text-align: center;
  color: var(--text-secondary);
  padding: 40px;
  font-size: 15px;
}
</style>
