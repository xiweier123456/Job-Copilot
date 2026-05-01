<template>
  <div class="card job-card">
    <div class="job-header">
      <span class="job-title">{{ job.job_title }}</span>
      <span class="salary" v-if="job.min_salary || job.max_salary">
        {{ formatSalary(job.min_salary, job.max_salary) }}
      </span>
    </div>
    <div class="job-meta">
      <span class="tag tag-gray">{{ job.company }}</span>
      <span class="tag tag-gray">{{ job.city }}</span>
      <span class="tag tag-gray">{{ job.industry }}</span>
      <span class="tag">{{ job.education }}</span>
      <span class="tag">{{ job.experience }}</span>
      <span v-if="job.score" class="tag tag-success">匹配度 {{ (job.score * 100).toFixed(1) }}%</span>
    </div>
    <div class="job-text">{{ job.text }}</div>
    <div class="job-footer">
      <span class="publish-date">发布于 {{ job.publish_date }}</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  job: { type: Object, required: true },
})

function formatSalary(min, max) {
  if (min && max) return `${(min / 1000).toFixed(0)}-${(max / 1000).toFixed(0)}K/月`
  if (min) return `${(min / 1000).toFixed(0)}K+/月`
  if (max) return `最高 ${(max / 1000).toFixed(0)}K/月`
  return ''
}
</script>

<style scoped>
.job-card {
  margin-bottom: 14px;
  transition: box-shadow 0.2s;
}
.job-card:hover {
  box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.job-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
  gap: 12px;
}
.job-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
}
.salary {
  font-size: 15px;
  font-weight: 600;
  color: #e04040;
  flex-shrink: 0;
}
.job-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 10px;
}
.job-text {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.65;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.job-footer {
  margin-top: 10px;
  font-size: 12px;
  color: #9ca3af;
}
</style>
