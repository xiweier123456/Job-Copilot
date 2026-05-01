import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import JobSearchView from '../views/JobSearchView.vue'
import ResumeMatchView from '../views/ResumeMatchView.vue'

const routes = [
  { path: '/', component: ChatView },
  { path: '/jobs', component: JobSearchView },
  { path: '/resume', component: ResumeMatchView },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
