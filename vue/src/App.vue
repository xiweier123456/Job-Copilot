<template>
  <NavBar />
  <router-view v-slot="{ Component }">
    <keep-alive>
      <component :is="Component" />
    </keep-alive>
  </router-view>
</template>

<script setup>
import NavBar from './components/NavBar.vue'
</script>
