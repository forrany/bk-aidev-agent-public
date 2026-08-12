const path = require('path');

// workspace 包（@blueking/chat-x / ai-blueking）位于独立 pnpm store，
// 常解析到另一份 vue（如 3.5.32），而本项目为 3.5.x 另一版本。
// webpack 按 issuer 向上查找 node_modules 时会打进双 Vue 实例；
// useTemplateRef 跨实例 defineProperty 会报
// "Cannot define property xxx, object is not extensible"。
const vuePkgPath = require.resolve('vue/package.json', { paths: [__dirname] });
const vueDir = path.dirname(vuePkgPath);

module.exports = {
  port: process.env.BK_APP_PORT,
  host: process.env.BK_APP_HOST,
  typescript: true,
  replaceStatic: true,
  parseNodeModules: false,
  server: 'http',
  configureWebpack: {
    resolve: {
      alias: {
        vue: vueDir,
        // 覆盖 cli-service 默认的相对路径 vue$ alias，强制落到同一份 Vue
        'vue$': path.join(vueDir, 'dist/vue.esm-bundler.js'),
      },
    },
  },
};
