import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { loadEnv } from "vite"

const DEFAULT_ENV_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const RUNTIME_GLOBAL_KEYS = ["BK_AIDEV_URL", "BK_AIDEV_API_URL"]

/**
 * 仅本地 dev 注入 window.BK_*；生产由 createDocsMiddleware / server.cjs 注入。
 * Vite 默认只把 VITE_* 写入 process.env，这里用 loadEnv(..., 'BK_') 读取 docs/.env*。
 */
export function resolveDevRuntimeGlobalsScript({
  mode = process.env.NODE_ENV || "development",
  envDir = DEFAULT_ENV_DIR,
  processEnv = process.env,
} = {}) {
  if (mode === "production") {
    return ""
  }

  const fileEnv = loadEnv(mode, envDir, "BK_")
  return RUNTIME_GLOBAL_KEYS.map((key) => {
    const value = processEnv[key] || fileEnv[key]
    if (!value || value.includes("{{")) {
      return ""
    }
    return `window.${key}=${JSON.stringify(value)};`
  })
    .filter(Boolean)
    .join("")
}
