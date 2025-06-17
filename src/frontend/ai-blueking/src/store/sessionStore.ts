import { ref } from "vue"
import type { ISession, ISessionContent, ISessionPrompt } from "@blueking/ai-ui-sdk/types"
import type { ISessionEditItem } from "./types"
import { uuid as generateUuid } from "../utils"

interface IAgentInfo {
  conversationSettings?: {
    openingRemark?: string
    predefinedQuestions?: string[]
  }
  promptSetting?: {
    content?: ISessionPrompt[]
  }
}

// 为所有需要的 SDK 方法定义一个清晰的接口（Interface），以获得更好的类型提示和代码健壮性。
interface SdkApi {
  setCurrentSession: (session: ISession) => void
  getSessionContentsApi: (sessionCode: string) => Promise<ISessionContent[]>
  setSessionContents: (contents: ISessionContent[]) => void
  modifySessionApi: (session: ISession) => Promise<any>
  deleteSessionApi: (sessionCode: string) => Promise<any>
  setCurrentSessionChain: (session: ISession) => void
  getSessionsApi: () => Promise<ISession[]>
  getAgentInfoApi: () => Promise<IAgentInfo>
  plusSessionApi: (session: ISession) => Promise<any>
  handleCompleteRole: (sessionCode: string, prompts: ISessionPrompt[]) => Promise<any>
}

const sessionList = ref<ISessionEditItem[]>([])
const currentSession = ref<ISessionEditItem | null>(null)
const sessionContentLoading = ref<boolean>(false)
let sdkApi: Partial<SdkApi> = {}

// 存储会话的原始值
const originalSessionValues = ref<Record<string, ISessionEditItem>>({})

/**
 * 使用会话管理 Store
 */
export function useSessionStore() {
  /**
   * 设置会话列表
   * @param sessions 会话列表
   */
  const setSessionList = (sessions: ISession[]) => {
    sessionList.value = sessions.map((item) => ({ ...item, isEdit: false }))
  }

  /**
   * 添加会话
   * @param session 会话信息
   */
  const addSession = (session: ISession) => {
    const newSession = { ...session, isEdit: false }
    sessionList.value.push(newSession)
    return newSession
  }

  /**
   * 创建新会话
   * @returns Promise<ISessionEditItem> 新创建的会话
   */
  const addNewSession = async () => {
    // 生成新的会话代码
    const sessionCode = generateUuid()

    // 查找现有会话中以 Chat- 开头的最大索引
    const chatPattern = /^Chat-(\d+)$/
    let maxIndex = 0

    sessionList.value.forEach((session) => {
      if (chatPattern.test(session.sessionName)) {
        const match = session.sessionName.match(chatPattern)
        if (match && match[1]) {
          const index = parseInt(match[1], 10)
          if (!isNaN(index) && index > maxIndex) {
            maxIndex = index
          }
        }
      }
    })

    // 新会话索引为最大索引 + 1
    const newIndex = maxIndex + 1
    const sessionName = `Chat-${newIndex}`

    const newSession: ISession = {
      sessionCode,
      sessionName,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }

    // 添加到本地会话列表
    const session = addSession(newSession)

    // 创建 session 并同步到后台
    if (!sdkApi.plusSessionApi) {
      throw new Error('plusSessionApi not registered')
    }
    await sdkApi.plusSessionApi(session)

    // 使用 switchCurrentSession 方法同时更新前端和 SDK
    await switchCurrentSession(session)

    return session
  }

  /**
   * 开始编辑会话
   * @param sessionCode 会话代码
   */
  const startEditSession = (sessionCode: string) => {
    const session = sessionList.value.find(s => s.sessionCode === sessionCode)
    if (session) {
      // 保存原始值
      originalSessionValues.value[sessionCode] = { ...session }
      // 设置编辑状态
      updateSession(sessionCode, { isEdit: true }, { syncBackend: false })
    }
  }

  /**
   * 结束编辑会话
   * @param sessionCode 会话代码
   * @param updates 更新内容
   */
  const finishEditSession = async (sessionCode: string, updates: Partial<ISessionEditItem>) => {
    const originalSession = originalSessionValues.value[sessionCode]
    if (!originalSession) {
      return null
    }

    // 检查内容是否有实际变化（与原始值比较）
    const hasContentChanged = Object.entries(updates).some(([key, value]) => {
      if (key === 'isEdit') return false
      return originalSession[key as keyof ISessionEditItem] !== value
    })

    // 更新会话
    const result = await updateSession(
      sessionCode,
      { ...updates, isEdit: false },
      { syncBackend: hasContentChanged }
    )

    // 清理原始值
    delete originalSessionValues.value[sessionCode]

    return result
  }

  /**
   * 更新会话
   * @param sessionCode 会话代码
   * @param updates 部分更新内容
   * @param options 更新选项
   * @param options.syncBackend 是否同步到后端，默认为 true
   * @param options.forceSync 是否强制同步到后端（不比较内容变化），默认为 false
   */
  const updateSession = async (
    sessionCode: string,
    updates: Partial<ISessionEditItem>,
    options: {
      syncBackend?: boolean
      forceSync?: boolean
    } = {}
  ) => {
    const { syncBackend = true, forceSync = false } = options
    const index = sessionList.value.findIndex((s) => s.sessionCode === sessionCode)
    
    if (index === -1) {
      return null
    }

    const oldSession = sessionList.value[index]
    const updatedSession = { ...oldSession, ...updates }

    // 更新本地状态
    sessionList.value[index] = updatedSession

    // 如果更新的是当前会话，也更新 currentSession
    if (currentSession.value && currentSession.value.sessionCode === sessionCode) {
      currentSession.value = { ...currentSession.value, ...updates }
    }

    // 判断是否需要同步到后端
    if ((syncBackend && !updates.isEdit) || forceSync) {
      try {
        if (!sdkApi.modifySessionApi) {
          throw new Error('modifySessionApi not registered')
        }
        // 同步到后端时，确保 isEdit 为 false
        const sessionToSync = { ...updatedSession, isEdit: false }
        await sdkApi.modifySessionApi(sessionToSync)
      } catch (error) {
        console.error('Failed to sync session to backend:', error)
      }
    }

    return updatedSession
  }

  /**
   * 删除会话
   * @param sessionCode 会话代码
   * @returns 如果删除的是当前会话，返回下一个可用的会话；否则返回 null
   */
  const deleteSession = (sessionCode: string): ISessionEditItem | null => {
    if (sdkApi.deleteSessionApi) {
      sdkApi.deleteSessionApi(sessionCode)
    }
    const index = sessionList.value.findIndex((s) => s.sessionCode === sessionCode)
    if (index !== -1) {
      // 如果删除的是当前会话，找到一个可用的会话
      let nextSession: ISessionEditItem | null = null
      if (currentSession.value && currentSession.value.sessionCode === sessionCode) {
        // 获取除了要删除的会话之外的所有会话
        const availableSessions = sessionList.value.filter((s) => s.sessionCode !== sessionCode)

        // 如果有其他会话
        if (availableSessions.length > 0) {
          // 优先选择今天的会话，否则选择第一个会话
          const today = new Date().toDateString()
          const todaySessions = availableSessions.filter((s) => new Date(s.createdAt).toDateString() === today)

          nextSession = todaySessions.length > 0 ? todaySessions[0] : availableSessions[0]
          currentSession.value = nextSession

          // 如果注册了 SDK 的 setCurrentSession 方法，也更新 SDK 的当前会话
          if (nextSession && sdkApi.setCurrentSession) {
            sdkApi.setCurrentSession(nextSession)
          }
        } else {
          currentSession.value = null
        }
      }

      // 执行删除
      sessionList.value.splice(index, 1)
      return nextSession
    }
    return null
  }

  /**
   * 设置当前会话（仅前端状态）
   * @param session 会话信息
   */
  const setCurrentSession = (session: ISession | ISessionEditItem) => {
    // 查找会话是否已在列表中
    const existingSession = sessionList.value.find((s) => s.sessionCode === session.sessionCode)

    if (existingSession) {
      // 如果会话已存在，直接设置为当前会话
      currentSession.value = existingSession
    } else {
      // 如果会话不存在，添加到列表并设置为当前会话
      const newSession = addSession(session)
      currentSession.value = newSession
    }
  }

  /**
   * 注册 SDK 方法
   * @param methods 部分 SDK 方法
   */
  const registerSdkMethods = (methods: Partial<SdkApi>) => {
    sdkApi = methods
  }

  /**
   * 切换当前会话（同时更新前端和 SDK）
   * @param session 会话信息
   */
  const switchCurrentSession = async (session: ISession | ISessionEditItem) => {
    if (sdkApi.setCurrentSessionChain) {
      await sdkApi.setCurrentSessionChain(session)
      setCurrentSession(session)
    }
  }

  /**
   * 初始化会话
   * @returns Promise<{
   *   openingRemark: string
   *   predefinedQuestions: string[]
   * }>
   */
  const initSession = async () => {
    // 获取会话列表
    if (!sdkApi.getSessionsApi) {
      throw new Error('getSessionsApi not registered')
    }
    const sessions = await sdkApi.getSessionsApi()
    setSessionList(sessions)

    // 创建新会话
    const newSession = await addNewSession()

    // 获取会话设置
    if (!sdkApi.getAgentInfoApi) {
      throw new Error('getAgentInfoApi not registered')
    }
    const { conversationSettings, promptSetting } = await sdkApi.getAgentInfoApi()

    // 处理角色设置
    if (promptSetting?.content?.length && sdkApi.handleCompleteRole) {
      await sdkApi.handleCompleteRole(newSession.sessionCode, promptSetting.content)
    }

    return {
      openingRemark: conversationSettings?.openingRemark || '',
      predefinedQuestions: conversationSettings?.predefinedQuestions || [],
    }
  }

  return {
    sessionList,
    currentSession,
    setSessionList,
    addSession,
    addNewSession,
    initSession,
    updateSession,
    startEditSession,
    finishEditSession,
    deleteSession,
    setCurrentSession,
    registerSdkMethods,
    switchCurrentSession,
    sessionContentLoading,
  }
}

// 创建一个单例实例
export const sessionStore = useSessionStore()
