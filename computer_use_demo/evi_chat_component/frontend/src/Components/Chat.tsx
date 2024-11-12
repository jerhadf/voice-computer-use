import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { ReactNode, useEffect } from "react"
import { VoiceProvider, useVoice, VoiceReadyState } from "@humeai/voice-react"

type ComponentValue = {
  events: ChatEvent[]
  is_muted: boolean
  is_connected: boolean
}

// Typed wrapper around Streamlit.setComponentValue
const setComponentValue = (value: ComponentValue) => {
  Streamlit.setComponentValue(value)
}

type UseVoiceReturn = ReturnType<typeof useVoice>
type Command =
  | {
    type:
    | "mute"
    | "unmute"
    | "pauseAssistant"
    | "resumeAssistant"
    | "muteAudio"
    | "unmuteAudio"
    | "connect"
    | "disconnect"
    | "clearAudioQueue"
  }
  | {
    type: "sendUserInput"
    message: Parameters<UseVoiceReturn["sendUserInput"]>[0]
  }
  | {
    type: "sendAssistantInput"
    message: Parameters<UseVoiceReturn["sendAssistantInput"]>[0]
  }
  | {
    type: "sendSessionSettings"
    message: Parameters<UseVoiceReturn["sendSessionSettings"]>[0]
  }
  | {
    type: "sendToolMessage"
    message: Parameters<UseVoiceReturn["sendToolMessage"]>[0]
  }

type InteractiveChatProps = {
  commands: Command[]
  events: ChatEvent[]
  listen_to: Array<Listenable>
  debug: boolean
}

const InteractiveChat = (props: InteractiveChatProps) => {
  const { commands, events, listen_to, debug } = props
  const {
    connect,
    disconnect,
    mute,
    unmute,
    muteAudio,
    unmuteAudio,
    sendUserInput,
    sendAssistantInput,
    sendSessionSettings,
    sendToolMessage,
    pauseAssistant,
    resumeAssistant,

    isMuted,
    readyState,
  } = useVoice()

  const [eventCursor, setEventCursor] = React.useState(0)

  // Make sure to disconnect the chat when the component unmounts.
  useEffect(() => {
    return () => {
      console.log('Unmounted')
      disconnect()
    }
  }, [])

  useEffect(() => {
    const newEvents = events.slice(eventCursor, events.length)
    if (newEvents.length === 0) {
      return
    }
    if (newEvents.some((e) => listenedTo(listen_to, e))) {
      setComponentValue({
        events: events,
        is_muted: isMuted,
        is_connected: readyState === VoiceReadyState.OPEN,
      })
      setEventCursor(events.length)
    } else {
      console.log(
        `None of the new events ${newEvents.map((e) => e.type)} were being listened to.`
      )
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events])

  useEffect(() => {
    console.log("Voice ready state changed to", readyState)
    setComponentValue({
      events: events,
      is_muted: isMuted,
      is_connected: readyState === VoiceReadyState.OPEN,
    })
  }, [isMuted, readyState])

  const dispatchCommand = (command: Command) => {
    console.log(`Dispatching command ${command.type} ...`)
    switch (command.type) {
      case "connect":
        connect()
        return
      case "disconnect":
        disconnect()
        return
      case "mute":
        mute()
        return
      case "unmute":
        unmute()
        return
      case "pauseAssistant":
        console.log("Paused assistant")
        pauseAssistant()
        return
      case "resumeAssistant":
        resumeAssistant()
        return
      case "muteAudio":
        muteAudio()
        return
      case "unmuteAudio":
        unmuteAudio()
        return
      case "sendUserInput":
        sendUserInput(command.message)
        return
      case "sendAssistantInput":
        sendAssistantInput(command.message)
        return
      case "sendSessionSettings":
        sendSessionSettings(command.message)
        return
      case "sendToolMessage":
        sendToolMessage(command.message)
        return
      case "clearAudioQueue":
        // This triggers the audio queue to clear but is kind of hacky.
        sendUserInput("")
        return
    }
  }

  const [commandCursor, setCommandCursor] = React.useState(0)
  useEffect(() => {
    console.log("commands updated", commands)
    if (commandCursor > commands.length) {
      console.error("Unexpected: cursor is greater than commands length")
    }
    const newCommands = commands.slice(commandCursor, commands.length)
    newCommands.forEach((command) => dispatchCommand(command))
    setCommandCursor(commands.length)
    console.log(
      `commands updated. There were ${newCommands.length} new commands. The cursor is now at ${commands.length}`
    )
  }, [commands])
  if (debug) {
    return (
      <pre>
        {JSON.stringify(
          { commandCursor, commands: commands.map((c, i) => [i, c]), events },
          null,
          2
        )}
      </pre>
    )
  }
  return <></>
}
type Listenable =
  | `message.${Parameters<NonNullable<VoiceProviderParam["onMessage"]>>[0]["type"]}`
  | ChatEvent["type"]
type StreamlitArgs = InteractiveChatProps & {
  hume_api_key: string
  config_id: string
}
const defaultListenTo: Array<Listenable> = [
  "message.user_message",
  "message.chat_metadata",
  "opened",
  "closed",
  "error",
]
const listenedTo = (
  listen_to: Array<Listenable> = defaultListenTo,
  event: ChatEvent
) => {
  const slug =
    event.type === "message" ? `message.${event.message.type}` : event.type
  console.log("event type was", slug)
  console.log("listen_to was", listen_to.join(", "))
  if (event.type === "message") {
    return listen_to.includes(`message.${event.message.type}`)
  }
  return listen_to.includes(event.type)
}

type VoiceProviderParam = Parameters<typeof VoiceProvider>[0]
type ChatEvent =
  | {
    type: "message"
    message: Parameters<NonNullable<VoiceProviderParam["onMessage"]>>[0]
  }
  | {
    type: "closed"
  }
  | {
    type: "opened"
  }
  | {
    type: "error"
    error: Parameters<NonNullable<VoiceProviderParam["onError"]>>[0]
  }

const Chat = (props: ComponentProps) => {
  const args = props.args as StreamlitArgs
  const { hume_api_key, listen_to, commands, config_id, debug } = args
  const [events, setEvents] = React.useState<ChatEvent[]>([])
  const addEvent = (event: ChatEvent) => {
    console.log("Event dispatched: ", event.type)
    setEvents([...events, event])
  }
  useEffect(() => {
    console.log("Rerendering component...")
  }, [])
  return (
    <VoiceProvider
      auth={{ type: "apiKey", value: hume_api_key }}
      configId={config_id}
      onMessage={(message) => {
        if (message.type === "user_message") {
          if (message.message.content === '.') {
            // See "clearAudioQueue". Unfortunately, when we `sendUserInput("")` the backend generates
            // userMessage(".") which we don't want to actually pass through.
            return
          }
        }
        addEvent({ type: "message", message })
      }}
      onOpen={() => {
        addEvent({ type: "opened" })
      }}
      onClose={() => {
        addEvent({ type: "closed" })
      }}
      onError={(error) => {
        addEvent({ type: "error", error })
      }}
    >
      <InteractiveChat
        commands={commands}
        events={events}
        listen_to={listen_to}
        debug={debug}
      />
    </VoiceProvider>
  )
}

class StreamlitWrapped extends StreamlitComponentBase<StreamlitArgs> {
  public render = (): ReactNode => {
    return <Chat {...this.props} />
  }
}

export default withStreamlitConnection(StreamlitWrapped)
