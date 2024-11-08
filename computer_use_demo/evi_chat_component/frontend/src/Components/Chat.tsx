import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { ReactNode, useEffect } from "react"
import { VoiceProvider, useVoice } from "@humeai/voice-react"
import { type Hume } from "hume"
import StartCall from "./StartCall"
import Controls from "./Controls"

type UseVoiceReturn = ReturnType<typeof useVoice>
type Command = {
  type: 'mute' | 'unmute' | 'pauseAssistant' | 'resumeAssistant' | 'muteAudio' | 'unmuteAudio'
} | {
  type: 'sendUserInput',
  message: Parameters<UseVoiceReturn["sendUserInput"]>[0]
} | {
  type: 'sendAssistantInput',
  message: Parameters<UseVoiceReturn["sendAssistantInput"]>[0]
} | {
  type: 'sendSessionSettings',
  message: Parameters<UseVoiceReturn["sendSessionSettings"]>[0]
} | {
  type: 'sendToolMessage',
  message: Parameters<UseVoiceReturn["sendToolMessage"]>[0]
}

type InteractivityProps = {
  commands: Command[]
}
const useInteractivity = ({
  commands,
}: InteractivityProps) => {
  const {
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
  } = useVoice()

  const dispatchCommand = (command: Command) => {
    switch (command.type) {
      case 'mute':
        mute()
        return
      case 'unmute':
        unmute()
        return
      case 'pauseAssistant':
        pauseAssistant()
        return
      case 'resumeAssistant':
        resumeAssistant()
        return
      case 'muteAudio':
        muteAudio()
        return
      case 'unmuteAudio':
        unmuteAudio()
        return
      case 'sendUserInput':
        sendUserInput(command.message)
        return
      case 'sendAssistantInput':
        sendAssistantInput(command.message)
        return
      case 'sendSessionSettings':
        sendSessionSettings(command.message)
        return
      case 'sendToolMessage':
        sendToolMessage(command.message)
        return
    }
  }

  const [cursor, setCursor] = React.useState(0)
  const newCommands = commands.slice(cursor, commands.length)
  useEffect(() => {
    newCommands.forEach((command) => dispatchCommand(command))
    setCursor(commands.length)
  }, [commands])

}

const InteractiveChat = (props: ComponentProps) => {
  useInteractivity(props.args)
  return (
    <>
      <Controls />,
      <StartCall />
    </>
  )
}
type Listenable = `message.${Parameters<NonNullable<VoiceProviderParam["onMessage"]>>[0]['type']}` | ChatEvent["type"]
type StreamlitArgs = InteractivityProps & {
  hume_api_key: string
  listen_to?: Array<Listenable>
}
const defaultListenTo: Array<Listenable> = ["message.user_message", "opened", "closed", "error"]
const listenedTo = (listen_to: Array<Listenable> = defaultListenTo, event: ChatEvent) => {
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
  const { hume_api_key, listen_to } = props.args as StreamlitArgs
  const [events, setEvents] = React.useState<ChatEvent[]>([])
  const [cursor, setCursor] = React.useState(0)
  const addEvent = (event: ChatEvent) => {
    setEvents([...events, event])
  }
  useEffect(() => {
    const newEvents = events.slice(cursor, events.length)
    if (newEvents.some(e => listenedTo(listen_to, e))) {
      Streamlit.setComponentValue(events);
      setCursor(events.length)
    }
  }, [events])
  return (
    <VoiceProvider
      auth={{ type: "apiKey", value: hume_api_key }}
      onMessage={(message) => {
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
      <InteractiveChat {...props} />
    </VoiceProvider>
  )
}

class StreamlitWrapped extends StreamlitComponentBase<StreamlitArgs> {
  public render = (): ReactNode => {
    return <Chat {...this.props} />
  }
}

export default withStreamlitConnection(StreamlitWrapped)
