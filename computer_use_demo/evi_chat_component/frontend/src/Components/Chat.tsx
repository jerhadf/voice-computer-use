import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { ReactNode, useEffect } from "react"
import { VoiceProvider, useVoice } from "@humeai/voice-react";
import { type Hume } from "hume"
import StartCall from "./StartCall";
import Controls from "./Controls";

type UseVoiceReturn = ReturnType<typeof useVoice>;
type InteractivityProps = {
  muted: boolean;
  assistant_paused: boolean;
  assistant_audio_muted: boolean;
  user_input_message: Parameters<UseVoiceReturn['sendUserInput']>[0] | null;
  assistant_input_message: Parameters<UseVoiceReturn['sendAssistantInput']>[0] | null;
  session_settings_message: Parameters<UseVoiceReturn['sendSessionSettings']>[0] | null;
  tool_response_message: Hume.empathicVoice.ToolResponseMessage;
  tool_error_message: Hume.empathicVoice.ToolErrorMessage;
}
const useInteractivity = ({
  muted,
  assistant_paused,
  assistant_audio_muted,
  user_input_message,
  assistant_input_message,
  session_settings_message,
  tool_response_message,
  tool_error_message,
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
  } = useVoice();

  const useMessageSender = <TMessage,>(message: TMessage | null, sendFn: (message: TMessage) => void) => {
    useEffect(() => {
      if (!message) return;
      sendFn(message);
    }, [message])
  }

  useMessageSender(user_input_message, sendUserInput);
  useMessageSender(assistant_input_message, sendAssistantInput);
  useMessageSender(session_settings_message, sendSessionSettings);
  useMessageSender(tool_response_message, sendToolMessage);
  useMessageSender(tool_error_message, sendToolMessage);

  const useToggleCommand = (state: boolean, enableFn: () => void, disableFn: () => void) => {
    useEffect(() => {
      if (state) {
        enableFn();
      } else {
        disableFn();
      }
    }, [state])
  }
  useToggleCommand(muted, mute, unmute);
  useToggleCommand(assistant_paused, pauseAssistant, resumeAssistant);
  useToggleCommand(assistant_audio_muted, muteAudio, unmuteAudio);
}

const InteractiveChat = (props: ComponentProps) => {
  useInteractivity(props.args);
  return <>
    <Controls />,
    <StartCall />
  </>
}
type StreamlitArgs = InteractivityProps & {
  hume_api_key: string,
}

type VoiceProviderParam = Parameters<typeof VoiceProvider>[0];
type ChatEvent = {
  type: 'message',
  message: Parameters<NonNullable<VoiceProviderParam['onMessage']>>[0]
} | {
  type: 'closed'
} | {
  type: 'opened'
} | {
  type: 'error',
  error: Parameters<NonNullable<VoiceProviderParam['onError']>>[0]
}

const Chat = (props: ComponentProps) => {
  const [events, setEvents] = React.useState<ChatEvent[]>([]);
  const addEvent = (event: ChatEvent) => {
    setEvents([...events, event]);
  }
  useEffect(() => {
    Streamlit.setComponentValue(events);
  }, [events])
  return <VoiceProvider
    auth={{ type: 'apiKey', value: props.args['hume_api_key'] as string }}
    onMessage={(message) => {
      addEvent({ type: 'message', message })
    }}
    onOpen={() => {
      addEvent({ type: 'opened' })
    }}
    onClose={() => {
      addEvent({ type: 'closed' })
    }}
    onError={(error) => {
      addEvent({ type: 'error', error })
    }}>
    <InteractiveChat {...props} />
  </VoiceProvider>
}

class StreamlitWrapped extends StreamlitComponentBase<StreamlitArgs> {
  public render = (): ReactNode => {
    return (
      <Chat {...this.props} />
    )
  }
}


export default withStreamlitConnection(Chat)
