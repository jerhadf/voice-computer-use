import React, { useEffect } from "react"
import ReactDOM from "react-dom"
import {Streamlit, StreamlitComponentBase, withStreamlitConnection} from "streamlit-component-lib"

const begin = Date.now()
const sinceSeconds = () => (Date.now() - begin) / 1000
const MyComponent = ({text}: {text: String}) => {
  useEffect(() => {
    setInterval(() => {
      Streamlit.setComponentValue(sinceSeconds())
    }, 1000)
  }, [])

  return <pre>{text}</pre>
}
class MyApp extends StreamlitComponentBase {
  render() {
    return <MyComponent text={this.props.args.text}/>
  }
}
const WrappedApp = withStreamlitConnection(MyApp)
ReactDOM.render(
  <React.StrictMode>
    <WrappedApp />
  </React.StrictMode>,
  document.getElementById("root")
)
