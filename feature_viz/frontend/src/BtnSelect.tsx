import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

import "./custom.css"

interface State {
  selected: number[]
}

class MyComponent extends StreamlitComponentBase<State> {
  public state = {
    selected: JSON.parse(this.props.args["selected"]) as number[],
  }

  componentDidMount(): void {
    super.componentDidMount()
    document.body.style.background = "transparent"
  }

  render(): ReactNode {
    // Arguments
    const words: string[] = this.props.args["words"]

    // Computed values
    const selected = this.state.selected
    const theme: string = this.props?.theme?.base ?? "light"

    return (
      <div className="wrapper">
        {words.map((option, idx) => (
          <button
            onClick={() => this.onClicked(idx)}
            className={`${theme} ${selected.includes(idx) ? "selected" : ""}`}
            key={idx}
          >
            {option}
          </button>
        ))}
      </div>
    )
  }

  private onClicked = (idx: number): void => {
    if (this.state.selected.includes(idx)) {
      let s = this.state.selected.filter((val: any) => val !== idx)
      this.setState(
        (prevState) => ({ current: idx, selected: s }),
        () => Streamlit.setComponentValue(s)
      )
    } else {
      let s = [...this.state.selected, idx]
      this.setState(
        (prevState) => ({ current: idx, selected: s }),
        () => Streamlit.setComponentValue(s)
      )
    }
  }
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(MyComponent)
