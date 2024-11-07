import streamlit.components.v1 as components

_component_func = components.declare_component(
    # We give the component a simple, descriptive name ("my_component"
    # does not fit this bill, so please choose something better for your
    # own component :)
    "test_component",
    # Pass `url` here to tell Streamlit that the component will be served
    # by the local dev server that you run via `npm run start`.
    # (This is useful while your component is in development.)
    url="http://localhost:3001",
    #path="./computer_use_demo/test_component/frontend/public")
)


def test_component(
    *,
    text: str,
    key=None
) -> int:
    component_value = _component_func(
        text=text,
        key=key,
        default=None)

    return component_value
