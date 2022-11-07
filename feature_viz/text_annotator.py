import os
import streamlit as st
import streamlit.components.v1 as components

__RELEASE = True
if not __RELEASE:
    _btn_select = components.declare_component(
        "btn_select",
        url="http://localhost:3001",
    )

else:
    _btn_select = components.declare_component(
        "btn_select",
        path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend/build"),
    )

def text_annotator(words, selected=[], format_func=str, key=None):
    """Create a new instance of "text_annotator".

    Parameters
    ----------
    words: Sequence, numpy.ndarray, pandas.Series, pandas.DataFrame or pandas.Index
        Labels for the select words.
        This will be cast to str internally by default.
        For pandas.DataFrame, the first column is selected.
    index: int, optional
        The index of the preselected option on first render.
        Default ``0``
    format_func: function
        Function to modify the display of the labels.
        It receives the option as an argument and its output will be cast to str.
        Default ``str``
    nav: bool, optional
        Whether to use this widget as a top-anchored navigation.
        Default ``False``
    key: str or None
        An optional key that uniquely identifies this component.
        If this is None and the component's arguments are changed,
        the component will be re-mounted in the Streamlit frontend and lose its current state.

    Returns
    -------
    string
        Which button was selected.

    Warning
    -------
    You can only have one `st_btn_select` as navigation, as others will be displayed on top of it.
    """
    key = st.type_util.to_key(key)
    opt = st.type_util.ensure_indexable(words)

    indices = _btn_select(
        words=[str(format_func(option)) for option in opt],
        selected=selected,
        key=key,
    )

    return indices
