import json
from dataclasses import dataclass
from typing import Dict
import inspect

import streamlit as st

# try:
#     import pyperclip
# except ImportError:
#     pyperclip = None


PIXELS_PER_LINE = 27
INDENT = 8


@st.cache(allow_output_mutation=True)
def state_singleton() -> Dict:
    return {}


STATE = state_singleton()


@dataclass
class JsonInputState:
    value: dict
    default_value: dict
    redraw_counter = 0


class CopyPasteError(Exception):
    pass


def dict_input(label, value, mutable_structure=False, key=None):
    """Display a dictionary or dictionary input widget.
    This implementation is composed of a number of streamlit widgets. It might
    be considered a prototype for a native streamlit widget (perhaps built off
    the existing interactive dictionary widget).
    Json text may be copied in and out of the widget.
    Parameters
    ----------
    label : str
        A short label explaining to the user what this input is for.
    value : dict or func
        The dictionary of values to edit or a function (with only named parameters).
    mutable_structure : bool
        If True allows changes to the structure of the initial value.
        Otherwise the keys and the type of their values are fixed.
        Defaults to False (non mutable).
    key : str
        An optional string to use as the unique key for the widget.
        If this is omitted, a key will be generated for the widget
        based on its content. Multiple widgets of the same type may
        not share the same key.
    Returns
    -------
    dict
        The current value of the input widget.
    Example
    -------
    >>> d = st.json_input('parameters', {'a': 1, 'b': 2.0, 'c': 'abc', 'd': {a: 2}})
    >>> st.write('The current parameters are', d)
    """
    try:
        param = inspect.signature(value).parameters
        value = {}
        for p in param.values():
            value[p.name] = p.default
    except TypeError:
        pass  # Assume value is a dict

    # check json can handle input
    value = json.loads(json.dumps(value))

    # Create state on first run
    state_key = f"json_input-{key if key else label}"
    if state_key not in STATE:
        STATE[state_key] = JsonInputState(value, value)
    state: JsonInputState = STATE[state_key]

    # containers
    text_con = st.empty()
    warning_con = st.empty()

    def json_input_text(msg=""):

        if msg:
            state.redraw_counter += 1
            state.default_value = state.value

        # Display warning
        if msg:
            warning_con.warning(msg)
        else:
            warning_con.empty()

        # Read value
        value_s = json.dumps(
            state.default_value, indent=INDENT, sort_keys=True
        )
        input_s = text_con.text_area(
            label,
            value_s,
            height=len(value_s.splitlines()) * PIXELS_PER_LINE,
            key=f"{key if key else label}-{state.redraw_counter}",
            # help="help"
        )

        # Decode
        try:
            new_value = json.loads(input_s)
        except json.decoder.JSONDecodeError:
            return json_input_text(
                "The last edit was invalid json and has been reverted"
            )

        # Check structure
        if not mutable_structure:
            if not keys_match(new_value, state.value):
                return json_input_text(
                    "The last edit changed the structure of the json "
                    "and has been reverted"
                )

            if not value_types_match(new_value, state.value):
                return json_input_text(
                    "The last edit changed the type of an entry "
                    "and has been reverted"
                )

        return new_value

    # Input a valid dict
    state.value = json_input_text()

    # Copy and paste buttons

    # try:
    #     copy_con, paste_con = st.beta_columns((1, 5))
    # except st.StreamlitAPIException:
    #     copy_con, paste_con = st.empty(), st.empty()

    # if copy_con.button("Copy", key=key if key else label + "-copy"):
    #     copy_json(state.value)

    # if paste_con.button("Paste", key=key if key else label + "-paste"):
    #     try:
    #         _new_value = paste_json(state.value, mutable_structure)
    #         state.default_value = state.value = _new_value
    #         state.redraw_counter += 1
    #         return json_input_text("")
    #     except CopyPasteError as e:
    #         st.warning(e)
    st.write("----")

    return state.value


# def copy_json(d):
#     if not pyperclip:
#         raise Exception("Install the `pyperclip` package")
#     pyperclip.copy(json.dumps(d, indent=INDENT, sort_keys=True))


# def paste_json(current_value, mutable_structure):
#     if not pyperclip:
#         raise Exception("Install the `pyperclip` package")
#     s = pyperclip.paste()
#     try:
#         new_value = json.loads(s)
#     except json.decoder.JSONDecodeError as e:
#         raise CopyPasteError(
#             f"Paste failed: Invalid json {e}: \n\n```\n{s}\n```"
#         )
#     if not mutable_structure:
#         if not keys_match(new_value, current_value):
#             raise CopyPasteError(
#                 "Paste failed: The json structure does not match that of the "
#                 "current dictionary (and widget's mutable_structure=False): "
#                 f"\n\n```\n{s}\n```"
#             )
#         elif not value_types_match(new_value, current_value):
#             raise CopyPasteError(
#                 "Paste failed: The type of a value does not match that of the"
#                 " current dictionary (and widget's mutable_structure=False): "
#                 f"\n\n```\n{s}\n```"
#             )
#     return new_value


def keys_match(d1, d2):

    if d1.keys() != d2.keys():
        return False

    for k, v in d1.items():
        if isinstance(v, dict):
            if not keys_match(v, d2[k]):
                return False

    for k, v in d2.items():
        if isinstance(v, dict):
            if not keys_match(v, d1[k]):
                return False

    return True


def value_types_match(d1, d2):
    # assuming the keys match
    for k in d1.keys():
        if isinstance(d1[k], dict):
            if not value_types_match(d1[k], d2[k]):
                return False
        if type(d1[k]) is not type(d2[k]):
            return False

    return True


if __name__ == "__main__":

    st.title("Toy `dict_input` widget")

    st.write(
        """
        A version of the standard dict view that is editable would be handy for
        quick prototyping, for when an app has many parameters, and as a
        supplemental way to copy configuration in and out of a streamlit app.
        
        A native `dict_input` widget might be used to edit a
        dictionary like this
        """
    )
    with st.echo():
        dict_template = {
            "a": 1,
            "b": 2.0,
            "c": "abc",
            "d": {"a": 3},
            "e": [4, 5.0, "def"],
        }

    st.write(
        """
        and might look like a cross between the widgets below. The left is an
        editable view of the standard dict widget on the right.
        """
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.write("A dict_input composite widget:")
        with st.echo():
            d = dict_input("Edit me!", dict_template)
    with col2:
        st.write("A standard dictionary view:")
        with st.echo():
            st.write(".", d)

    st.write(
        """
        The view on the left can be edited. It will revert to its last valid
        state if invalid json is entered, or if the key-structure of the dict
        is changed or the type of a value is changed from that of its initial
        value (`config`).  The buttons copy json out of the widget or into it.
    
        ### Call with a function
        The value given to `json_input` might be a function rather than a dict.
        As long as all the parameters have defaults then the inital dict is
        inferred.  For example:
        """
    )

    with st.echo():
        def func(a=1, b=2.0, c="c"):
            return a, b, c

        config = dict_input("Parameters to call `func` with", func)
        
        st.write("func output:\n\n", func(**config))

    st.write(
        """
        ### Options
        `dict_input` might also take a `dataclass` (not implemented). The option
        `mutable_structure` may be set to True allowing the key structure and
        value types to change (implemented)."""
    )