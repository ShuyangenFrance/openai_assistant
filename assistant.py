import os
import ast
import streamlit as st
import openai
from openai import  AssistantEventHandler
import math
from typing_extensions import override
from openai.types.beta.threads import Text, TextDelta
import json

def calculate_tax(revenue: str):
    try:
        revenue = float(revenue)
    except ValueError:
        raise ValueError("The revenue should be a string representation of a number.")

    if revenue <= 10000:
        tax = 0
    elif revenue <= 30000:
        tax = 0.10 * (revenue - 10000)
    elif revenue <= 70000:
        tax = 2000 + 0.20 * (revenue - 30000)
    elif revenue <= 150000:
        tax = 10000 + 0.30 * (revenue - 70000)
    else:
        tax = 34000 + 0.40 * (revenue - 150000)

    return tax




function_tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_tax",
            "description": "Get the tax for given revenue in euro",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {
                        "type": "string",
                        "description": "Annual revenue in euro"
                    }
                },
                "required": ["revenue"]
            }
        }
    }
]

class EventHandler(AssistantEventHandler):
    """
    Event handler for the assistant stream
    """

    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == 'thread.run.requires_action':
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)

    @override
    def on_text_created(self, text: Text) -> None:
        """
        Handler for when a text is created
        """
        # This try-except block will update the earlier expander for code to complete.
        # Note the indexing. We are updating the x-1 textbox where x is the current textbox.
        # Note how `on_tool_call_done` creates a new textbook (which is the x_th textbox, so we want to access the x-1_th)
        # This is to address an edge case where code is executed, but there is no output textbox (e.g. a graph is created)
        try:
            st.session_state[f"code_expander_{len(st.session_state.text_boxes) - 1}"].update(state="complete",
                                                                                             expanded=False)
        except KeyError:
            pass

        # Create a new text box
        st.session_state.text_boxes.append(st.empty())
        # Display the text in the newly created text box
        st.session_state.text_boxes[-1].info("".join(st.session_state["assistant_text"][-1]))

    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text):
        """
        Handler for when a text delta is created
        """
        # Clear the latest text box
        st.session_state.text_boxes[-1].empty()
        # If there is text written, add it to latest element in the assistant text list
        if delta.value:
            st.session_state.assistant_text[-1] += delta.value
            #st.session_state.chat_history.append(("assistant", delta.value))
        # Re-display the full text in the latest text box
        st.session_state.text_boxes[-1].info("".join(st.session_state["assistant_text"][-1]))

    def on_text_done(self, text: Text):
        """
        Handler for when text is done
        """
        # Create new text box and element in the assistant text list
        st.session_state.text_boxes.append(st.empty())
        st.session_state.assistant_text.append("")
        st.session_state.chat_history.append(("assistant", text.value))
    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "calculate_tax":
                try:
                    # Extract revenue from tool parameters
                    revenue = ast.literal_eval(tool.function.arguments)["revenue"]
                    # Call your calculate_tax function to get the tax
                    tax_result = calculate_tax(revenue)
                    # Append tool output in the required format
                    tool_outputs.append({"tool_call_id": tool.id, "output": f"{tax_result}"})
                except ValueError as e:
                    # Handle any errors when calculating tax
                    tool_outputs.append({"tool_call_id": tool.id, "error": str(e)})
        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs)

    def submit_tool_outputs(self, tool_outputs):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                print(text, end="", flush=True)
            print()

# Initialize the OpenAI client
client = openai.Client(api_key=os.environ.get("OPENAI_API_KEY"))

assistant = client.beta.assistants.create(
    name="Assistant",
    instructions="",
    tools=function_tools,
    model="gpt-4o",
)

st.title("💬 Chatbot")
text_box = st.empty()


# Initialize chat history in session state if not already done
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if "assistant_text" not in st.session_state:
    st.session_state.assistant_text = [""]

if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

if "text_boxes" not in st.session_state:
    st.session_state.text_boxes = []

def display_chat_history():
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.chat_message("User").write(content)
        else:
            st.chat_message("Assistant").write(content)

display_chat_history()

if prompt := st.chat_input("Enter your message"):
    st.session_state.chat_history.append(("user", prompt))

    # Assuming the user input is related to calculating tax
    if "tax" in prompt.lower():  # You can adjust this condition based on your actual use case
        try:
            tax_result = calculate_tax(prompt)  # Assuming prompt contains revenue
            st.session_state.chat_history.append(
                ("assistant", f"Tax for revenue {tax_result['revenue']}: {tax_result['tax']}"))
        except ValueError as e:
            st.session_state.chat_history.append(("assistant", str(e)))

    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt
    )

    st.session_state.text_boxes.append(st.empty())
    st.session_state.text_boxes[-1].success(f" {prompt}")

    with client.beta.threads.runs.stream(thread_id=st.session_state.thread_id,
                                         assistant_id=assistant.id,
                                         event_handler=EventHandler(),
                                         temperature=0) as stream:
        stream.until_done()

