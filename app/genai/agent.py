import json
import os
import streamlit as st
from .tools import build_tools_dict, get_genai_declarations
from ..config import FAMILY_NAMES

_GENAI_FAMILY_CODE_MAP = ", ".join(f"{code}={name}" for code, name in FAMILY_NAMES.items())

GENAI_SYSTEM_PROMPT = (
    "You are an analyst embedded in a retail demand-forecasting dashboard (Corporación "
    "Favorita Kaggle dataset). Only state numbers that come from a tool result — never "
    "estimate or invent sales figures, SHAP values, or error metrics. If a tool returns "
    "an error or no data, say so plainly rather than filling the gap yourself. If a "
    "question is unrelated to this forecasting project, politely decline. "
    f"Product family codes: {_GENAI_FAMILY_CODE_MAP}. When the user names a family in "
    "words (e.g. 'beverages'), map it to the matching numeric code before calling a tool."
)

try:
    from google import genai as _google_genai
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False


def _get_gemini_api_key():
    """Return the Gemini API key from (in priority order):
    1. Sidebar text-input (session state)
    2. .streamlit/secrets.toml
    3. GEMINI_API_KEY environment variable
    """
    sidebar_key = st.session_state.get("sidebar_gemini_key", "").strip()
    if sidebar_key:
        return sidebar_key
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    return key or os.environ.get("GEMINI_API_KEY")

def _validate_gemini_key(key):
    if not key:
        return False, "No key provided."
    if len(key) < 20:
        return False, "Key looks too short — double-check you copied the full value."
    return True, ""

@st.cache_resource
def get_genai_client(api_key):
    return _google_genai.Client(api_key=api_key)

@st.cache_data(show_spinner=False)
def generate_quick_insight_gemini(selected_families_str, date_min_str, date_max_str, total_sales, leading_family, trend_direction):
    if not GENAI_SDK_AVAILABLE:
        return "The `google-genai` package isn't installed. Quick Insight is disabled."
    key = _get_gemini_api_key()
    if not key:
        return ("GEMINI_API_KEY not found. Paste a valid key in the sidebar or add it to "
                "`.streamlit/secrets.toml`. Get one free at https://aistudio.google.com/apikey")
    try:
        client = _google_genai.Client(api_key=key)
        prompt = (
            f"Write a 2-3 sentence plain-English caption summarizing these sales metrics for {selected_families_str} "
            f"between {date_min_str} and {date_max_str}:\n"
            f"- Total Sales: {total_sales:,.0f}\n"
            f"- Leading Family: {leading_family}\n"
            f"- Trend Direction: {trend_direction}\n"
            "Keep it professional and concise. Do not add any filler text."
        )
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "401" in err or "UNAUTHENTICATED" in err:
            return ("⚠️ Invalid API key — your GEMINI_API_KEY was rejected (401). "
                    "Make sure you're using a key from https://aistudio.google.com/apikey "
                    "(it should start with `AIzaSy…`). Paste a valid key in the sidebar.")
        return f"Could not generate Quick Insight via Gemini API: {e}"

def run_genai_agent(client, user_message, chat_contents=None,
                     model="gemini-3.6-flash", max_rounds=5, df=None, lgb_model=None):
    tool_log = []
    types = _google_genai.types

    contents = list(chat_contents or [])
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
    
    tools_dict = build_tools_dict(df, lgb_model)
    genai_tools = get_genai_declarations()

    config = types.GenerateContentConfig(
        system_instruction=GENAI_SYSTEM_PROMPT,
        tools=genai_tools,
    )

    for _ in range(max_rounds):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        function_call_parts = [
            p for p in (response.candidates[0].content.parts or [])
            if p.function_call is not None
        ]

        if not function_call_parts:
            answer_text = response.text or "(No response text)"
            contents.append(response.candidates[0].content)
            return answer_text, tool_log, contents

        contents.append(response.candidates[0].content)

        fn_response_parts = []
        for part in function_call_parts:
            fc = part.function_call
            fn = tools_dict.get(fc.name)
            args = dict(fc.args) if fc.args else {}
            result = fn(**args) if fn else {"error": f"Unknown tool '{fc.name}'"}
            tool_log.append({"tool": fc.name, "input": args, "result": result})
            fn_response_parts.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": json.dumps(result, default=str)},
                ))
            )

        contents.append(types.Content(role="user", parts=fn_response_parts))

    return "Ran out of tool-call rounds — try a more specific question.", tool_log, contents

def render_genai_tab(df, model):
    st.header("🤖 Ask the GenAI Analyst")
    st.caption(
        "Answers are grounded in this app's own pipeline — every number traces back "
        "to a tool call (shown below each response), and real-world context (the 2016 "
        "earthquake, oil price shocks, holiday quirks) is retrieved from a small knowledge base."
    )

    if not GENAI_SDK_AVAILABLE:
        st.error(
            "The `google-genai` package isn't installed. Add `google-genai` to "
            "requirements.txt to enable this tab."
        )
    else:
        genai_api_key = _get_gemini_api_key()
        key_valid, key_err = _validate_gemini_key(genai_api_key) if genai_api_key else (False, "")
        if not genai_api_key:
            st.warning(
                "No `GEMINI_API_KEY` found. Paste a valid key in the **sidebar** 🔑 or add it "
                "to `.streamlit/secrets.toml`. Get a free key at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
            )
        elif not key_valid:
            st.error(f"⚠️ {key_err}")
        else:
            genai_client = get_genai_client(genai_api_key)

            if "genai_chat_history" not in st.session_state:
                st.session_state.genai_chat_history = []
            if "genai_chat_contents" not in st.session_state:
                st.session_state.genai_chat_contents = []

            st.markdown("**Try an example:**")
            example_questions = [
                "Why did sales spike around the 2016 earthquake?",
                "Which model has the lowest error overall?",
                "Any flagged anomalies for store 1, family 3 (beverages)?",
            ]
            example_cols = st.columns(len(example_questions))
            clicked_example = None
            for col, q in zip(example_cols, example_questions):
                if col.button(q, use_container_width=True, key=f"genai_ex_{q}"):
                    clicked_example = q

            for msg in st.session_state.genai_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            typed_q = st.chat_input("Ask about forecasts, drivers, anomalies, or model accuracy…")
            user_q = typed_q or clicked_example

            if user_q:
                st.session_state.genai_chat_history.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            genai_answer, genai_tool_log, genai_new_contents = run_genai_agent(
                                genai_client, user_q, st.session_state.genai_chat_contents,
                                df=df, lgb_model=model
                            )
                        except Exception as e:
                            err = str(e)
                            if "401" in err or "UNAUTHENTICATED" in err:
                                genai_answer = (
                                    "🔑 **Invalid API key.** Your `GEMINI_API_KEY` was "
                                    "rejected (401 Unauthenticated). Please paste a valid "
                                    "key in the **sidebar** — it should start with `AIzaSy…`. "
                                    "Get one free at [aistudio.google.com/apikey]"
                                    "(https://aistudio.google.com/apikey)."
                                )
                            elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                                genai_answer = (
                                    "⏳ **Rate limit hit.** The Gemini free tier limits "
                                    "requests per minute — wait 60 seconds and try again."
                                )
                            else:
                                genai_answer = (
                                    f"Something went wrong calling the Gemini API: {e}. "
                                    "Check the sidebar to verify your API key is valid."
                                )
                            genai_tool_log = []
                            genai_new_contents = st.session_state.genai_chat_contents
                    st.markdown(genai_answer)
                    if genai_tool_log:
                        with st.expander(f"🔧 {len(genai_tool_log)} tool call(s) used"):
                            for t in genai_tool_log:
                                st.json(t)

                st.session_state.genai_chat_contents = genai_new_contents
                st.session_state.genai_chat_history.append({"role": "assistant", "content": genai_answer})
