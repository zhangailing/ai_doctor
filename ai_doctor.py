import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import json
import inspect


# Initialize session state for messages and agent
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

# OpenAI client setup
@st.cache_resource
def get_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    base_url = st.secrets.get("BASE_URL")
    
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


client = get_client()

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: List = []

# Modified Response class - removed pydantic model validation for agent
class SimpleResponse:
    def __init__(self, agent, messages):
        self.agent = agent
        self.messages = messages

def function_to_schema(func):
    """Convert a Python function to an OpenAI tool schema"""
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        # Skip self parameter for methods
        if name == "self":
            continue
            
        param_type = "string"
        if param.annotation == int:
            param_type = "integer"
        elif param.annotation == float:
            param_type = "number"
        elif param.annotation == bool:
            param_type = "boolean"
            
        schema["function"]["parameters"]["properties"][name] = {"type": param_type}
        
        # Add required parameters
        if param.default == inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(name)
            
    return schema

def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    st.info(f"{agent_name}: {name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments

def run_full_turn(agent, messages):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages,
            tools=tool_schemas if tool_schemas else None,
        )
        message = response.choices[0].message
        message_dict = {"role": message.role}
        if message.content:
            message_dict["content"] = message.content
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        messages.append(message_dict)

        if message.content:  # display agent response
            st.markdown(f"**{current_agent.name}**: {message.content}")

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if isinstance(result, Agent):  # if agent transfer, update current agent
                current_agent = result
                result = f"Transfered to {current_agent.name}. Adopt persona immediately."

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    new_messages = messages[num_init_messages:]
    # Use SimpleResponse instead of pydantic Response
    return SimpleResponse(agent=current_agent, messages=new_messages)

# Tool functions
def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    st.warning("Escalating to human agent...")
    st.markdown("### Escalation Report")
    st.markdown(f"**Summary**: {summary}")
    return "Escalated to human agent. A customer service representative will contact you shortly."

def transfer_to_doctor_search_agent():
    """和找医生以及挂号的代理进行转接"""
    st.markdown("### 转接到医生搜索代理")
    return doctor_search_agent

def transfer_to_service_search_agent():
    """转接到医院、科室搜索代理"""
    st.markdown("### 转接到医院科室搜索代理")
    return service_search_agent

def transfer_back_to_triage():
    """
    Transfer the user back to the initial triage agent.
    
    This function is used when:
    - The current specialized agent (doctor search or hospital department search) has completed its task
    - The user needs assistance outside the current agent's scope
    - The user explicitly requests to go back to the main menu or start over
    - A different type of healthcare inquiry is needed that requires initial assessment
    
    Returns:
        The triage_agent object to handle the conversation going forward
    """
    st.markdown("### 转接回初步分诊代理")
    return triage_agent

def find_doctor(product, price: int):
    """Find a doctor for the user."""
    st.markdown("### 找医生")
    st.markdown(f"**Product**: {product}")
    st.markdown(f"**Price**: ${price}")
    
    col1, col2 = st.columns(2)
    with col1:
        confirm = st.button("Confirm Order")
    with col2:
        cancel = st.button("Cancel Order")
    
    if confirm:
        st.success("Order execution successful!")
        return "Success"
    elif cancel:
        st.error("Order cancelled!")
        return "User cancelled order."
    else:
        return "Waiting for user confirmation..."

def look_up_department(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    st.info(f"Found item: {item_id}")
    return item_id

def search_hospital_info(item_id):
    """Use to find hospital information."""
    st.markdown("### Hospital Information")
    st.markdown(f"**Item ID**: {item_id}")
    st.success("Hospital information retrieved successfully!")
    return "success"

def execute_refund(item_id, reason="not provided"):
    st.markdown("### Refund Summary")
    st.markdown(f"**Item ID**: {item_id}")
    st.markdown(f"**Reason**: {reason}")
    st.success("Refund execution successful!")
    return "success"

# Define agents
triage_agent = Agent(
    name="health Agent",
    instructions=(
        "你是百度AI健康助手，一个专业的医疗咨询虚拟助手。\n"
        "在回答用户时，保持简洁友好，语气亲切专业。\n"
        "你的主要职责包括：\n"
        "1. 简短自我介绍，表明你是AI健康助手\n"
        "2. 理解用户的健康需求或问题\n"
        "3. 根据用户需求，提供初步的健康信息和建议\n"
        "4. 判断用户是需要找医生还是需要查找医院/科室信息\n"
        "5. 自然地引导用户表达具体需求，不要生硬地询问\n"
        "6. 根据用户需求，适时转接到医生搜索代理或服务搜索代理\n"
        "7. 当遇到复杂或紧急情况时，适时升级至人工客服\n\n"
        "注意：回答要简洁明了，避免过长解释。在引导用户时，保持对话自然流畅。"
    ),
    tools=[transfer_to_doctor_search_agent, transfer_to_service_search_agent, escalate_to_human],
)

doctor_search_agent = Agent(
    name="Doctor Search Agent",
    instructions=(
        "You are an assistant helping users find a suitable doctor.\n"
        "Always answer concisely, preferably in a sentence or less.\n"
        "Follow this routine with the user:\n"
        "1. Understand their medical needs or concerns.\n"
        "2. Provide information on available doctors based on their needs.\n"
        "3. Offer to assist further with scheduling an appointment if requested.\n"
        "4. Ensure clarity and accuracy in all communications to maintain user trust and satisfaction.\n"
    ),
    tools=[find_doctor, transfer_back_to_triage],  # 确保这些工具函数符合优化后的流程
)

service_search_agent = Agent(
    name="Hospital Department Search Agent",
    instructions=(
        "You are a hospital department search agent, helping users find appropriate medical departments based on their symptoms or healthcare needs."
        "Always answer concisely and clearly."
        "Follow the following routine with the user:"
        "1. First, ask about their symptoms or medical needs if not already provided.\n"
        "2. Based on the symptoms/needs, recommend an appropriate hospital department (科室).\n"
        "3. Provide a brief explanation of why this department is suitable for their condition.\n"
        "4. If the user needs more information about the department, provide basic details about what the department handles.\n"
        "5. If you cannot determine an appropriate department or the query is outside medical scope, suggest consulting with a general practitioner first."
    ),
    tools=[look_up_department, search_hospital_info, transfer_back_to_triage],
)

# Initialize agent if not already done
if st.session_state.agent is None:
    st.session_state.agent = triage_agent

# Streamlit UI
st.title("AI健康助手")
st.write("问答、找医生、找服务")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You**: {message['content']}")
    elif message["role"] == "assistant" and message.get("content"):
        agent_name = st.session_state.agent.name if st.session_state.agent else "Assistant"
        st.markdown(f"**{agent_name}**: {message['content']}")

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"**You**: {user_input}")
    
    # Process the message
    response = run_full_turn(st.session_state.agent, st.session_state.messages)
    
    # Update agent and messages
    st.session_state.agent = response.agent
    st.session_state.messages.extend(response.messages)
    
    # Force a rerun to display the new messages
    st.rerun()

# Add a reset button
if st.button("Reset Conversation"):
    st.session_state.messages = []
    st.session_state.agent = triage_agent
    st.rerun()