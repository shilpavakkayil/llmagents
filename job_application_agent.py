from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os
import re
import streamlit as st
import fitz
import magic

load_dotenv()
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = OllamaLLM(
    model = "llama3.2",
    base_url= "http://localhost:11434"
)
applicant_details = {
    "name": None,
    "skills": None,
    "email": None
}

def get_applicant_details(text:str)->str:
    #name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE) 
    name_match = re.search(r"(?:my name is|i am)\s+([A-Za-z\s\-]+)", text, re.IGNORECASE)
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)  
    #skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+([a-zA-Z, ]+)", text, re.IGNORECASE)

    response = [] 

    if name_match: 
        applicant_details["name"] = name_match.group(1).title()
        response.append(" Name saved.") 


    if email_match:
        applicant_details["email"] = email_match.group(0)
        response.append(" Email saved.")
    if skills_match:
        applicant_details["skills"] = skills_match.group(1).strip()
        response.append(" Skills saved.")

    '''if not any([name_match, email_match, skills_match]):
        return " Please provide your name, email, or skills"'''

    return " ".join(response) + " Let me check what else I need."

def get_file_type(file_stream):
    # Read the first 2048 bytes for detection (magic numbers)
    file_stream.seek(0)  # Ensure we're at the start
    buffer = file_stream.read(2048)
    file_stream.seek(0)  # Reset stream position after reading

    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(buffer)
    
    if mime_type == "application/pdf":
        return "pdf"
    elif mime_type == "text/plain":
        return "txt"
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


def extract_pdf_text(file_uploaded):
    #file_type = get_file_type(file_uploaded)
    file_uploaded.seek(0)

    name = (getattr(file_uploaded, "name", "") or "").lower()
    mime_type = getattr(file_uploaded, "type", None)
    if mime_type == "application/pdf":
        file_type = "pdf"
    elif mime_type == "text/plain":
        file_type = "txt"
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")
    if file_type == "pdf":
        doc = fitz.open(stream=file_uploaded.read(), filetype="txt")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    elif file_type == "txt":
        return file_uploaded.read().decode('utf-8')  # Decode bytes to string
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT allowed.")


def parse_info_from_cv(text: str):
    info_from_cv = {"name": None, "email": None, "skills": None}
    name_match = re.search(r"(?:Full Name:|Name:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    email_match = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", text)
    skills_match = re.search(r"Skills\s*-+\s*(.*?)\n(?:Projects|Certifications|$)", text, re.DOTALL)

    if name_match:
        info_from_cv["name"] = name_match.group(1).strip()
    if email_match:
        info_from_cv["email"] = email_match.group(0).strip()
    if skills_match:
        skills = skills_match.group(1).replace("\n", ", ").replace("\u2022", "").replace("-", "")
        info_from_cv["skills"] = re.sub(r"\s+", " ", skills.strip())

    return info_from_cv


def verify_application(_: str) -> str:
    #print(applicant_details.values())
    if all(applicant_details.values()):
        return f" You're ready! Name: {applicant_details['name']}, Email: {applicant_details['email']}, Skills: {applicant_details['skills']}."
    else:
        missing = [k for k, v in applicant_details.items() if not v]
        return f" Still need: {', '.join(missing)}. Please ask the user to provide this."


tools = [
    Tool(
        name="get_applicant_details",
        func=get_applicant_details,
        description="Use this to extract name, email, and skills from the user's message."
    ),
    Tool(
        name="verify_application",
        func=verify_application,
        description="Check if name, email, and skills are provided. If not, tell the user to provide the missing information",
        return_direct=True  
    )

]
SYSTEM_PROMPT = """You are a helpful job application assistant. 
Your goal is to collect the user's name, email, and skills. 
Use the tools provided to extract this information and check whether all required data is collected.
Once everything is collected, inform the user that the application info is complete and stop.
"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    #agent_kwargs={"system_message": SYSTEM_PROMPT}
)

'''print(" Hi! I'm your job application assistant. Please tell me your name, email, and skills.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print(" Bye! All the best.")
        break

    response = agent.invoke({"input": user_input})
    print("Bot:", response["output"])

    # If goal achieved, stop
    if "you're ready" in response["output"].lower():
        print(" Applicant details are verified!")
        break'''

# Streamlit UI
st.set_page_config(page_title=" llm agent-assisting job applications", layout="centered")
st.title(" Your Job Assistant")
st.markdown("Give me your **name**, **email**, and **skills** to get started with your application!")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "goal_complete" not in st.session_state:
    st.session_state.goal_complete = False
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False
if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""

# Upload resume
st.sidebar.header(" Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])

if resume:
    st.sidebar.success("Resume uploaded!")
    text = extract_pdf_text(resume)
    extracted = parse_info_from_cv(text)
    for key in applicant_details:
        if extracted[key]:
            applicant_details[key] = extracted[key]
    st.sidebar.info(" Extracted info from resume:")
    for key, value in extracted.items():
        st.sidebar.markdown(f"**{key.capitalize()}:** {value}")

# Reset chat
if st.sidebar.button(" Reset Chat"):
    st.session_state.chat_history.clear()
    st.session_state.goal_complete = False
    st.session_state.download_ready = False
    st.session_state.application_summary = ""
    for key in applicant_details:
        applicant_details[key] = None
    st.rerun()

# Chat input
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    get_applicant_details(user_input)
    response = agent.invoke({"input": user_input})
    bot_reply = response["output"]
    st.session_state.chat_history.append(("bot", bot_reply))
    goal_status = verify_application("check")
    st.session_state.chat_history.append(("status", goal_status))

    if "you're ready" in goal_status.lower():
        st.session_state.goal_complete = True
        summary = (
            f" Name: {applicant_details['name']}\n"
            f" Email: {applicant_details['email']}\n"
            f" Skills: {applicant_details['skills']}\n"
        )
        st.session_state.application_summary = summary
        st.session_state.download_ready = True

# Chat UI with avatars
for sender, message in st.session_state.chat_history:
    if sender == "user":
        with st.chat_message("🧑"):
            st.markdown(message)
    elif sender == "bot":
        with st.chat_message("🤖"):
            st.markdown(message)
    elif sender == "status":
        with st.chat_message("📊"):
            st.info(message)

# Final message
if st.session_state.goal_complete:
    st.success(" All information verified! You're ready to apply!")

# Download summary
if st.session_state.download_ready:
    st.download_button(
        label=" Download Application Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain"
    )
