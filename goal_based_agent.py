from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os
import re


applicant_details = {
    "name": None,
    "skills": None,
    "email": None
}


def get_applicant_details(text:str)->str:
    name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE) 
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)  
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)
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

    if not any([name_match, email_match, skills_match]):
        return " Please provide your name, email, or skills"

    return " ".join(response) + " Let me check what else I need."

def verify_application(_: str) -> str:
    #print(applicant_details.values())
    if all(applicant_details.values()):
        return f" You're ready! Name: {applicant_details['name']}, Email: {applicant_details['email']}, Skills: {applicant_details['skills']}."
    else:
        missing = [k for k, v in applicant_details.items() if not v]
        return f" Still need: {', '.join(missing)}. Please ask the user to provide this."


load_dotenv()

#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = OllamaLLM(
    model = "llama3.2",
    base_url= "http://localhost:11434"
)
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
    verbose=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)

print(" Hi! I'm your job application assistant. Please tell me your name, email, and skills.")

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
        break


