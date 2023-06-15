
import gradio as gr #importing gradio to make an UI where we will interact with our Custom ChatBot
from dotenv import load_dotenv #remember we created the .env file so in order to access it we need dotenv
import os #importing os module to access env file key
load_dotenv() #this specific fucntion helps
openai_api_key = os.getenv("OPENAI_API_KEY")
import warnings
warnings.filterwarnings("ignore")
from llama_index import VectorStoreIndex, SimpleDirectoryReader

#to read your all the pdf or txt documents present inside training_documents directory
def read():
    global query_engine
    directory_path = r'.\training_documents'
    documents = SimpleDirectoryReader(directory_path).load_data()  #reads and loads the data for further indexing
    index = VectorStoreIndex.from_documents(documents) #index your data in chunks for a better understanding
    query_engine = index.as_query_engine() #return the answers to your queries with the help of indexed data

# Initialize the history list
history = []


def chat(input_query):
    global history

    # Update the history with the new input
    history.append(input_query)

    # Concatenate the input_query with the history
    full_query = " ".join(history)
    
    #the response to the input query
    response = query_engine.query(full_query) 
    #this is the one line which helps remembering the context to the full conversation which means
    #the Bot can understand your follow up question

    # Update the history with the bot's response
    history.append(str(response))  # Convert bot's response to string
    
    # Create the chat response with the full history
    chat_response = "Conversation History:\n\n"
    for i in range(0, len(history), 2): #whole history in a list, here 2 means step i.e. alternate index (Q1,A1,Q2,A2,...,Qn,An)
        user_utterance = history[i] #here i means start from 0 then 2 then 4 i.e. Q1,Q2,Q3,..,Qn
        bot_utterance = history[i+1] #similarly i+1 means from 1 then 3 then 5 i.e. A1,A2,A3,..,An
        chat_response += f"User: {user_utterance.capitalize()}\n" #to print all the Q1,Q2.. as they are all from the User
        chat_response += f"Bot: {bot_utterance.capitalize()}\n\n" #to print all the A1,A1..as they are all from the Bot

    return chat_response

read() #this will run only once so that we do not train the same data every time users asks the question

# Creating the interface with gradio, gr is the nickname here which we have mentioned in the line 2 as (import gradio as gr)

input = gr.inputs.Textbox(label="Ask your questions?") #the input box

output = gr.outputs.Textbox(label="Chatbot Response") #the output box

interface = gr.Interface(fn=chat, inputs=input, outputs=output, title="Custom ChatGPT Bot") #fn=chat means we are calling our chat function everytime ideally till infinity

# Launch the interface
interface.launch()
