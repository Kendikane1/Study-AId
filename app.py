## Post test 28/1/24
import tempfile
import os
from streamlit_option_menu import option_menu
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from gtts import gTTS
from io import BytesIO

## Pre test 14/1/24
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

## Login - Sign Up
import streamlit_authenticator as stauth
from dependancies import sign_up, fetch_users

# Configurating the page
st.set_page_config( page_title="Study-AId :- AI Crafted by Students, For Students!", initial_sidebar_state="collapsed", layout="wide")

def main():
    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user["key"])
            usernames.append(user["username"])
            passwords.append(user["password"])

        credentials = {"usernames": {}}

        for index in range(len(emails)):
            credentials["usernames"][usernames[index]] = {"name": emails[index], "password": passwords[index]}

        Authenticator = stauth.Authenticate(credentials, cookie_name="Streamlit", key = "abcdef", cookie_expiry_days=4)

        emails, authentication_status, username = Authenticator.login(":green[Login]", "main")

        if not authentication_status:
            sign_up()

        if username:
            if username in usernames:
                if authentication_status:
                    #Let User see app
                    st.sidebar.subheader(f"Welcome {username}")
                    Authenticator.logout("Log Out", "sidebar")
                    
                selected = option_menu(
                menu_title=None,  # required
                options=["HomePage", "Pdf-Reader", "Text2Speech", "Time-Blocking-Calendar"],  # required
                icons=["house", "book-half", "headset", "calendar"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
                )
                ## HOMEPAGE
                if selected == "HomePage":
                    ## INTRODUCTION
                    with st.container():
                            st.title("Study-AId :- AI Crafted by Students, For Students!")
                            st.subheader("INTRODUCTION")
                            st.write("blabalablablablabalbalablabalbalablablablablbalbalablabalbalabalblablablablabalbalablablabalbalabalbalablbalablabalbalbalablbalbalbalbbalalbalbalbalbalablabala")    
                    ## BACKGROUND
                    ## METHODOLOGY
                    ## RESULTS
                    ## CONCLUSION
                    ## FORM
                            
                ## PDFREADER
                if selected == "Pdf-Reader":
                        
                    def get_pdf_text(pdf_docs):
                            text = ""
                            for pdf in pdf_docs:
                                pdf_reader = PdfReader(pdf)
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
                            return text


                    def get_text_chunks(text):
                            text_splitter = CharacterTextSplitter(
                                separator="\n",
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len
                            )
                            chunks = text_splitter.split_text(text)
                            return chunks


                    def get_vectorstore(text_chunks):
                            embeddings = OpenAIEmbeddings()
                            # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
                            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                            return vectorstore


                    def get_conversation_chain(vectorstore):
                            llm = ChatOpenAI()
                            # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

                            memory = ConversationBufferMemory(
                                memory_key='chat_history', return_messages=True)
                            conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                memory=memory
                            )
                            return conversation_chain


                    def handle_userinput(user_question):
                            response = st.session_state.conversation({'question': user_question})
                            st.session_state.chat_history = response['chat_history']

                            for i, message in enumerate(st.session_state.chat_history):
                                if i % 2 == 0:
                                    st.write(user_template.replace(
                                        "{{MSG}}", message.content), unsafe_allow_html=True)
                                else:
                                    st.write(bot_template.replace(
                                        "{{MSG}}", message.content), unsafe_allow_html=True)

                    def main():
                        load_dotenv()
                        st.write(css, unsafe_allow_html=True)

                        if "conversation" not in st.session_state:
                                st.session_state.conversation = None
                        if "chat_history" not in st.session_state:
                                st.session_state.chat_history = None

                        st.header("Chat with multiple PDFs :books:")
                        st.write("Click on the sidebar and upload your learning materials 😊")
                        user_question = st.text_input("Have a conversation with your documents:")
                        if user_question:
                                handle_userinput(user_question)

                        with st.sidebar:
                                st.subheader("Your documents")
                                pdf_docs = st.file_uploader(
                                    "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                                if st.button("Process"):
                                    with st.spinner("Processing"):
                                        # get pdf text
                                        raw_text = get_pdf_text(pdf_docs)

                                        # get the text chunks
                                        text_chunks = get_text_chunks(raw_text)

                                        # create vector store
                                        vectorstore = get_vectorstore(text_chunks)

                                        # create conversation chain
                                        st.session_state.conversation = get_conversation_chain(
                                            vectorstore)
                            
                    if __name__ == '__main__':
                            main()
                
                ## TEXT2SPEECH
                def main():      
                    if selected == "Text2Speech":
                        st.title("Convert your learning materials into audiobooks!")
                        pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
                    
                        if pdf_file:
                            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                                tmp_file.write(pdf_file.getvalue())
                                tmp_file.seek(0)

                                pdf_reader = PdfReader(tmp_file)
                                num_pages = len(pdf_reader.pages)

                                if num_pages > 0:
                                    page_range = st.slider(
                                        "Select a page range:",
                                        min_value=1,
                                        max_value=num_pages,
                                        value=(1, num_pages)
                                    )

                                    page_text = ""
                                    for page_num in range(*page_range):
                                        page_text += pdf_reader.pages[page_num].extract_text()

                                    if page_text:
                                        tts_language = "en" # Set the desired language (e.g., 'en' for English)
                                        tts_text = page_text
                                        tts_speech = gTTS(text=tts_text, lang=tts_language, slow=False)
                                        tts_speech.save("output.mp3")

                                        # Display the audio output
                                        audio_file = open("output.mp3", "rb")
                                        audio_bytes = audio_file.read()
                                        st.audio(audio_bytes, format="audio/mp3")
                                        audio_file.close()

                                        # Remove the audio file
                                        os.remove("output.mp3")
                                    else:
                                        st.write("No text found in the selected page range.")
                                else:
                                    st.write("The PDF file is empty.")
                        else:
                            st.write("Our Text2Speech feature is an adoptive method to advocate towards the various learning preferences and physiological / psychological impairments experienced by students countrywide. (auditory learners, loss of sight, dyslexia, etc)")
                            st.write("More features to be added include : - language changing options - changeable AI voices and accents -")
                if __name__ == '__main__':
                        main()

                ## TBC
                if selected == "Time-Blocking-Calendar":
                        st.title(f"Feature coming soon!")
                        
            else:
                st.warning("Username Does Not Exist, Please Sign Up")
                print("Username does not exist")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()