## Post test 28/1/24
import tempfile
import os
from streamlit_option_menu import option_menu
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
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

## Login - Sign Up
import streamlit_authenticator as stauth
from dependancies import sign_up, fetch_users

from streamlit_lottie import st_lottie
import requests


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
                    
                    def load_lottieurl(url):
                        r = requests.get(url)
                        if r.status_code != 200:
                            return None
                        return r.json()
                    
                    def local_css(file_name):
                        with open(file_name) as f:
                            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


                    local_css("style/style.css")
                    
                    lottie_coding = load_lottieurl("https://lottie.host/ce84f7b8-eda7-4302-8cc8-862e94c9aa3d/F7icUYVDmh.json")
                    
                    with st.container():
                            st.title("Study-AId :- AI Crafted by Students, For Students!")
                            st.subheader("INTRODUCTION")
                            st.write("Study-AId is a platform aimed to maximise the time spent and efficiency of student's learning through Artificial Intelligence integrated features and opportunities. “Study-AId” addresses the critical issue of students potentially misusing AI tools in educational settings, thus leading to compromising academic integrity, misrepresenting their work, or gaining academic merits without actively participating in the learning process. ")
                            st.write("Innovation in Study-AId lies in its unique approach to creating personalised learning environments for students. By analysing learner characteristics and data, the tool adapts various effective learning methods such as active recall, spaced repetition and time-blocking through various on-site features that caters to individual needs.")    
                    ## BACKGROUND
                    with st.container():
                        st.write("---")
                        left_column, right_column = st.columns(2)
                        with left_column:
                            st.subheader("OUR BACKGROUND")
                            st.write("##")
                            st.write(
                                """
                                Study-AId started off as a mere idea to create an AI chatbot to converse with our PDF's for easier learning and personal use. However as selections for our Inter-Foundation Innovation Competition (PIITRAM) was nearby, our thorough research led us to realise that this new and evolving technology could potentially transform the education sector in Malaysia and impact students nationwide.
                                
                                Our newfound motivation came from wanting to spread awareness regarding the potential of incorporating AI technology in the education sector, as Malaysia would be one of the first nations to do so. 
                                
                                Most importantly, we aim to break down the barriers between the effective use of AI and students. We believe in the ability of AI to create a personalised learning environment for students to make full use of their studying sessions and maximise the retention of knowledge from the session.
                                """
                            )
                        with right_column:
                            st_lottie(lottie_coding, height="300", key="coding")
                    ## METHODOLOGY
                    with st.container():
                        st.write("---")
                        left2_column, right2_column = st.columns(2)
                        with left2_column:
                            st.subheader("METHODOLOGY")
                            st.write("##")
                            st.write(
                                """
                                The timeline and progression of Study-AId are the according:
                                
                                January 2nd, 2024
                                - Researching into the back-end development of AI applications (LLM models, Vector databases, Embedding models).
                                - Construct the logical flowchart of the PDFReader.
                                - Code out the PDFReader prototype (single PDF uploads) through tutorials on youtube.
                                
                                January 3rd, 2024
                                - Discussions regarding the contents of our poster for PIITRAM selections (12th January).
                                - Choosing out designs, type of paper and colours for poster and flyers.
                                - Completion of App mock-up.
                                
                                January 7th
                                - Researching on Login/Sign up features and how to connect to a free relational database service.
                                - Completion of first poster and flyers.
                                
                                January 12th 
                                - Exhibition of our product and prototype to PASUM students and lecturers for qualifications.
                                - Successfully qualified for PIITRAM!!
                                
                                January 23rd, 2024
                                - Meeting on potential wide range users testing for collection of user data.
                                - Discussion on new and improved poster design, contents and impact.
                                
                                January 28th, 2024
                                - Completion of finalised Study-AId Website and deployment to internet
                                
                                January 30th - February 1st, 2024 (soon)
                                - User testing for three days to receive feedback and suggestions
                                
                                February 9th, 2024 (soon)
                                - Poster submission for PIITRAM
                                
                                February 24th, 2024 (soon)
                                - D-DAY, Competing in Negeri Sembilan for gold medal (insyaAllah)
                            
                                """
                            )
                    
                    ## FORM
                    with st.container():
                        st.write("---")
                        st.subheader("Send us an email for any enquiries or feedbacks!")
                        st.write("##")

                        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
                        contact_form = """
                        <form action="https://formsubmit.co/arizakml@gmail.com" method="POST">
                            <input type="hidden" name="_captcha" value="false">
                            <input type="text" name="name" placeholder="Your name" required>
                            <input type="email" name="email" placeholder="Your email" required>
                            <textarea name="message" placeholder="Your message here" required></textarea>
                            <button type="submit">Send</button>
                        </form>
                        """
                        left_column, right_column = st.columns(2)
                        with left_column:
                            st.markdown(contact_form, unsafe_allow_html=True)
                        with right_column:
                            st.empty()
                            
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
                                        st.session_state.conversation = get_conversation_chain(vectorstore)
                            
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