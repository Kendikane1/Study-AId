from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import streamlit_authenticator as stauth
## Linking between files
from dependancies import sign_up, fetch_users

st.set_page_config( page_title="Study-AId :- Empowering the Future: AI Crafted by Students, For Students!", initial_sidebar_state="collapsed")

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
        
        email, authentication_status, username = Authenticator.login(":green[Login]", "main")
        
        info, info1 = st.columns(2)
        
        if not authentication_status:
            sign_up()
        
        if username:
            if username in usernames:
                if authentication_status:
                    #Let User see app
                    st.sidebar.subheader(f"Welcome {username}")
                    Authenticator.logout("Log Out", "sidebar")
                    
                    st.subheader("Study-AId, your friendly learning AI 💬")
                    st.markdown(
                        """
                        ---
                        Created with 🤍 by the StudyAId team.
                        
                        """
                        
                    )
                    # THE PRODUCT (STUDYAID)
                load_dotenv()

                # upload file
                pdf = st.file_uploader("Upload your PDF", type="pdf")

                # extract the texts of the file
                if pdf is not None:
                    pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # split into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)

                # create embeddings
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

                # show user input
                user_question = st.text_input(
                    "Ask a question about your learning material 🤗")
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)

                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(
                            input_documents=docs, question=user_question)
                        print(cb)

                    st.write(response)
                    st.markdown(
                        """
                        ---
                        Created with 🤍 by the StudyAId team.
                        
                        """
                        
                    )
                    
                elif not authentication_status:
                    with info:
                        st.error("Incorrect Password or Username")
            else:
                with info:
                    st.warning("Username Does Not Exist, Please Sign Up")
                
    except: 
        st.success("PDF content can come from any subject!")

if __name__ == '__main__':
    main()