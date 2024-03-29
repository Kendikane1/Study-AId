import streamlit as st
import streamlit_authenticator as stauth
import datetime
import re
from deta import Deta

DETA_KEY = "c0mgzuc64dp_7hKbHVmjfsndL3KUSmQM8yFcYhJCcj49"

deta = Deta(DETA_KEY)

db = deta.Base("StudyAId_stauth")


def insert_user(email, username, password):
    date_joined = str(datetime.datetime.now())

    return db.put({"key": email, "username": username, "password": password, "date_joined": date_joined})


def fetch_users():
    users = db.fetch()
    return users.items


def get_user_emails():
    user = db.fetch()
    emails = []
    for user in user.items:
        emails.append(user["key"])
    return emails


def get_usernames():
    users = db.fetch()
    usernames = []
    for user in users.items:
        usernames.append(user["username"])
    return usernames


def validate_email(email):
    pattern = "^[a-zA-Z0-9-_]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"

    if re.match(pattern, email):
        return True
    return False


def validate_username(username):
    pattern = "^[a-zA-Z0-9_]*$"
    if re.match(pattern, username):
        return True
    return False


def sign_up():
    with st.form(key="sign up", clear_on_submit=True):
        st.subheader(":green[Sign Up]")
        email = st.text_input(":blue[Email]",
                              placeholder="Enter Your Email")
        username = st.text_input(
            ":blue[Username]", placeholder="Enter Your Username")
        password1 = st.text_input(
            ":blue[Password]", placeholder="Enter Your Password", type="password")
        password2 = st.text_input(
            ":blue[Confirm Password]", placeholder="Confirm Your Password", type="password")

        if email:
            if validate_email(email):
                if email not in get_user_emails():
                    if validate_username(username):
                        if username not in get_usernames():
                            if len(username) >= 2:
                                if len(password1) >= 6:
                                    if password1 == password2:
                                        # Add user to database
                                        hashed_password = stauth.Hasher([password2]).generate()
                                        insert_user(email, username, hashed_password[0])
                                        st.success("Account Created Successfully")
                                        st.balloons()
                                    else:
                                        st.warning("Password Does Not Match")

                                else:
                                    st.warning("Password Is Too Short")

                            else:
                                st.warning("Username Too Short")
                        else:
                            st.warning("Username Already Axists")
                    else:
                        st.warning("Invalid Username")
                else:
                    st.warning("Email Already Exists")
            else:
                st.warning("Invalid Email")

        btn1, btn2, btn3, btn4, btn5 = st.columns(5)

        with btn3:
            st.form_submit_button("Sign Up")

