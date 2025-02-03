# import streamlit as st
# import psycopg2
# from passlib.hash import pbkdf2_sha256

# # Database connection parameters
# db_params = {
#     'dbname': 'stock',
#     'user': 'postgres',
#     'password': 'vedant',
#     'host': 'localhost',
#     'port': '5432',
# }

# # Function to authenticate user
# def authenticate_user(username, password):
#     try:
#         # Connect to the PostgreSQL database
#         conn = psycopg2.connect(**db_params)
#         cursor = conn.cursor()

#         # Check if the username exists in the users table
#         cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
#         user_data = cursor.fetchone()

#         if user_data:
#             # Verify the password
#             hashed_password = user_data[2]  # Assuming password is stored in the third column
#             if pbkdf2_sha256.verify(password, hashed_password):
#                 return True
#             else:
#                 return False
#         else:
#             return False
#     except Exception as e:
#         st.error(f"Error: {e}")
#     finally:
#         # Close the database connection
#         if conn:
#             conn.close()

# # Streamlit app
# def main():
#     st.title("Sign In Page")

#     # Input fields for username and password
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     # Login button
#     if st.button("Sign In"):
#         if username and password:
#             if authenticate_user(username, password):
#                 st.success("Sign successful!")
#                 st.session_state.logged_in = True
#                 #st.sidebar.success("Select an option")
#                 # Redirect to the dashboard.py
#                 # You can use Streamlit's st.experimental_rerun to rerun the app with a different script
#                 # st.experimental_rerun(script_runner="dashboard.py")
#             else:
#                 st.error("Invalid username or password")
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False

# if __name__ == "__main__":
#     main()


import streamlit as st
import psycopg2
from passlib.hash import pbkdf2_sha256

# Database connection parameters
db_params = {
    'dbname': 'stock',
    'user': 'postgres',
    'password': 'vedant',
    'host': 'localhost',
    'port': '5432',
}

# Function to authenticate user
def authenticate_user(username, password):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Check if the username exists in the users table
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        if user_data:
            # Verify the password
            hashed_password = user_data[2]  # Assuming password is stored in the third column
            if pbkdf2_sha256.verify(password, hashed_password):
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        # Close the database connection
        if conn:
            conn.close()

# Decorator function to check if user is logged in
def login_required(func):
    def wrapper(*args, **kwargs):
        if "logged_in" in st.session_state and st.session_state.logged_in:
            return func(*args, **kwargs)
        else:
            st.error("You need to log in first!")
    return wrapper

# Streamlit app
def main():
    st.title("Sign In Page")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Sign In"):
        if username and password:
            if authenticate_user(username, password):
                st.success("Sign successful!")
                st.session_state.logged_in = True
            else:
                st.error("Invalid username or password")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Add a button to access the main functionality
    if st.button("Main Functionality"):
        main_functionality()

@login_required
def main_functionality():
    st.write("Welcome to the main functionality!")

if __name__ == "__main__":
    main()


# import streamlit as st
# import psycopg2
# from passlib.hash import pbkdf2_sha256

# # Database connection parameters
# db_params = {
#     'dbname': 'stock',
#     'user': 'postgres',
#     'password': 'vedant',
#     'host': 'localhost',
#     'port': '5432',
# }

# # Function to authenticate user
# def authenticate_user(username, password):
#     try:
#         # Connect to the PostgreSQL database
#         conn = psycopg2.connect(**db_params)
#         cursor = conn.cursor()

#         # Check if the username exists in the users table
#         cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
#         user_data = cursor.fetchone()

#         if user_data:
#             # Verify the password
#             hashed_password = user_data[2]  # Assuming password is stored in the third column
#             if pbkdf2_sha256.verify(password, hashed_password):
#                 return True
#             else:
#                 return False
#         else:
#             return False
#     except Exception as e:
#         st.error(f"Error: {e}")
#     finally:
#         # Close the database connection
#         if conn:
#             conn.close()

# # Decorator function to check if user is logged in
# def login_required(func):
#     def wrapper(*args, **kwargs):
#         if "logged_in" in st.session_state and st.session_state.logged_in:
#             return func(*args, **kwargs)
#         else:
#             st.error("You need to log in first!")
#     return wrapper

# # Streamlit app
# def main():
#     st.title("Sign In Page")

#     # Input fields for username and password
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")

#     # Login button
#     if st.button("Sign In"):
#         if username and password:
#             if authenticate_user(username, password):
#                 st.experimental_set_query_params(logged_in=True)
#                 st.experimental_rerun()
#             else:
#                 st.error("Invalid username or password")

#     if st.experimental_get_query_params().get("logged_in", False):
#         main_functionality()

# @login_required
# def main_functionality():
#     st.write("Welcome to the main functionality!")

# if __name__ == "__main__":
#     main()
    