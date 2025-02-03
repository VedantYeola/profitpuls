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
        with psycopg2.connect(**db_params) as conn:
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
    except psycopg2.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return False

# Function to register a new user
def register_user(username, password):
    try:
        # Hash the password before storing it in the database
        hashed_password = pbkdf2_sha256.hash(password)

        # Connect to the PostgreSQL database
        with psycopg2.connect(**db_params) as conn:
            cursor = conn.cursor()

            # Insert the new user into the users table
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()

            return True
    except psycopg2.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return False

# Streamlit app
def main():
    st.title("Sign Up")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Sign-up button
    if st.button("Sign Up"):
        if username and password:
            if register_user(username, password):
                st.success("Sign-up successful! Please login.")
            else:
                st.error("Failed to Sign up. Please try again.")

if __name__ == "__main__":
    main()
