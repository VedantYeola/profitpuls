import streamlit as st

def dashboard_page():
    st.title("Dashboard")

    # Display personalized greeting
    st.write("Hii !")

    # You can continue adding more content for your dashboard

# Example usage:
# If you want to test this separately, you can include the following lines at the end
# of the dashboard.py file for standalone testing:

# if __name__ == "__main__":
#     username = "John"  # Replace with the actual authenticated username
#     dashboard_page(username)
