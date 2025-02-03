import streamlit as st
import psycopg2
import pandas as pd


@st.expiremental_singleton
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        return cur.fetchall()
    
rows =run_query("SELECT * from user")