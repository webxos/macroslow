# app.py
# Description: Streamlit dashboard for the Lawmakers Suite 2048-AES. Provides a user-friendly web interface for legal professionals and students to submit research queries, view results, and interact with the MCP server. Integrates with the FastAPI server for secure query processing and displays encrypted outputs.

import streamlit as st
import requests
import json

# Streamlit page configuration
st.set_page_config(page_title="Lawmakers Suite 2048-AES", layout="wide")

# Dashboard title
st.title("Lawmakers Suite 2048-AES: Legal Research Dashboard")

# Input form for legal research queries
st.header("Submit a Legal Research Query")
query = st.text_area("Enter your query (e.g., case law search, contract analysis):", height=150)

# Submit button
if st.button("Submit Query"):
    if query:
        try:
            # Send query to FastAPI server
            response = requests.post("http://localhost:8000/query", json={"text": query})
            response.raise_for_status()
            result = response.json()
            st.success("Query processed successfully!")
            st.write("**Encrypted Query Output (hex):**")
            st.code(result.get("encrypted_query", "No output"))
        except requests.RequestException as e:
            st.error(f"Error connecting to API: {e}")
    else:
        st.warning("Please enter a query.")

# Resource discovery section
st.header("Available Resources")
if st.button("List Available Resources"):
    try:
        response = requests.get("http://localhost:8000/resources")
        response.raise_for_status()
        resources = response.json().get("resources", [])
        st.write("**Available Data Sources:**")
        for resource in resources:
            st.write(f"- {resource}")
    except requests.RequestException as e:
        st.error(f"Error fetching resources: {e}")