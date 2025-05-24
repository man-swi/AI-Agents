import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import Ollama
import matplotlib.pyplot as plt
import io

# --- Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="LLM Data Visualization Pipeline") # MOVED HERE

# --- LLM Configuration ---
@st.cache_resource # Cache the LLM connection
def get_llm():
    try:
        # Ensure Ollama server is running
        # You can specify a different model if you have others pulled, e.g., "llama2"
        llm = Ollama(model="mistral", temperature=0)
        # Test with a simple prompt to ensure connection
        llm.invoke("Why is the sky blue?")
        return llm
    except Exception as e:
        # These st.error/st.info calls are now fine because set_page_config has already run
        st.error(f"Failed to connect to Ollama or initialize LLM: {e}")
        st.info("Please ensure Ollama is installed, running, and the 'mistral' model is pulled (`ollama pull mistral`).")
        return None

llm = get_llm() # This call is now AFTER st.set_page_config

# --- Agent Explanation ---
AGENT_EXPLANATION = """
### How the LangChain Pandas Agent Works:

The magic behind this application is a **LangChain Pandas DataFrame Agent**. Here's a simplified breakdown:

1.  **User Input:** You upload a CSV/Excel file and ask a question in natural language (e.g., "What is the average age?", "Plot sales over time").

2.  **Data Context:** The agent is given access to your Pandas DataFrame. It "knows" about the columns, their data types, and a sample of the data (like `df.head()`).

3.  **Prompt Engineering (Behind the Scenes):**
    *   Your question and the DataFrame's information are combined into a carefully crafted prompt for the LLM (Mistral, in this case).
    *   This prompt instructs the LLM to act like a data analyst and generate Python code (using Pandas, Matplotlib, etc.) that can answer your question using the provided DataFrame (typically referred to as `df` in the generated code).

4.  **LLM Code Generation:**
    *   The LLM processes the prompt and generates a snippet of Python code. For example, if you ask "What's the average salary?", it might generate:
        ```python
        print(df['salary'].mean())
        ```
    *   If you ask for a plot, it might generate code like:
        ```python
        import matplotlib.pyplot as plt
        df.plot(kind='bar', x='category', y='value')
        plt.title('Value by Category')
        plt.xlabel('Category')
        plt.ylabel('Value')
        # The agent ensures the plot is available for display
        ```

5.  **Secure Code Execution (Python REPL Tool):**
    *   The agent uses a specialized tool, often a Python REPL (Read-Eval-Print Loop), to execute the generated Python code.
    *   This execution happens in an environment where the DataFrame `df` is accessible.
    *   **Security Note:** While LangChain agents aim for safety, executing LLM-generated code always carries some inherent risk. The tools are generally designed to be sandboxed to an extent, but for highly sensitive data or production systems, more robust sandboxing (e.g., Docker containers) would be necessary. For this demonstration, we rely on LangChain's built-in mechanisms.

6.  **Observation and Iteration (ReAct Pattern):**
    *   The agent observes the output of the executed code (e.g., a number, a table, or a confirmation that a plot was generated).
    *   Sometimes, the LLM might need multiple steps (Thought -> Action -> Observation -> Thought...). This is part of the "ReAct" (Reasoning and Acting) framework many agents use. It might first think about what to do, generate some code, see the result, and then decide on the next step if needed.

7.  **Final Answer/Visualization:**
    *   The agent compiles the result from the code execution into a final answer.
    *   If a plot was generated, Streamlit's `st.pyplot()` command (often implicitly triggered by the agent's execution context) captures and displays the Matplotlib figure.

Essentially, the agent acts as an intermediary, translating your natural language questions into executable code, running it, and then presenting the results back to you.
"""

# --- Streamlit App (Title and Markdown for the main page) ---
st.title("📊 LLM-Powered Data Analysis & Visualization")
st.markdown("Upload your CSV or Excel file, ask questions, and let the LLM generate insights and visualizations!")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls & Info")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    st.markdown("---")
    st.header("Agent Explanation")
    with st.expander("How does this work?", expanded=False):
        st.markdown(AGENT_EXPLANATION)
    st.markdown("---")
    st.info("Note: Ensure Ollama (with Mistral model) is running locally.")
    if not llm: # This check happens after get_llm() is called
        st.warning("LLM not initialized. Please check Ollama setup.")


# --- Main App Logic ---
if uploaded_file is not None and llm is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop() # st.stop() is fine here as set_page_config has run

        st.subheader("📄 Uploaded Data Preview (First 5 rows):")
        st.dataframe(df.head())

        # Initialize agent if not already in session state or if df changes
        # This is important because the agent is created with a specific df
        if 'data_agent' not in st.session_state or 'current_df_name' not in st.session_state or st.session_state.current_df_name != uploaded_file.name:
            with st.spinner("Initializing LLM Agent for your data..."):
                st.session_state.data_agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    agent_type="openai-tools", 
                    verbose=True,
                    allow_dangerous_code=True, 
                    agent_executor_kwargs={"handle_parsing_errors": True}
                )
                st.session_state.current_df_name = uploaded_file.name 
            st.success("Agent initialized and ready!")

        st.subheader("💬 Ask questions about your data:")
        user_question = st.text_input("e.g., 'What is the average sales?', 'Plot sales by region', 'Show me a histogram of age'")

        if st.button("🚀 Get Answer & Visualize"):
            if user_question:
                with st.spinner("🤖 LLM is thinking and generating code..."):
                    try:
                        plt.clf() 
                        response = st.session_state.data_agent.invoke(user_question)
                        answer = response.get('output', "No textual output from agent.")

                        st.subheader("💡 LLM's Answer:")
                        st.markdown(answer)
                        
                        fig = plt.gcf() 
                        if any(ax.has_data() for ax in fig.get_axes()): 
                            st.subheader("📊 Generated Visualization:")
                            st.pyplot(fig)
                        else:
                            if "plot" in answer.lower() or "chart" in answer.lower() or "graph" in answer.lower() or "visualization" in answer.lower():
                                st.info("The LLM mentioned a plot, but it might not have been generated in a way Streamlit could capture directly. Try phrasing your plot request more explicitly, e.g., 'Create a bar chart of X vs Y'.")

                    except Exception as e:
                        st.error(f"An error occurred while processing your question: {e}")
                        st.warning("The LLM might have generated invalid code. Try rephrasing your question or check the data.")
            else:
                st.warning("Please enter a question.")
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
elif uploaded_file is None and llm is not None:
    st.info("☝️ Upload a CSV or Excel file to get started.")
elif llm is None:
    # This st.error is fine; it's for the main page content if LLM failed to initialize earlier
    # but after set_page_config has already run.
    st.error("LLM is not available. Cannot proceed. Please check Ollama setup in the sidebar.")

st.markdown("---")
st.markdown("Built with ❤️ by an AI enthusiast.")