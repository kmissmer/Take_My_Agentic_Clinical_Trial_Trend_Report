import streamlit as st
from datetime import datetime
from agents import summary_agent 

def main():
    # Streamlit app title
    st.title('Clinical Trial Trend Detector')

    # User input for first and second month/year
    start_year = st.number_input('Enter the start year:', min_value=2000, max_value=datetime.now().year, value=2020)
    start_month = st.number_input('Enter the start month:', min_value=1, max_value=12, value=1)

    end_year = st.number_input('Enter the end year:', min_value=2000, max_value=datetime.now().year, value=2025)
    end_month = st.number_input('Enter the end month:', min_value=1, max_value=12, value=4)

    # When the user clicks 'Submit'
    if st.button('Submit'):
        # Create an instance of the agent with user inputs
        agent = summary_agent.SummaryAgent(start_year, start_month, end_year, end_month)

        # Execute the full trend detection process
        summary, trend_increases, trend_decreases = agent.execute()

        # Display the result
        st.subheader('Summary')
        st.write(summary)

        st.subheader('Notable Increases:')
        for increase in trend_increases:
            st.write(increase)

        st.subheader('Notable Decreases:')
        if len(trend_decreases) == 0:
            st.write("No significant decreases.")
        else:
            for decrease in trend_decreases:
                st.write(decrease)

if __name__ == "__main__":
    main()
