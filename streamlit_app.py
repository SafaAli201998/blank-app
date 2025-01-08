import streamlit as st
import pandas as pd

# Title and Introduction
st.title("🎈 Interactive Data Dashboard")
st.write(
    "Upload a CSV file to explore data, filter results, and visualize it dynamically. "
    "For more details, visit [docs.streamlit.io](https://docs.streamlit.io/)."
)

# File Upload Section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("Data Preview")
    st.write(df.head())

    # Data Summary
    st.subheader("Data Summary")
    st.write(df.describe())

    # Sidebar for Data Filtering
    with st.sidebar:
        st.subheader("Filter Data")
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select column to filter by", columns)
        selected_value = st.selectbox("Select value", df[selected_column].unique())

    # Filter Data
    filtered_df = df[df[selected_column] == selected_value]
    st.write("Filtered Data:", filtered_df)

    # Data Visualization Section
    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", columns)
    y_column = st.selectbox("Select y-axis column", columns)
    plot_type = st.selectbox("Select Plot Type", ["Line", "Bar"])

    if st.button("Generate Plot"):
        if pd.api.types.is_numeric_dtype(filtered_df[y_column]):
            if plot_type == "Line":
                st.line_chart(filtered_df.set_index(x_column)[y_column])
            elif plot_type == "Bar":
                st.bar_chart(filtered_df.set_index(x_column)[y_column])
        else:
            st.warning(f"Cannot plot {y_column} because it is not numeric.")

    # Download Filtered Data
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name='filtered_data.csv',
        mime='text/csv',
    )

else:
    st.write("Waiting for file upload...")
