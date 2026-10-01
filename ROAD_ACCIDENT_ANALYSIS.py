
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Analysis",
    layout="wide"
)


# ============================================================
# 2. READ CLEANED DATA
# ============================================================

df = pd.read_csv("ROAD_ACCEDENT_CLEANED.csv")


# ============================================================
# 3. DASHBOARD HEADER
# ============================================================

st.title("Road Accident Analysis")

st.markdown(
    """
    ### Project Overview

    This project analyzes road accident casualty data after data cleaning,
    preprocessing, and exploratory data analysis. The dashboard presents
    the final cleaned dataset and the main statistical relationships
    identified during the analysis.
    """
)

st.markdown("---")


# ============================================================
# 4. CREATE PAGES / TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Project Overview",
    "Bivariate Analysis",
    "Multivariate Analysis",
    "Data Explorer"
])


# ============================================================
# TAB 1 — PROJECT OVERVIEW
# ============================================================

with tab1:

    st.header("Project Overview")

    st.markdown(
        """
        The dataset contains casualty-level records from road accidents.
        Data preprocessing was performed to remove duplicate records,
        handle invalid coded values, construct related variables, and
        obtain the final analysis dataset.
        """
    )

    # --------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Casualty Records",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Variables",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.markdown("---")


    # --------------------------------------------------------
    # Column Description
    # --------------------------------------------------------

    st.subheader("Final Dataset Variables")

    column_description = pd.DataFrame({
        "Column": [
            "vehicle_reference",
            "casualty_reference",
            "casualty_sex",
            "casualty_age",
            "age_band_of_casualty",
            "severity",
            "car_passenger",
            "bus_or_coach_passenger",
            "pedestrian_road_maintenance_worker",
            "protection_status",
            "ped_behavior",
            "zone_type"
        ],

        "Description": [
            "Reference code identifying the vehicle associated with the casualty record.",
            "Reference code associated with the casualty record.",
            "Coded sex of the casualty.",
            "Age of the casualty.",
            "Coded age-band category of the casualty.",
            "Coded casualty severity level.",
            "Coded indicator describing the casualty's car passenger status.",
            "Coded indicator describing the casualty's bus or coach passenger status.",
            "Coded indicator describing whether the casualty was a pedestrian road maintenance worker.",
            "Derived protection-status indicator created from casualty class.",
            "Combined variable representing pedestrian location and pedestrian movement.",
            "Combined variable representing casualty home-area type and casualty IMD decile."
        ]
    })

    st.dataframe(
        column_description,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")


    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )

    st.markdown("---")


    # --------------------------------------------------------
    # Distribution Analysis
    # --------------------------------------------------------

    st.subheader("Distribution of Numerical Variables")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # First row
    col1, col2 = st.columns(2)

    with col1:

        selected_num_1 = st.selectbox(
            "Select first variable:",
            numeric_columns,
            key="distribution_1"
        )

        fig_dist_1 = px.histogram(
            df,
            x=selected_num_1,
            nbins=20,
            title=f"Distribution of {selected_num_1}",
            color_discrete_sequence=["#636EFA"]
        )

        fig_dist_1.update_layout(
            xaxis_title=selected_num_1,
            yaxis_title="Frequency"
        )

        st.plotly_chart(
            fig_dist_1,
            use_container_width=True
        )



    st.markdown("---")


    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    st.subheader("Correlation Heatmap")

    corr = df.select_dtypes(
        include=np.number
    ).corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix of Numerical Variables",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    fig_corr.update_layout(
        height=750
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# ============================================================
# TAB 2 — BIVARIATE ANALYSIS
# ============================================================

with tab2:

    st.header("Bivariate Analysis")

    # --------------------------------------------------------
    # Question 1
    # --------------------------------------------------------

    st.subheader(
        "1. Which casualty severity has the highest number of casualties?"
    )

    severity_counts = (
        df["severity"]
        .value_counts()
        .sort_values(ascending=False)
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Casualties"
    ]

    fig1 = px.bar(
        severity_counts,
        x="Severity",
        y="Casualties",
        title="Number of Casualties by Severity",
        text_auto=True,
        color="Casualties",
        color_continuous_scale="Viridis"
    )

    fig1.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Question 2
    # --------------------------------------------------------

    st.subheader(
        "2. What are the age bands with the highest number of casualties?"
    )

    age_band_counts = (
        df["age_band_of_casualty"]
        .value_counts()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    age_band_counts.columns = [
        "Age Band",
        "Casualties"
    ]

    fig2 = px.bar(
        age_band_counts,
        x="Age Band",
        y="Casualties",
        title="Top 10 Age Bands by Number of Casualties",
        text_auto=True,
        color="Casualties",
        color_continuous_scale="Blues"
    )

    fig2.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Question 3
    # --------------------------------------------------------

    st.subheader(
        "3. What are the casualty severity levels with the highest number of casualties by sex?"
    )

    severity_sex = (
        df.groupby(
            [
                "casualty_sex",
                "severity"
            ]
        )
        .size()
        .reset_index(name="Casualties")
        .sort_values(
            "Casualties",
            ascending=False
        )
        .head(5)
    )

    fig3 = px.bar(
        severity_sex,
        x="casualty_sex",
        y="Casualties",
        color="severity",
        title="Casualty Severity by Sex",
        text_auto=True,
        labels={
            "casualty_sex": "Casualty Sex",
            "Casualties": "Number of Casualties",
            "severity": "Severity"
        },
        color_continuous_scale="Magma"
    )

    fig3.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Question 4
    # --------------------------------------------------------

    st.subheader(
        "4. What is the distribution of casualties by protection status?"
    )

    protection_counts = (
        df["protection_status"]
        .value_counts()
        .sort_values(ascending=False)
        .reset_index()
    )

    protection_counts.columns = [
        "Protection Status",
        "Casualties"
    ]

    fig4 = px.bar(
        protection_counts,
        x="Protection Status",
        y="Casualties",
        title="Number of Casualties by Protection Status",
        text_auto=True,
        color="Casualties",
        color_continuous_scale="Sunset"
    )

    fig4.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )


# ============================================================
# TAB 3 — MULTIVARIATE ANALYSIS
# ============================================================

with tab3:

    st.header("Multivariate Analysis")


    # --------------------------------------------------------
    # Question 1
    # --------------------------------------------------------

    st.subheader(
        "1. How does casualty severity vary across age bands and sex?"
    )

    age_severity = (
        df.groupby(
            [
                "age_band_of_casualty",
                "casualty_sex",
                "severity"
            ]
        )
        .size()
        .reset_index(name="Casualties")
    )

    fig5 = px.bar(
        age_severity,
        x="age_band_of_casualty",
        y="Casualties",
        color="severity",
        facet_col="casualty_sex",
        barmode="group",
        title="Casualty Severity across Age Bands and Sex",
        labels={
            "age_band_of_casualty": "Age Band",
            "Casualties": "Number of Casualties",
            "severity": "Severity",
            "casualty_sex": "Sex"
        },
        color_continuous_scale="Viridis"
    )

    fig5.update_layout(
        height=650
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Question 2
    # --------------------------------------------------------

    st.subheader(
        "2. How does casualty severity vary across protection status and sex?"
    )

    protection_severity = (
        df.groupby(
            [
                "protection_status",
                "casualty_sex",
                "severity"
            ]
        )
        .size()
        .reset_index(name="Casualties")
    )

    fig6 = px.bar(
        protection_severity,
        x="protection_status",
        y="Casualties",
        color="severity",
        facet_col="casualty_sex",
        barmode="group",
        title="Casualty Severity across Protection Status and Sex",
        labels={
            "protection_status": "Protection Status",
            "Casualties": "Number of Casualties",
            "severity": "Severity",
            "casualty_sex": "Sex"
        },
        color_continuous_scale="Plasma"
    )

    fig6.update_layout(
        height=600
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )


# ============================================================
# TAB 4 — DATA EXPLORER
# ============================================================

with tab4:

    st.header("Data Explorer")

    st.markdown(
        """
        Use the controls below to explore the final cleaned dataset.
        """
    )

    selected_column = st.selectbox(
        "Select a variable:",
        df.columns,
        key="explorer_column"
    )

    st.subheader(
        f"Distribution of {selected_column}"
    )

    if pd.api.types.is_numeric_dtype(df[selected_column]):

        fig_explorer = px.histogram(
            df,
            x=selected_column,
            nbins=20,
            title=f"Distribution of {selected_column}",
            color_discrete_sequence=["#636EFA"]
        )

        st.plotly_chart(
            fig_explorer,
            use_container_width=True
        )

    else:

        value_counts = (
            df[selected_column]
            .value_counts()
            .head(20)
            .reset_index()
        )

        value_counts.columns = [
            selected_column,
            "Count"
        ]

        fig_explorer = px.bar(
            value_counts,
            x=selected_column,
            y="Count",
            title=f"Top Values of {selected_column}",
            text_auto=True,
            color="Count",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(
            fig_explorer,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Selected Variable Summary")

    if pd.api.types.is_numeric_dtype(df[selected_column]):

        st.dataframe(
            df[selected_column]
            .describe()
            .to_frame()
            .round(2),
            use_container_width=True
        )

    else:

        st.dataframe(
            df[selected_column]
            .value_counts()
            .rename("Count")
            .to_frame(),
            use_container_width=True
        )
