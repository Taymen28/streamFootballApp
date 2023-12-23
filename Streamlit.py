import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


st.title('Player Stats Explorer')

col1, col2 = st.columns(2)
expander = st.expander("Percentile filters")
expanderAtt = st.expander("Stats filters")
# Web scraping of NBA player stats


@st.cache_data
def filedownload():
    Players = pd.read_csv('final_data.csv')

    return Players


Players = filedownload()
df = Players.copy().round(1)

dfs = {'All': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
       'MP', 'G+A',
               'Goals',

               'G/Sh',
               'G/SoT',
               'AvgShotDistance',
               'xG',
               'npxG',
               'npxG/Sh',
               'npG-xG',


               'Assists',
               'xAG',
               'A-xAG',

               'DrbPast', 'G+APer90',
               'GoalsPer90',
               'ShotsPer90',
               'SoT%',
               'SoTPer90',
               'npxGPer90',
               'npG-xGPer90',
               'PassesCompletedPer90',
               'TotCmp%',
               'TotalPassDistPer90',
               'ProgPassDistPer90',
               'ShortPassCmpPer90',
               'MedPassCmpPer90',
               'LongPassCmpPer90',
               'LongPassCmp%',
               'AssistsPer90',
               'xAGPer90',
               'A-xAGPer90',
               'KeyPassesPer90',
               'Final1/3CmpPer90',
               'PenAreaCmpPer90',
               'CrsPenAreaCmpPer90',
               'ProgPassesPer90',
               'ThruBallsPer90',
               'SwitchesPer90',
               'SCAPer90',
               'SCAPassLivePer90',
               'SCADribPer90',
               'SCAShPer90',
               'SCADefPer90',
               'GCAPer90',
               'TklPer90',
               'TklWinPossPer90',
               'Def3rdTklPer90',
               'Mid3rdTklPer90',
               'Att3rdTklPer90',
               'DrbTklPer90',
               'DrbPastAttPer90',
               'DrbTkl%',
               'DrbPastPer90',
               'BlocksPer90',
               'IntPer90',
               'Tkl+IntPer90',
               'ClrPer90',
               'ErrPer90',
               'TouchesPer90',
               'DefPenTouchPer90',
               'Def3rdTouchPer90',
               'Mid3rdTouchPer90',
               'Att3rdTouchPer90',
               'AttPenTouchPer90',
               'SuccDrbPer90',
               'DrbSucc%',
               'TimesTackled%',
               'TimesTackledPer90',
               'CarriesPer90',
               'TotalCarryDistancePer90',
               'ProgCarryDistancePer90',
               'ProgCarriesPer90',
               'CarriesToFinalThirdPer90',
               'CarriesToPenAreaPer90',
               'CarryMistakesPer90',
               'DisposesedPer90',
               'ReceivedPassPer90',
               'ProgPassesRecPer90',
               'FlsPer90',
               'FldPer90',
               'OffPer90',
               'RecovPer90',
               'AerialWinsPer90',
               'AerialLossPer90',
               'AerialWin%',
               #    'AvgTeamPoss',
               #    'OppTouches',
               #    'TeamMins',
               #    'TeamTouches90',
               'pAdjTkl+IntPer90',
               #    'pAdjClrPer90',
               #    'pAdjShBlocksPer90',
               #    'pAdjPassBlocksPer90',
               'pAdjIntPer90',
               'pAdjDrbTklPer90',
               'pAdjTklWinPossPer90',
               'pAdjDrbPastPer90',
               'pAdjAerialWinsPer90',
               'pAdjAerialLossPer90',
               'pAdjDrbPastAttPer90',
               'TouchCentrality',
               'Tkl+IntPer600OppTouch',
               'pAdjTouchesPer90',
               'CarriesPer50Touches',
               'ProgCarriesPer50Touches',
               'ProgPassesPer50CmpPasses'
               ],
       'General': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
                   'MP', 'G+A',
                   'Goals',
                   'xG',
                   'npxG',
                   'Assists',
                   'xAG',
                   'G+APer90',
                   'GoalsPer90',
                   'ShotsPer90',
                   'npxGPer90',
                   'npG-xGPer90',
                   'PassesCompletedPer90',
                   'TotCmp%',
                   'AssistsPer90',
                   'xAGPer90',
                   'SCAPer90',
                   'TklPer90',
                   'IntPer90',
                   'Tkl+IntPer90',
                   'TouchesPer90',
                   'SuccDrbPer90',
                   'DrbSucc%',
                   'ProgCarriesPer90',
                   'AerialWinsPer90',
                   'AerialLossPer90',
                   'AerialWin%',
                   'AvgTeamPoss',
                   'OppTouches'],

       'Shooting': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
                    'MP',
                    'Goals',
                    'G/Sh',
                    'G/SoT',
                    'AvgShotDistance',
                    'xG',
                    'npxG',
                    'npxG/Sh',
                    'npG-xG',
                    'GoalsPer90',
                    'ShotsPer90',
                    'SoT%',
                    'SoTPer90',
                    'npxGPer90',
                    'npG-xGPer90'],

       'Passing': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
                   'MP',

                   'Assists',
                   'xAG',
                   'A-xAG',
                   'PassesCompletedPer90',
                   'TotCmp%',
                   'TotalPassDistPer90',
                   'ProgPassDistPer90',
                   'ShortPassCmpPer90',
                   'MedPassCmpPer90',
                   'LongPassCmpPer90',
                   'LongPassCmp%',
                   'AssistsPer90',
                   'xAGPer90',
                   'A-xAGPer90',
                   'KeyPassesPer90',
                   'Final1/3CmpPer90',
                   'PenAreaCmpPer90',
                   'CrsPenAreaCmpPer90',
                   'ProgPassesPer90', 'ProgPassesPer50CmpPasses',
                   'ThruBallsPer90',
                   'SwitchesPer90',
                   'SCAPassLivePer90',

                   ],

       'Dribbling': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
                     'MP',
                     'SCADribPer90',
                     'TouchesPer90',
                     'DefPenTouchPer90',
                     'Def3rdTouchPer90',
                     'Mid3rdTouchPer90',
                     'Att3rdTouchPer90',
                     'AttPenTouchPer90',
                     'SuccDrbPer90',
                     'DrbSucc%',
                     'TimesTackled%',
                     'TimesTackledPer90',
                     'CarriesPer90',
                     'TotalCarryDistancePer90',
                     'ProgCarryDistancePer90',
                     'ProgCarriesPer90',
                     'CarriesToFinalThirdPer90',
                     'CarriesToPenAreaPer90',
                     'CarryMistakesPer90',
                     'DisposesedPer90',
                     'ReceivedPassPer90',
                     'ProgPassesRecPer90', 'TouchCentrality',
                     'Tkl+IntPer600OppTouch',
                     'pAdjTouchesPer90',
                     'CarriesPer50Touches',
                     'ProgCarriesPer50Touches',],

       'Defending': ['Player', 'Main Position', 'Nation', 'Squad', 'Comp', 'Age',
                     'MP',
                     'TklPer90',
                     'TklWinPossPer90',
                     'Def3rdTklPer90',
                     'Mid3rdTklPer90',
                     'Att3rdTklPer90',
                     'DrbTklPer90',
                     'DrbPastAttPer90',
                     'DrbTkl%',
                     'DrbPastPer90',
                     'BlocksPer90',
                     'IntPer90',
                     'Tkl+IntPer90',
                     'ClrPer90',
                     'ErrPer90',

                     'RecovPer90',
                     'AerialWinsPer90',
                     'AerialLossPer90',
                     'AerialWin%',


                     'pAdjTkl+IntPer90',
                     'pAdjClrPer90',
                     'pAdjShBlocksPer90',
                     'pAdjPassBlocksPer90',
                     'pAdjIntPer90',
                     'pAdjDrbTklPer90',
                     'pAdjTklWinPossPer90',
                     'pAdjDrbPastPer90',
                     'pAdjAerialWinsPer90',
                     'pAdjAerialLossPer90',
                     'pAdjDrbPastAttPer90',
                     ]
       }

# Sidebar - Team selection

# competition selector
sorted_unique_comp = sorted(Players.Comp.unique())
selected_comp = st.sidebar.multiselect(
    'Competition', sorted_unique_comp, sorted_unique_comp)

# teams selector
sorted_unique_team = sorted(
    Players[Players['Comp'].isin(selected_comp)].Squad.unique())
selected_team = st.sidebar.multiselect(
    'Team', sorted_unique_team, sorted_unique_team)

# age selector
max_age = st.sidebar.slider(
    'Max Age for filtered players', df.Age.min(), df.Age.max(), df.Age.max())
max_age_all = st.sidebar.slider(
    'Max Age for all Players ',  df.Age.min(), df.Age.max(), df.Age.max())


# 90s selector
min_90s = st.sidebar.slider('Min 90s', df.MP.min(), df.MP.max(), df.MP.min())


# columns selector
st.header('Stats Types')
df_selector = st.radio('Stats type', dfs.keys())


# Sidebar - Position selection
unique_pos = Players['Main Position'].unique()
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)


df_selected_team = Players[(Players.Squad.isin(
    selected_team)) & (Players['Main Position'].isin(selected_pos)) & (Players.Comp.isin(selected_comp))]
df_selected_team = df_selected_team[(df_selected_team.Age < max_age) & (
    df_selected_team['MP'] > min_90s)]
df_selected_team = df_selected_team.fillna(0)


# Filtering data
with st.sidebar:

    st.header('Filtering Percentiles')
    columns_to_filter = ['G+A',
                         'Goals',

                         'G/Sh',
                         'G/SoT',
                         'AvgShotDistance',
                         'xG',
                         'npxG',
                         'npxG/Sh',
                         'npG-xG',


                         'Assists',
                         'xAG',
                         'A-xAG',

                         'DrbPast',



                         'G+APer90',
                         'GoalsPer90',
                         'ShotsPer90',
                         'SoT%',
                         'SoTPer90',
                         'npxGPer90',
                         'npG-xGPer90',
                         'PassesCompletedPer90',
                         'TotCmp%',
                         'TotalPassDistPer90',
                         'ProgPassDistPer90',
                         'ShortPassCmpPer90',
                         'MedPassCmpPer90',
                         'LongPassCmpPer90',
                         'LongPassCmp%',
                         'AssistsPer90',
                         'xAGPer90',
                         'A-xAGPer90',
                         'KeyPassesPer90',
                         'Final1/3CmpPer90',
                         'PenAreaCmpPer90',
                         'CrsPenAreaCmpPer90',
                         'ProgPassesPer90',
                         'ThruBallsPer90',
                         'SwitchesPer90',
                         'SCAPer90',
                         'SCAPassLivePer90',
                         'SCADribPer90',
                         'SCAShPer90',
                         'SCADefPer90',
                         'GCAPer90',
                         'TklPer90',
                         'TklWinPossPer90',
                         'Def3rdTklPer90',
                         'Mid3rdTklPer90',
                         'Att3rdTklPer90',
                         'DrbTklPer90',
                         'DrbPastAttPer90',
                         'DrbTkl%',
                         'DrbPastPer90',
                         'BlocksPer90',
                         'IntPer90',
                         'Tkl+IntPer90',
                         'ClrPer90',
                         'ErrPer90',
                         'TouchesPer90',
                         'DefPenTouchPer90',
                         'Def3rdTouchPer90',
                         'Mid3rdTouchPer90',
                         'Att3rdTouchPer90',
                         'AttPenTouchPer90',
                         'SuccDrbPer90',
                         'DrbSucc%',
                         'TimesTackled%',
                         'TimesTackledPer90',
                         'CarriesPer90',
                         'TotalCarryDistancePer90',
                         'ProgCarryDistancePer90',
                         'ProgCarriesPer90',
                         'CarriesToFinalThirdPer90',
                         'CarriesToPenAreaPer90',
                         'CarryMistakesPer90',
                         'DisposesedPer90',
                         'ReceivedPassPer90',
                         'ProgPassesRecPer90',
                         'FlsPer90',
                         'FldPer90',
                         'OffPer90',
                         'RecovPer90',
                         'AerialWinsPer90',
                         'AerialLossPer90',
                         'AerialWin%',
                         'pAdjTkl+IntPer90',
                         'pAdjClrPer90',
                         'pAdjShBlocksPer90',
                         'pAdjPassBlocksPer90',
                         'pAdjIntPer90',
                         'pAdjDrbTklPer90',
                         'pAdjTklWinPossPer90',
                         'pAdjDrbPastPer90',
                         'pAdjAerialWinsPer90',
                         'pAdjAerialLossPer90',
                         'pAdjDrbPastAttPer90',
                         'TouchCentrality',
                         'Tkl+IntPer600OppTouch',
                         'pAdjTouchesPer90',
                         'CarriesPer50Touches',
                         'ProgCarriesPer50Touches',
                         'ProgPassesPer50CmpPasses']

    # Dictionary to store the selected percentile values for each column
    selected_percentiles = {}

    # Create sliders for each column
    for column in columns_to_filter:
        percentile = st.slider(f'{column} Percentile', 0, 100, key=column)
        selected_percentiles[column] = percentile

    # Apply filters based on selected percentiles
    filtered_df = df_selected_team.copy()  # Make a copy of the original DataFrame

    # Build the filter conditions
    filter_conditions = []
    for column, percentile in selected_percentiles.items():
        column_values = filtered_df[column]
        threshold = column_values.quantile(percentile / 100)
        condition = column_values >= (threshold)
        filter_conditions.append(condition)

    # Combine the filter conditions using "&" (AND) operator
    if filter_conditions:
        combined_condition = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_condition &= condition
        filtered_df = filtered_df[combined_condition]
    else:
        filtered_df = filtered_df[False]


df_col_fil = filtered_df[filtered_df.Age < max_age_all]

df = df_col_fil[dfs[df_selector]]
st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(
    df.shape[1]) + ' columns.')
cm = sns.color_palette("Spectral", as_cmap=True)
grad_select = st.multiselect(
    label='Select Stats to apply gradient', options=df.columns, default=[],)
st.image('download.png', caption='Gradient key')
df = df.drop_duplicates()
df = df.dropna()
dfheat = df
cols_to_move = grad_select
df = df[cols_to_move +
        [col for col in df.columns if ((col not in cols_to_move))]]
df = df.sort_values(by=grad_select,
                    ascending=False)
df = (df.style
      .background_gradient(cmap=cm, subset=grad_select)
      .set_caption('Percentiles').format(precision=1))
st.dataframe(df)


def filedownloader(df):
    csv = df.to_csv(index=False)
    # strings <-> bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href


st.markdown(filedownloader(dfheat), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    dfheat[grad_select].to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)
