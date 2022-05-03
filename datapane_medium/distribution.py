import datapane as dp
import pandas as pd
import plotly.express as px

df = pd.read_csv('./df/PBC_dataset_normal_df_merged.csv')

fig_distrib = px.histogram(df, x='label',
                           template='plotly_dark',
                           color='origin',
                           color_discrete_sequence=[
                               '#4C78A8',  '#E45756', '#72B7B2']
                           )

dp.Report(
    dp.Plot(fig_distrib),
).upload(name='figure_distribution')
