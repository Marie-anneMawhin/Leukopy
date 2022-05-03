import datapane as dp
import pandas as pd
import plotly.express as px

df_umap = pd.read_csv('./df/UMAP.csv')

fig_distrib = px.histogram(df, x='label',
                           template='plotly_dark',
                           color='origin',
                           color_discrete_sequence=[
                               '#4C78A8',  '#E45756', '#72B7B2']
                           )

umap_plot = px.scatter_3d(
    df_umap, x='UMAP_1', y='UMAP_2', z='UMAP_3',
    color='label', opacity=0.9,
    color_discrete_sequence=px.colors.qualitative.Vivid,
    template='plotly_dark')

umap_plot.update_traces(marker=dict(size=2.5))

umap_plot.update_layout(title='UMAP embedding',
                        legend=dict(itemsizing='constant',
                                    itemwidth=45,
                                    ),
                        margin=dict(l=0, r=0, b=0, t=80),
                        width=800,
                        height=800
                        )

range_plot_x = [df_umap.UMAP_1.min(), df_umap.UMAP_1.max()]
range_plot_y = [df_umap.UMAP_2.min(), df_umap.UMAP_2.max()]
range_plot_z = [df_umap.UMAP_3.min(), df_umap.UMAP_3.max()]

umap_plot.update_scenes(xaxis=dict(range=range_plot_x),
                        yaxis=dict(range=range_plot_y),
                        zaxis=dict(range=range_plot_z))
umap_plot.show()

dp.Report(
    dp.Plot(umap_plot),
).upload(name='figure_umap')



