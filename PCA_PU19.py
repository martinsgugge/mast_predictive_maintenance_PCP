import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from array_manipulation import remove_on_off, quantile_remove_up


def pca(df, no_component, filename):
    """df = pd.DataFrame(data)

    df.columns = tagnames
    df['meas_time'] = pd.DatetimeIndex(df['meas_time'])
    #print(df.columns)
    #print(df)

    """
    print(df.columns)
    recolor = False
    timeIndex = pd.to_datetime(df['Time']).astype(np.int64)
    if "State" in df.columns:
        colorIndex = df["State"]
        df.drop(columns=["State"], axis=1, inplace=True)
        recolor = True


    features = df.columns[1:]


    df = df.dropna(axis=0)
    #print('scaled')
    #print(df)

    pca = PCA(n_components=no_component)
    components = pca.fit_transform(df[features])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    total_var = pca.explained_variance_ratio_.sum() * 100

    if recolor:
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=colorIndex,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels=labels
        )
        # fig.show()
        plotly.offline.plot(fig, filename='./PCA/PCA score ' + filename + '.html', auto_open=False)
        fig2 = px.scatter(
            components, x=0, y=1, color=colorIndex,
            title=f'PC1, PC2 Total Explained Variance: {total_var:.2f}%',
            labels=labels
        )

        # fig.show()
        plotly.offline.plot(fig2, filename='./PCA/PCA score PC1 PC2' + filename + '.html', auto_open=False)

        fig3 = px.scatter(
            components, x=0, y=2, color=colorIndex,
            title=f'PC1, PC3 Total Explained Variance: {total_var:.2f}%',
            labels=labels
        )
        # fig.show()
        plotly.offline.plot(fig3, filename='./PCA/PCA score PC1 PC3' + filename + '.html', auto_open=False)


    else:
        fig = px.scatter_3d(
            components, x=0, y=1, z=2, color=timeIndex,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels=labels
        )
        # fig.show()
        plotly.offline.plot(fig, filename='./PCA/PCA score ' + filename + '.html', auto_open=False)

    fig4 = px.scatter()
    for i, feature in enumerate(features):
        fig4.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1],

        )
        fig4.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],

            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",

            text=feature
        )
    plotly.offline.plot(fig4, filename='./PCA/PCA loading PC1 PC2' + filename + '.html', auto_open=False)

    fig5 = px.scatter()
    for i, feature in enumerate(features):
        fig5.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 2],

        )
        fig5.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 2],

            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",

            text=feature
        )
    plotly.offline.plot(fig5, filename='./PCA/PCA loading PC1 PC3' + filename + '.html', auto_open=False)

    #fig2.show()
    #df = pd.concat([df, colorIndex], axis=1)
    #df["PU19_State"] = colorIndex.values
    #df = pd.merge_asof(df, colorIndex, on='meas_time')


if __name__ == '__main__':
    extension = 'csv'
    all_datafiles = [i for i in glob.glob('data/*.{}'.format(extension))]
    df = pd.concat([pd.read_csv(f, sep=';', usecols=['Time', 'HYG_FT02', 'HYG_PT15', 'HYG_PU19_PW_PV', 'HYG_PU19_TQ_PV',
                                                     'HYG_PU19_SF_PV', 'HYG_PU19_MO', 'HYG_PT16'], decimal='.',
                                encoding="ISO-8859-1") for f in all_datafiles])
    df['Time'] = pd.DatetimeIndex(df['Time'])

    start = datetime.datetime(2021, 4, 1, tzinfo=datetime.timezone.utc)
    end = datetime.datetime(2021, 5, 1, tzinfo=datetime.timezone.utc)

    mask = (df['Time'] >= start) & (df['Time'] < end)
    df = df.loc[mask]

    features = ['HYG_FT02', 'HYG_PT15', 'HYG_PU19_PW_PV', 'HYG_PU19_TQ_PV', 'HYG_PU19_SF_PV', 'HYG_PU19_MO', 'HYG_PT16']

    fig = px.line(df, x='Time', y=features)
    fig.show()

    timeIndex = pd.to_datetime(df['Time']).astype(np.int64)

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    pca = PCA(n_components=3)
    components = pca.fit_transform(df[features])
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=timeIndex,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels=labels
    )
    fig.show()

    # for i, feature in enumerate(features):
    #     fig.add_shape(
    #         type='line',
    #         x0=0, y0=0,
    #         x1=loadings[i, 0],
    #         y1=loadings[i, 1]
    #     )
    #     fig.add_annotation(
    #         x=loadings[i, 0],
    #         y=loadings[i, 1],
    #         ax=0, ay=0,
    #         xanchor="center",
    #         yanchor="bottom",
    #         text=feature,
    #     )

    # fig.show()