"""Helper Functions to do Plotting"""


def generateLayoutBar(col_name):
    """
    Generate a layout object for bar chart
    """
    layout_bar = go.Layout(
        autosize=False,  # auto size the graph? use False if you are specifying the height and width
        width=800,  # height of the figure in pixels
        height=600,  # height of the figure in pixels
        title="Distribution of {} column".format(col_name),  # title of the figure
        # more granular control on the title font
        titlefont=dict(
            family='Courier New, monospace',  # font family
            size=14,  # size of the font
            color='black'  # color of the font
        ),
        # granular control on the axes objects
        xaxis=dict(
            tickfont=dict(
                family='Courier New, monospace',  # font family
                size=14,  # size of ticks displayed on the x axis
                color='black'  # color of the font
            )
        ),
        yaxis=dict(
            #         range=[0,100],
            title='Percentage',
            titlefont=dict(
                size=14,
                color='black'
            ),
            tickfont=dict(
                family='Courier New, monospace',  # font family
                size=14,  # size of ticks displayed on the y axis
                color='black'  # color of the font
            )
        ),
        font=dict(
            family='Courier New, monospace',  # font family
            color="white",  # color of the font
            size=12  # size of the font displayed on the bar
        )
    )
    return layout_bar


def plotBar(dataframe_name, col_name):
    """
    Plot a bar chart for the categorical columns

    Arguments:
    dataframe name
    categorical column name

    Output:
    Plot
    """
    # create a table with value counts
    temp = dataframe_name[col_name].value_counts()
    # creating a Bar chart object of plotly
    data = [go.Bar(
            x=temp.index.astype(str),  # x axis values
            y=np.round(temp.values.astype(float) / temp.values.sum(), 4) * 100,  # y axis values
            text=['{}%'.format(i) for i in np.round(temp.values.astype(float) / temp.values.sum(), 4) * 100],
            # text to be displayed on the bar, we are doing this to display the '%' symbol along with the number on the bar
            textposition='auto',  # specify at which position on the bar the text should appear
            marker=dict(color='#0047AB'),)]  # change color of the bar
    # color used here Cobalt Blue

    layout_bar = generateLayoutBar(col_name=col_name)

    fig = go.Figure(data=data, layout=layout_bar)
    return iplot(fig)
