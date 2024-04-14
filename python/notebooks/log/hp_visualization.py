import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv('combined_log_data.csv')
df['log_Loss'] = np.log(df['Loss'])
colorscale = [[0, 'green'], [0.5, 'yellow'], [1, 'red']]

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['log_Loss'],
                    colorscale = colorscale,
                    cmin = df['log_Loss'].min(),
                    cmax = df['log_Loss'].max(),
                    showscale = True),
        dimensions = [
            dict(range = [df['num_layers'].min(), df['num_layers'].max()],
                 label = 'Number of Layers', values = df['num_layers']),
            dict(tickvals = [i for i in range(len(df['activation_fn'].unique()))],
                 ticktext = df['activation_fn'].unique().tolist(),
                 label = 'Activation Function', values = df['activation_fn'].astype('category').cat.codes),
            dict(range = [df['batch_size'].min(), df['batch_size'].max()],
                 label = 'Batch Size', values = df['batch_size']),
            dict(range = [df['learning_rate'].min(), df['learning_rate'].max()],
                 label = 'Learning Rate', values = df['learning_rate']),
            dict(range = [df['log_Loss'].min(), df['log_Loss'].max()],
                 label = 'Log Loss', values = df['log_Loss'])
        ]
    )
)

fig.update_layout(
    plot_bgcolor = 'white',
    paper_bgcolor = 'white',
    title = 'Hyperparameter Optimization Results'
)

fig.write_html('parallel_coordinates_plot_log_loss.html')
fig.show()
