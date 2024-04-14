import os
import pandas as pd
import re

def extract_data_from_logs(directory):
    data = []
    log_files = [f'log{i}.txt' for i in range(1, 8)]

    for file in log_files:
        full_path = os.path.join(directory, file)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
                iterations = content.split('Finished Training')
                for iteration in iterations[:-1]:
                    hyperparams_match = re.search(r'Iteration \d+(:|, Parameters:)( num_layers=\d+, activation_fn=\w+, batch_size=\d+, learning_rate=\d+\.\d+)', iteration)
                    if hyperparams_match:
                        hyperparams_str = hyperparams_match.group(2).strip()
                        hyperparams_dict = dict(param.split('=') for param in hyperparams_str.split(', '))
                        epochs = re.findall(r'Epoch \d+/\d+\s+Loss: (\d+\.\d+), Training Accuracy: (\d+\.\d+)%\, Test Accuracy: (\d+\.\d+)%', iteration)
                        min_loss = float('inf')
                        min_loss_data = {}
                        for epoch_data in epochs:
                            loss, training_acc, test_acc = map(float, epoch_data)
                            if loss < min_loss:
                                min_loss = loss
                                min_loss_data = {
                                    'Loss': loss,
                                    'Training Accuracy': training_acc,
                                    'Test Accuracy': test_acc
                                }

                        if min_loss_data:
                            row = {
                                'Iteration': file + '_BestEpoch',
                                **hyperparams_dict,
                                **min_loss_data
                            }
                            data.append(row)

    df = pd.DataFrame(data)
    csv_file_path = os.path.join(directory, 'combined_log_data.csv')
    df.to_csv(csv_file_path, index=False)
    
extract_data_from_logs('logs')