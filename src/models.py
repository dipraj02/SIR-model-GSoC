import torch
import torch.nn as nn

class SIR_LSTM(nn.Module):
    """
    LSTM Model to estimate SIR parameters (beta, gamma) from noisy time-series data.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=2):
        super(SIR_LSTM, self).__init__()
        
        # 1. LSTM Layer
        # input_size=3 because we have S, I, R
        # hidden_size=64 is the number of features in the hidden state
        # batch_first=True means input shape is (Batch, Seq, Feature)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 # Dropout to prevent overfitting
        )
        
        # 2. Fully Connected Layer (Regressor)
        # Takes the last hidden state and maps it to beta, gamma
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass logic.
        x shape: (Batch_Size, Sequence_Length, 3)
        """
        # Pass input through LSTM
        # out shape: (Batch, Seq, Hidden)
        # hn shape: (Num_Layers, Batch, Hidden) - The final hidden state
        out, (hn, cn) = self.lstm(x)
        
        # We only care about the LAST hidden state (summary of the whole epidemic)
        # We take the hidden state of the last layer
        last_hidden_state = hn[-1, :, :] 
        
        # Pass through the linear layer to predict beta and gamma
        predictions = self.fc(last_hidden_state)
        
        return predictions