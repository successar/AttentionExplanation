basic_lstm = {
    'weight_decay' : 1e-5,
    'bsize' : 64,
    'hidden_size' : 128
}

lstm_tanh = dict(**basic_lstm, **{
    'attention' : 'tanh',
    'exp_dirname' : 'lstm+attention(tanh)'
})

lstm_dot_product = dict(**basic_lstm, **{
    'attention' : 'dot',
    'exp_dirname' : 'lstm+attention(dot)'
})