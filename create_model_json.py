import os
import json
from models.sequential import (simple_dense, SequentialSubClass,
                               seq_conv_mnist, seq_conv_cifar,
                               conv_lstm, seq_lstm)
from models.functional.rnn_babi import return_babi_rnn
from models.functional.babi_memnn import return_baabi_memnn
from models.functional.mnist_acgan import return_mnist_acgan


def write_model_to_json(model, filename):
    """Write the model to json filename"""
    if not os.path.exists('./jsons'):
        os.makedirs('./jsons')
    filepath = os.path.join('./jsons', filename)
    model_json_str = model.to_json()
    model_json_dict = json.loads(model_json_str)
    with open(filepath, 'w') as model_json_file:
        json.dump(model_json_dict, model_json_file, indent=4)

if __name__ == "__main__":

    simple_dense_model = simple_dense()
    write_model_to_json(simple_dense_model, 'sequential_dense.json')

    current_model = SequentialSubClass()
    current_model.load_model()
    write_model_to_json(current_model,
                        'sequential_subclass.json')

    current_model = seq_conv_mnist()
    write_model_to_json(current_model, 'sequential_conv_mnist.json')

    current_model = seq_conv_cifar()
    write_model_to_json(current_model, 'sequential_conv_cifar.json')

    current_model = conv_lstm()
    write_model_to_json(current_model, 'sequential_conv_lstm.json')

    current_model = seq_lstm()
    write_model_to_json(current_model, 'sequential_lstm.json')

    current_model = return_babi_rnn()
    write_model_to_json(current_model, 'functional_rnn_babi.json')

    current_model = return_baabi_memnn()
    write_model_to_json(current_model, 'functional_memnn_babi.json')

    current_model = return_mnist_acgan()
    write_model_to_json(current_model, 'function_mnist_acgan.json')
