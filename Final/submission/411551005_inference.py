"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this [default: ./test/task1/0A0F3XPtOO32ZQ8p.png]
-m MODEL     model file [default: ./val_90.pt]
-s BS       batch size [default: 512]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: prefix_beam_search]
-b BEAM   beam size [default: 20]

"""
from docopt import docopt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import common_config as config
from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
import pandas as pd


def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    ans = []
    for i in range(len(preds)):
        ans.append("".join(preds[i]))
    filename_df = pd.read_csv("sample_submission.csv")
    label_df = pd.DataFrame(ans, columns=['label'])
    combined_df = pd.concat([filename_df["filename"], label_df],axis=1 , ignore_index=True)
    combined_df.to_csv('output1.csv', index=False, header=["filename", "label"])


def main():
    arguments = docopt(__doc__)

    images = arguments['IMAGE']
    reload_checkpoint = arguments['-m']
    batch_size = int(arguments['-s'])
    decode_method = arguments['-d']
    beam_size = int(arguments['-b'])

    img_height = config['img_height']
    img_width = config['img_width']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    predict_dataset = Synth90kDataset("test", mode= "test")

    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
