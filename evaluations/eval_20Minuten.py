import json
from metrics import caculate_sari, caculate_bleu, caculate_rouge


def eval(input_sequences, predicted_sequences, ground_truths):
    bleu_1 = caculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = caculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = caculate_rouge(predicted_sequences, ground_truths)
    sari = caculate_sari(input_sequences, predicted_sequences, ground_truths)
    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "sari": sari}
    return evaluation_result
