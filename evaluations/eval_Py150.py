import re
import json
from metrics import caculate_fuzz


def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def eval(predicted_sequences, ground_truths):
    outputs = []
    for output in predicted_sequences:
        outputs.append(postprocess(output))
    gts = []
    for gt in ground_truths:
        gts.append(postprocess(gt))

    fuzz = caculate_fuzz(outputs, gts)
    evaluation_result = {"similarity": fuzz}
    
    return evaluation_result
