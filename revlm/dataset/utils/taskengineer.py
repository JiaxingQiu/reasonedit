# TaskEngineer is a class that engineers task-dependent inputs and outputs for a given example.
# ex is a binding to the dict stored in self.data, mutable in place
# (1) prompt 
# (2) gold
#   - label (answer text)
#   - choices (if applicable)
#   - label_letter (if applicable)
# (3) pred
# for mc
#   - label_text: label from model answer text, compared with choices in mc
#   - label_scores: scores of label from model answer text, compared with choices in mc
#   - label_maxprob: max probability of label from model answer text, compared with choices in mc
# for mci
#   - label_text: label from model answer text, compared with choices in mci and mc
#   - letter_text: letter from model answer text 
#   - label_scores: scores of label from model answer text, compared with choices in mci and mc
#   - letter_scores: scores of letter from model answer text 
#   - label_maxprob: max probability of label from model answer text, compared with choices in mci and mc
#   - letter_maxprob: max probability of letter from model answer text 
# for qa
#   - label_text: label from model answer text, compared with choices in qa
#   - label_scores: scores of label from model answer text, compared with choices in qa

import random
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def norm(s):
    return str(s).strip().lower() if s is not None else None

def get_entailment_score(prem: str, hyp: str, tokenizer, model, labels) -> float:
    """Compute entailment score between premise and hypothesis using tokenizer and model."""
    enc = tokenizer(prem, hyp, return_tensors="pt", padding=True, truncation=True, max_length=1280)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    # Find entailment label index
    ent_idx = next((i for i, lbl in enumerate(labels) if lbl.lower().startswith("entail")), 0)
    return float(probs[0, ent_idx])


def _acc_and_cm(y_true_idx, y_pred_idx, num_classes=4):
    """Return (acc, n, cm) given parallel lists of class indices (or None)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    total = 0
    correct = 0
    for gt, pr in zip(y_true_idx, y_pred_idx):
        if gt is None or pr is None:
            continue
        cm[gt, pr] += 1
        total += 1
        if gt == pr:
            correct += 1
    acc = (correct / total) if total > 0 else 0.0
    return acc, total, cm


class TaskIOEngineer:
    def __init__(self):
        self.name = "default"
    def eng_prompt(self, ex):
        raise NotImplementedError("Subclasses must implement eng_prompt")
    def eng_golds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_golds")
    def eng_preds(self, ex):
        raise NotImplementedError("Subclasses must implement eng_preds")


class MCTaskEngineer(TaskIOEngineer):
    def __init__(self, 
                with_rationale=False,
                use_cot=False,
                rationale_in_prompt=True,
                shuffle_choices=False, 
                seed=333,
                **kwargs):
        super().__init__()
        self.name = "mc"
        self.with_rationale = with_rationale
        self.rationale_in_prompt = rationale_in_prompt
        self.shuffle_choices = shuffle_choices
        self.seed = seed
        self.rng = random.Random(seed)
        self.use_cot = use_cot
    

    def _eng_choices(self, s: str):
        # str_choices i.e. "car; person; flower; animal"
        choices_list = s.split(';')
        choices_list = [choice.strip() for choice in choices_list]
        if self.shuffle_choices:
            self.rng.shuffle(choices_list)
        choices_str = '; '.join(choices_list)
        return {"str": choices_str, "ls": choices_list}

    def eng_golds(self, ex):
        ex['gold'] = {}
        ex['gold']['label'] = str(ex['answer']).lower().strip()
        ex['gold']['choices'] = self._eng_choices(ex['choices'])
        # Training target: optionally include rationale instead of putting it in the prompt
        target = ex['gold']['label']
        if self.with_rationale and not self.rationale_in_prompt:
            if self.use_cot:
                ex['gold']['label_train'] = f"{target}. {ex.get('cot','')}".strip()
            else:
                ex['gold']['label_train'] = f"{target}. {ex.get('rationale','')}".strip()
        else:
            ex['gold']['label_train'] = target
    
    def eng_prompt(self, ex):
        # sys_prompt = "Choose the correct answer from the options."
        # base = f"{sys_prompt} {ex['question']} Options: {ex['gold']['choices']['str']}".strip()
        sys_prompt = ""
        base = f"{sys_prompt}{ex['question']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if (self.with_rationale and self.rationale_in_prompt) else base

    def eng_preds(self, ex, a: str, model):
        """ example s: 
        {'cab': {'avg_nll': 3.2672276496887207,
            'sum_nll': 3.2672276496887207,
            'num_tokens': 1,
            'prob': 0.8913333874394938},
        'skateboarder': {'avg_nll': 1.5983691215515137,
            'sum_nll': 6.393476486206055,
            'num_tokens': 4,
            'prob': 0.039113578901274634},...}
        """
        # text-based generation
        ex['pred'] = {}
        ex['pred']['answer'] = a
        # label is the substring in answer(a) that matches any element of ex['gold']['choices']['ls']
        for choice in ex['gold']['choices']['ls']:
            if choice.lower() in a.lower().strip():
                ex['pred']['label_text'] = choice
                break
        # score-based generation
        s = model.score_choices_single(ex['image'], ex['prompt'], ex['gold']['choices']['ls'])
        ex['pred']['label_scores'] = s
        ex['pred']['label_maxprob'] = max(s, key=lambda k: s[k]['prob'])

    def eval(self, vlmdataset):
        """Evaluate MC task using ex['pred'] and ex['gold'].
        Returns dict with accuracies and counts for text extraction and max-prob methods.
        """
        for i in range(min(10, len(vlmdataset.data))):
            print(vlmdataset.data[i])
            print("-"*50)

        y_true_idx = []
        y_pred_text_idx = []
        y_pred_max_idx = []

        def find_choice_index(pred_val, choices_ls):
            pred_norm = norm(pred_val)
            for j, c in enumerate(choices_ls):
                if norm(c) == pred_norm:
                    return j
            return None

        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            g_label = gold.get("label")
            choices_ls = gold.get("choices", {}).get("ls", [])

            if not choices_ls or not g_label:
                continue

            g_idx = find_choice_index(g_label, choices_ls)

            # Text extraction method
            p_label_text = pred.get("label_text")
            p_text_idx = find_choice_index(p_label_text, choices_ls) if p_label_text else None

            # Max-prob method
            p_label_max = pred.get("label_maxprob")
            p_max_idx = find_choice_index(p_label_max, choices_ls) if p_label_max else None

            if g_idx is not None:
                y_true_idx.append(g_idx)
                y_pred_text_idx.append(p_text_idx)
                y_pred_max_idx.append(p_max_idx)

        acc_text, n_text, cm_text = _acc_and_cm(y_true_idx, y_pred_text_idx, num_classes=4)
        acc_max, n_max, cm_max = _acc_and_cm(y_true_idx, y_pred_max_idx, num_classes=4)

        return {
            "text": {"accuracy": acc_text, "n": n_text, "confusion_matrix": cm_text.tolist()},
            "maxprob": {"accuracy": acc_max, "n": n_max, "confusion_matrix": cm_max.tolist()},
        }



class MCITaskEngineer(TaskIOEngineer):
    def __init__(self, 
                with_rationale=False,
                use_cot=False,
                rationale_in_prompt=True,
                shuffle_choices=False,
                unpaired=False,
                seed=333,
                **kwargs):
        super().__init__()
        self.name = "mci"
        self.with_rationale = with_rationale
        self.rationale_in_prompt = rationale_in_prompt
        self.shuffle_choices = shuffle_choices
        self.unpaired = unpaired
        self.seed = seed
        self.rng = random.Random(seed)
        self.use_cot = use_cot

    def extract_choice_pairs(self, s: str):
        pairs = re.findall(r"\(([A-D])\)\s*(.+)", s)
        return [(ltr, txt.strip()) for (ltr, txt) in pairs]

    def get_gold_label_letter(self, label_s:str, idx_choices_s:str):
        pairs = self.extract_choice_pairs(idx_choices_s)
        letter = None
        for ltr, opt in pairs:
            if opt.strip().lower() == label_s.lower():
                letter = ltr
                break
        return letter


    def eng_idx_choices(self, idx_choices_s: str):
        # idx_choises_s : letter indexed choices string like "(A) car\n(B) bike\n(C) train\n(D) bus"
        # return s: 
        # "(D) bus\n\n(B) bike(A) car\n(C) train" if paired shuffle (maintain letter-option association)
        # "(D) car\n(C) bike\n(A) train\n(B) bus" if unpaired shuffle (shuffle both letter and option)
        pairs = self.extract_choice_pairs(idx_choices_s)
        # If shuffle disabled, keep original order
        if not self.shuffle_choices:
            s = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])
            return {"str": s, "ls": pairs}

        if not self.unpaired:  # paired shuffle
            self.rng.shuffle(pairs)
        else:  # unpaired shuffle
            letters = [ltr for ltr, _ in pairs]
            options = [opt for _, opt in pairs]
            self.rng.shuffle(letters)
            self.rng.shuffle(options)
            pairs = list(zip(letters, options))
        s = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])
        return {"str": s, "ls": pairs}
    

    def eng_golds(self, ex):
        ex['gold'] = {}
        ex['gold']['choices'] = self.eng_idx_choices(ex['idx_choices'])
        ex['gold']['label'] = str(ex['answer']).lower().strip()
        ex['gold']['label_letter'] = self.get_gold_label_letter(ex['gold']['label'], ex['gold']['choices']['str'])
        # Training target: include letter prefix when available, e.g., "(A) car"
        base_target = f"({ex['gold']['label_letter']}) {ex['gold']['label']}" if ex['gold']['label_letter'] else ex['gold']['label']
        # Optionally append rationale to target (when not injecting it into the prompt)
        if self.with_rationale and not self.rationale_in_prompt:
            if self.use_cot:
                ex['gold']['label_train'] = f"{base_target}. {ex.get('cot','')}".strip()
            else:
                ex['gold']['label_train'] = f"{base_target}. {ex.get('rationale','')}".strip()
        else:
            ex['gold']['label_train'] = base_target

    def eng_prompt(self, ex):
        sys_prompt = "Choose A/B/C/D from the options."
        base = f"{sys_prompt} {ex['question']} Options: {ex['gold']['choices']['str']}".strip()
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if (self.with_rationale and self.rationale_in_prompt) else base

    def eng_preds(self, ex, a: str, model):
        # text-based generation
        ex['pred'] = {}
        ex['pred']['answer'] = a
        ex['pred']['letter_text'] = None
        ex['pred']['label_text'] = None
        # label is the substring in answer(a) that matches any element of ex['gold']['choices']['ls']
        for (ltr, _) in ex['gold']['choices']['ls']:
            if ltr.lower() in a.lower().strip():
                ex['pred']['letter_text'] = ltr
                break
        for (_, choice) in ex['gold']['choices']['ls']:
            if choice.lower() in a.lower().strip():
                ex['pred']['label_text'] = choice
                break
        
        # score-based generation
        label_texts = [choice for _, choice in ex['gold']['choices']['ls']]
        label_letters = [ltr for ltr, _ in ex['gold']['choices']['ls']]
        ex['pred']['label_scores'] = model.score_choices_single(ex['image'], ex['prompt'], label_texts)
        ex['pred']['letter_scores'] = model.score_choices_single(ex['image'], ex['prompt'], label_letters)
        ex['pred']['label_maxprob'] = max(ex['pred']['label_scores'], key=lambda k: ex['pred']['label_scores'][k]['prob'])
        ex['pred']['letter_maxprob'] = max(ex['pred']['letter_scores'], key=lambda k: ex['pred']['letter_scores'][k]['prob'])
        
    def eval(self, vlmdataset):
        """Evaluate MCI task using ex['pred'] and ex['gold'].
        Returns dict with accuracies and confusion matrices for text/letter extraction and max-prob methods.
        """
        for i in range(10):
            print(vlmdataset.data[i])
            print("-"*50)
        
        letters = ["A", "B", "C", "D"]
        letter_to_idx = {c: i for i, c in enumerate(letters)}

        y_true_letter_idx = []
        y_pred_letter_text_idx = []
        y_pred_letter_max_idx = []
        y_true_label_idx = []
        y_pred_label_text_idx = []
        y_pred_label_max_idx = []

        def find_choice_index(pred_val, choices_ls):
            pred_norm = norm(pred_val)
            for j, c in enumerate(choices_ls):
                if norm(c) == pred_norm:
                    return j
            return None

        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            g_label = gold.get("label")
            g_letter = gold.get("label_letter")
            choices_ls = gold.get("choices", {}).get("ls", [])

            if not choices_ls:
                continue

            # Gold indices
            g_letter_idx = letter_to_idx.get(str(g_letter).upper()) if g_letter else None
            g_label_idx = None
            if g_label:
                # Extract option texts from (letter, text) pairs
                label_texts = [c[1] if isinstance(c, (tuple, list)) and len(c) >= 2 else c for c in choices_ls]
                g_label_idx = find_choice_index(g_label, label_texts)

            # Letter predictions
            p_letter_text = pred.get("letter_text")
            p_letter_text_idx = letter_to_idx.get(str(p_letter_text).upper()) if p_letter_text else None

            p_letter_max = pred.get("letter_maxprob")
            p_letter_max_idx = letter_to_idx.get(str(p_letter_max).upper()) if p_letter_max else None

            # Label predictions
            p_label_text = pred.get("label_text")
            p_label_text_idx = None
            p_label_max_idx = None
            if g_label_idx is not None:
                label_texts = [c[1] if isinstance(c, (tuple, list)) and len(c) >= 2 else c for c in choices_ls]
                p_label_text_idx = find_choice_index(p_label_text, label_texts) if p_label_text else None
                p_label_max = pred.get("label_maxprob")
                p_label_max_idx = find_choice_index(p_label_max, label_texts) if p_label_max else None

            # Collect
            if g_letter_idx is not None:
                y_true_letter_idx.append(g_letter_idx)
                y_pred_letter_text_idx.append(p_letter_text_idx)
                y_pred_letter_max_idx.append(p_letter_max_idx)

            if g_label_idx is not None:
                y_true_label_idx.append(g_label_idx)
                y_pred_label_text_idx.append(p_label_text_idx)
                y_pred_label_max_idx.append(p_label_max_idx)

        # Letter metrics
        acc_letter_text, n_letter_text, cm_letter_text = _acc_and_cm(y_true_letter_idx, y_pred_letter_text_idx)
        acc_letter_max, n_letter_max, cm_letter_max = _acc_and_cm(y_true_letter_idx, y_pred_letter_max_idx)

        # Label metrics
        acc_label_text, n_label_text, cm_label_text = _acc_and_cm(y_true_label_idx, y_pred_label_text_idx)
        acc_label_max, n_label_max, cm_label_max = _acc_and_cm(y_true_label_idx, y_pred_label_max_idx)

        return {
            "letter_text": {"accuracy": acc_letter_text, "n": n_letter_text, "confusion_matrix": cm_letter_text.tolist()},
            "letter_maxprob": {"accuracy": acc_letter_max, "n": n_letter_max, "confusion_matrix": cm_letter_max.tolist()},
            "label_text": {"accuracy": acc_label_text, "n": n_label_text, "confusion_matrix": cm_label_text.tolist()},
            "label_maxprob": {"accuracy": acc_label_max, "n": n_label_max, "confusion_matrix": cm_label_max.tolist()},
        }



class QATaskEngineer(TaskIOEngineer):
    def __init__(self, 
                 with_rationale=False, 
                 use_cot=False,
                 rationale_in_prompt=True, 
                 **kwargs):
        super().__init__()
        self.with_rationale = with_rationale
        self.rationale_in_prompt = rationale_in_prompt
        self.name = "qa"
        self.seed = 333
        self.use_cot = use_cot

    def eng_golds(self, ex): 
        ex['gold'] = {}
        ex['gold']['label'] = str(ex['answer']).lower().strip()
        target = ex['gold']['label']
        if self.with_rationale and not self.rationale_in_prompt:
            if self.use_cot:
                ex['gold']['label_train'] = f"{target}. {ex.get('cot','')}".strip()
            else:
                ex['gold']['label_train'] = f"{target}. {ex.get('rationale','')}".strip()
        else:
            ex['gold']['label_train'] = target
    
    def eng_prompt(self, ex):
        sys_prompt = "Answer the question in one word or phrase."
        base = f"{sys_prompt} {ex['question']}".strip()     
        ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if self.with_rationale and self.rationale_in_prompt else base
    
    def eng_preds(self, ex, a: str, model):
        ex['pred'] = {}
        ex['pred']['answer'] = a
        ex['pred']['label_text'] = a.lower().strip()
        ex['pred']['label_scores'] = model.score_choices_single(ex['image'], ex['prompt'], [ex['gold']['label']])
        

    def eval(self, vlmdataset):
        """Evaluate QA task using ex['pred'] and ex['gold'].
        Returns dict with label-based accuracy (substring/text matching) and bidirectional NLI.
        """
        for i in range(10):
            print(vlmdataset.data[i])
            print("-"*50)
        
        # Load NLI model
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        model_name = "tasksource/deberta-base-long-nli"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        id2label = model.config.id2label
        labels = [lbl for _, lbl in sorted(id2label.items())]
        
        # accuracy
        label_hit = label_total = 0
        fwd_sum = bwd_sum = bi_cnt = 0.0
        
        for ex in vlmdataset.data:
            gold = ex.get("gold", {})
            pred = ex.get("pred", {})
            g_label = gold.get("label")
            p_label = pred.get("label_text")
            if g_label is None or p_label is None:
                continue
            label_total += 1
            if norm(g_label) == norm(p_label):
                label_hit += 1
            
            # Bidirectional NLI: pred -> gold and gold -> pred
            s_fwd = get_entailment_score(p_label, g_label, tokenizer, model, labels)
            s_bwd = get_entailment_score(g_label, p_label, tokenizer, model, labels)
            fwd_sum += s_fwd
            bwd_sum += s_bwd
            if s_fwd >= 0.5 and s_bwd >= 0.5:
                bi_cnt += 1
        
        acc = (label_hit / label_total) if label_total > 0 else 0.0
        
        
        return {
            "text": {"accuracy": acc, "n": label_total},
            "nli": {
                "pred_to_gold": fwd_sum / label_total if label_total > 0 else 0.0,
                "gold_to_pred": bwd_sum / label_total if label_total > 0 else 0.0,
                "bi_frac": bi_cnt / label_total if label_total > 0 else 0.0,
            },
        }

    

def get_taskengineer(task: str, **kwargs):
    task = task.lower()
    if task == "mc":
        return MCTaskEngineer(**kwargs)
    elif task == "mci":
        return MCITaskEngineer(**kwargs)
    elif task == "qa":
        return QATaskEngineer(**kwargs)
    else:
        raise ValueError(f"Unknown task engineer: {task}")