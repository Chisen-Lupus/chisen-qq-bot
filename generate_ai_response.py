# generate_ai_response.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# ---- 可按需修改 ----
BASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
ADAPTER    = 'outputs/qwen7b_lora_mps_run1/checkpoint-200'
DEVICE     = 'cpu'  # 或 'mps'

# ---- 全局: 懒加载，避免每次请求都重新加载模型 ----
_tokenizer = None
_model = None
_eos_id = None
_stopping = None

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids_list):
        super().__init__()
        self.token_ids_list = token_ids_list
    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist()
        for ids in self.token_ids_list:
            L = len(ids)
            if L>0 and seq[-L:]==ids:
                return True
        return False

def _lazy_init():
    global _tokenizer, _model, _eos_id, _stopping
    if _tokenizer is not None and _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)
    _model = PeftModel.from_pretrained(base, ADAPTER)
    _model.eval()

    _eos_id = _tokenizer.convert_tokens_to_ids('<|im_end|>')
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _eos_id

    stop_strings = ['<|im_end|>', '<|im_start|>user', '<|im_start|>system']
    stop_ids = [_tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
    _stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

def generate_ai_response(user_text: str) -> str:
    """
    接收用户文本，返回“赤弦”风格的回复。
    """
    _lazy_init()

    messages = [
        {'role': 'system', 'content': '你现在扮演角色“赤弦”，自称赤弦。'},
        {'role': 'user',   'content': user_text or ''},
    ]

    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(prompt, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=_eos_id,
            pad_token_id=_tokenizer.pad_token_id,
            stopping_criteria=_stopping,
            no_repeat_ngram_size=6,
            renormalize_logits=True,
        )

    gen = out[0][inputs['input_ids'].shape[1]:]
    text = _tokenizer.decode(gen, skip_special_tokens=False)
    reply = text.split('<|im_end|>')[0].strip()
    return reply or '（咳咳…让我再想想怎么回你更有赤弦的味儿~）'
