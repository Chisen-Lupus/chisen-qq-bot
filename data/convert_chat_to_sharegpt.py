#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将“QQ/群聊导出样式（时间+昵称(UID)一行，下一行内容）”批量转换为 ShareGPT JSONL。

新增/关键特性：
1) 跳过文件前 7 行（总文件头）；删除空行；支持 --max-lines 仅解析前 N 行。
2) 文本保留极短噪声（如单问号/纯符号），但不保留图片；“去图片”与“去词汇黑名单”在 should_keep_text 中统一处理。
3) 基于时间戳的滑动窗口：为 assistant（目标 UID）的回复挑选上文，在“该目标块首条消息”之前 10 分钟内，取最多 10 条消息（**包括赤弦的历史发言**）。
4) system 默认改为：你现在扮演角色"赤弦",自称赤弦.
5) 每条 JSONL 样本附带 "ts" 字段，为该样本对应目标块首条消息的时间戳（YYYY-MM-DD HH:MM:SS）。
6) 上文中出现的目标 UID（赤弦）统一名字为“赤弦”，忽略原始昵称，防止信息丢失与名称漂移。

用法示例（测试前 1000 行）：
python convert_chat_to_sharegpt.py \
  --input chat_raw.txt \
  --output akaxian_sharegpt.jsonl \
  --target-id 1583438367 \
  --add-system \
  --max-lines 1000
  
minimalist example
  --input L-弦狼实验室.txt \
  --output akaxian_sharegpt.jsonl \
  --target-id 1583438367 \
  --split 0.9
Wrote 32765 train samples to: akaxian_sharegpt.train.jsonl
Wrote 3641 val samples to: akaxian_sharegpt.val.jsonl

"""

import argparse
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Iterable
import random
import unicodedata
import os

# 头部行：时间戳 + 名称(UID)
HEADER_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(.+?)\((\d+)\)\s*$'
)
# 例：2021-08-17 10:33:42 【卷不动了】群精北...(1583438367)
# group(1)=时间戳, group(2)=昵称含群名, group(3)=UID

IMG_PAT = re.compile(r'\[图片\]', flags=re.IGNORECASE)
TIME_FMT = '%Y-%m-%d %H:%M:%S'

# 连续重复 @ 提及（支持名字中含空格/中文/括号/标点），例如：
# "@群精北 时不时提醒大家学习（低浮上） @群精北 时不时提醒大家学习（低浮上）"
# 折叠为 "@群精北 时不时提醒大家学习（低浮上）"
MENTION_DUP_RE = re.compile(
    r'(@\s*(?P<n>[^@]+?\S))'          # 第一次 @名字（名字以非空白结尾，避免捕获尾空格）
    r'(?:\s+@\s*(?P=n))+'             # 后续一个或多个相同 @名字
    r'(?=(?:\s|[,，、;；:：!！\?？]|$))' # 第二个名字后需跟空白/常见分隔/行尾，避免过度匹配
)

def norm_text(s: str) -> str:
    """
    清洗文本（用户名/消息正文都用）：
      - 去掉方括号形式的 U+XXXX 占位（如 [U+202E]）；
      - 去掉 \\uXXXX 文本转义；
      - 去掉真实 Unicode 控制字符（含所有 bidi 控制符）；
      - 收敛多余空白；
      - 折叠重复 @ 提及。
    """
    if not s:
        return ''

    # 1) 删除 "[U+202E]" 这种占位
    s = re.sub(r'\[U\+[0-9A-Fa-f]{4,6}\]', '', s)
    # 2) 删除 "\u202E" 这种转义文本
    s = re.sub(r'\\u[0-9A-Fa-f]{4}', '', s)

    # 3) 删除真实控制字符
    cleaned = []
    for ch in s:
        cat = unicodedata.category(ch)
        bidi = unicodedata.bidirectional(ch)
        if cat.startswith('C'):
            continue
        if bidi in ('LRE','RLE','LRO','RLO','PDF','LRI','RLI','FSI','PDI'):
            continue
        cleaned.append(ch)
    s = ''.join(cleaned)

    # 4) 收敛多余空白
    s = re.sub(r'\s+', ' ', s).strip()

    # 5) 折叠重复 @
    prev = None
    while prev != s:
        prev = s
        s = MENTION_DUP_RE.sub(r'\1', s)

    return s

def parse_lines(fp: Iterable[str], skip_header_lines: int = 7, max_lines: int = 0) -> Iterable[Dict]:
    """
    解析两行一条的日志：第一行 header，下一行 content（可能为空/缺失）。
    - 跳过文件开头的 skip_header_lines 行（总文件头）。
    - 删除空行（空行直接忽略，不参与配对）。
    - 可选限制仅读取前 max_lines 行（用于测试调试）。

    产出记录：{'ts','dt','name','uid','text'}，按时间顺序。
      ts: 原始字符串时间戳
      dt: datetime 对象（便于时间窗口筛选）
      name: 群内显示名称（可能含群名）
      uid: 说话者 UID
      text: 该条消息文本；若内容行缺失，可能为空字符串
    """
    pending_header: Optional[Tuple[str, str, str]] = None  # (ts, name, uid)
    line_idx = 0

    for raw in fp:
        line_idx += 1
        # 若设置 max_lines，超过则停止
        if max_lines and line_idx > max_lines:
            break

        # 跳过总文件头
        if line_idx <= skip_header_lines:
            continue

        line = raw.rstrip('\n')
        if not line.strip():  # 删除空行
            continue

        m = HEADER_RE.match(line)
        if m:
            # 如果有上一个 header 未配对内容，则认为其内容为空，直接产出
            if pending_header is not None:
                ts, name, uid = pending_header
                yield {
                    'ts': ts,
                    'dt': datetime.strptime(ts, TIME_FMT),
                    'name': norm_text(name),   # <<< 清洗用户名
                    'uid': uid,
                    'text': ''
                }
            pending_header = (m.group(1), m.group(2), m.group(3))
            continue

        # 非 header：若有 pending_header，则这是内容；否则视作噪声忽略
        if pending_header is not None:
            ts, name, uid = pending_header
            text = norm_text(line)
            yield {
                'ts': ts,
                'dt': datetime.strptime(ts, TIME_FMT),
                'name': norm_text(name),   # <<< 这里也加
                'uid': uid,
                'text': text
            }
            pending_header = None
        else:
            # 孤立内容行（不应出现），忽略
            continue

    # 文件结尾，如仍有未配对 header，产出空内容
    if pending_header is not None:
        ts, name, uid = pending_header
        yield {
            'ts': ts,
            'dt': datetime.strptime(ts, TIME_FMT),
            'name': name,
            'uid': uid,
            'text': ''
        }

def should_keep_text(s: str, min_chars: int, drop_words: List[str]) -> bool:
    """
    文本过滤（集中处理图片与黑名单）：
      - 空串直接丢弃；
      - 若包含 [图片]（IMG_PAT 命中），丢弃；
      - 若文本等于黑名单词汇之一（去首尾空白后比较），丢弃；
      - 允许保留极短噪声（如单问号/纯符号），除了被黑名单命中的；
      - 对长度阈值以下的文本，按白名单与“纯符号不超过3个”策略放行。
    """
    if s is None:
        return False
    t = s.strip()
    if not t:
        return False

    # 1) 去图片
    if IMG_PAT.search(t):
        return False

    # 2) 去黑名单（完全匹配）
    if drop_words:
        if t in drop_words:
            return False

    # 3) 极短文本策略
    if len(t) < min_chars:
        # 白名单：保留常见情绪/极短符号
        allow_set = {
            '?', '？', '草', 'woc', 'WOC', 'tf', 'Tf', '…', '。', '！', '!', '.', '；', ';', '~'
        }
        if t in allow_set:
            return True
        if all(not ch.isalnum() for ch in t) and len(t) <= 3:
            return True
        return False

    return True

def pack_user_context(
    ctx: List[Dict],
    include_names: bool,
    target_uid: str,
    min_chars: int,
    drop_words: List[str],
    target_display_name: str = '赤弦'
) -> str:
    """
    将若干条消息合并为 human 文本。
    - include_names=True 时使用 '昵称: 内容'；若是目标 UID，则统一昵称为 target_display_name（“赤弦”）。
    - 基于 should_keep_text 再过滤一次。
    """
    lines = []
    for msg in ctx:
        text = msg['text'].strip()
        if not should_keep_text(text, min_chars=min_chars, drop_words=drop_words):
            continue
        if include_names:
            name = target_display_name if msg['uid'] == target_uid else msg['name']
            lines.append(f"{name}: {text}" if text else f"{name}: (空)")
        else:
            if text:
                lines.append(text)
    return '\n'.join(lines).strip()

def pack_assistant_block(
    block: List[Dict],
    min_chars: int,
    drop_words: List[str]
) -> str:
    """
    将一段目标角色的连续发言合并为 gpt 文本。
    若某条文本为空或被 should_keep_text 过滤（如图片/黑名单），会跳过；全部为空则返回空字符串。
    """
    lines = []
    for msg in block:
        t = msg['text'].strip()
        if should_keep_text(t, min_chars=min_chars, drop_words=drop_words):
            lines.append(t)
    return '\n'.join(lines).strip()

def build_sharegpt_samples(
    records: Iterable[Dict],
    target_uid: str,
    include_names_in_user: bool = True,
    add_system: bool = False,
    system_prompt: Optional[str] = None,
    # 时间窗与上限
    time_window_seconds: int = 600,
    max_context_msgs: int = 10,
    min_chars: int = 2,
    drop_words: Optional[List[str]] = None,
) -> Iterable[Dict]:
    """
    将线性消息流组装为 ShareGPT conversations（时间窗版本）：
      - 连续的 target_uid 消息并块 → assistant。
      - gpt 前，抓取“在该目标块首条消息时间戳之前 <= time_window_seconds”的消息（**包括赤弦**），
        最多 max_context_msgs 条，按时间顺序合并为 user；
        若消息来自目标 UID，则在 human 端统一用名字“赤弦”。
      - 每个样本携带 "ts" = 目标块首条消息时间戳。
    """
    if drop_words is None:
        drop_words = []

    recent_msgs: List[Dict] = []            # 缓存近期所有消息（含目标和非目标），供时间窗筛选
    current_target_block: List[Dict] = []    # 连续目标消息块

    def flush_block() -> Optional[Dict]:
        """
        将当前目标块转为一个 ShareGPT 样本，并清空目标块。
        上下文从 recent_msgs 中按时间窗过滤，最多取 max_context_msgs 条。
        """
        nonlocal current_target_block, recent_msgs
        if not current_target_block:
            return None

        # 目标块首条消息时间
        first_dt = current_target_block[0]['dt']
        first_ts = current_target_block[0]['ts']  # 字符串形式，写入样本 "ts"
        window_start = first_dt - timedelta(seconds=time_window_seconds)

        # 过滤出时间窗内的历史消息（严格早于 first_dt）；包含任何 UID
        ctx = [m for m in recent_msgs if window_start <= m['dt'] < first_dt]
        # 只取末尾最多 max_context_msgs 条（保持时间顺序不变）
        if len(ctx) > max_context_msgs:
            ctx = ctx[-max_context_msgs:]

        user_text = pack_user_context(
            ctx,
            include_names=include_names_in_user,
            target_uid=target_uid,
            min_chars=min_chars,
            drop_words=drop_words,
            target_display_name='赤弦'
        )
        assistant_text = pack_assistant_block(
            current_target_block,
            min_chars=min_chars,
            drop_words=drop_words
        )

        # 若两边都为空，跳过
        # if not user_text and not assistant_text:
        if not assistant_text:
            current_target_block = []
            return None
        if not user_text: 
            user_text = '随便说点'

        conversations = []
        if add_system and system_prompt:
            conversations.append({'from': 'system', 'value': system_prompt})
        if user_text:
            conversations.append({'from': 'human', 'value': user_text})
        if assistant_text:
            conversations.append({'from': 'gpt', 'value': assistant_text})

        sample = {
            'ts': first_ts,                 # <<< 为每条样本添加时间戳
            'conversations': conversations
        }
        current_target_block = []
        return sample

    for msg in records:
        # 将消息推进总缓冲，用于“时间窗上文”筛选
        recent_msgs.append(msg)

        # 用一个滚动清理，防止 recent_msgs 无限制增长：
        # 以当前消息时间为参照，清理早于当前 60 分钟前的历史
        cutoff = msg['dt'] - timedelta(minutes=60)
        if recent_msgs and recent_msgs[0]['dt'] < cutoff:
            i = 0
            while i < len(recent_msgs) and recent_msgs[i]['dt'] < cutoff:
                i += 1
            recent_msgs = recent_msgs[i:]

        # 目标块累计与切换
        if msg['uid'] == target_uid:
            current_target_block.append(msg)
        else:
            # 遇到非目标：若之前在积累目标块，先产出样本
            if current_target_block:
                sample = flush_block()
                if sample:
                    yield sample

    # 结束时还有目标块
    if current_target_block:
        sample = flush_block()
        if sample:
            yield sample

def write_jsonl(path, samples):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            # 严格 JSONL：单行一个对象，不使用 indent
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

def write_preview(path, samples, indent=2):
    # 仅供人工查看的漂亮版（不是 JSONL）
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False, indent=indent))
            f.write('\n')  # 以空行分隔对象，方便浏览
            
def main():
    parser = argparse.ArgumentParser(
        description='Convert QQ-like chat logs (header+content lines) to ShareGPT JSONL (time-window context, with target history allowed in context).'
    )
    parser.add_argument('--input', required=True, help='输入原始聊天记录 txt')
    parser.add_argument('--output', required=True, help='输出 sharegpt jsonl（或前缀，若 --split>0）')
    parser.add_argument('--target-id', default='1583438367', help='目标角色 UID（assistant）')
    parser.add_argument('--skip-header-lines', type=int, default=7, help='跳过文件开头行数（总文件头），默认 7')
    parser.add_argument('--max-lines', type=int, default=0, help='仅解析前 N 行用于测试（0 表示不限制）')

    # 时间窗与上文控制
    parser.add_argument('--time-window-seconds', type=int, default=600, help='时间窗（秒），默认 600=10 分钟')
    parser.add_argument('--max-context-msgs', type=int, default=10, help='时间窗内最多取多少条消息（含赤弦），默认 10')

    # 文本过滤
    parser.add_argument('--min-chars', type=int, default=2, help='极短文本阈值；低于此长度按白名单/符号策略放行')
    parser.add_argument('--drop-words', type=str, default='', help='逗号分隔的黑名单词汇，完全匹配则丢弃，如："已撤回,广告,无意义"')

    # 组织形式
    parser.add_argument('--no-names', action='store_true', help='human 文本不带昵称前缀（默认带昵称: 内容）')
    parser.add_argument('--add-system', action='store_true', help='每条样本最前加入 system 提示')
    parser.add_argument('--system', default='你现在扮演角色"赤弦",自称赤弦.',
                        help='system 提示内容（默认已按需求修改）')

    # 采样条数限制（写出前 N 条样本；与 max-lines 不同）
    parser.add_argument('--max-samples', type=int, default=0, help='最多输出多少条样本，0 表示不限制')

    # 新增：随机拆分

    parser.add_argument('--split', type=float, default=0.0,
                        help='按比例拆分 train/val，例如 0.9 表示 90%/10%；0 表示不拆分')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于 train/val 拆分')
    parser.add_argument('--preview', action='store_true',
                        help='额外输出一个 .preview.json（漂亮版，仅供人工检查，不用于训练）')

    args = parser.parse_args()
    include_names_in_user = not args.no_names

    # 解析黑名单
    drop_words = []
    if args.drop_words.strip():
        drop_words = [w.strip() for w in args.drop_words.split(',') if w.strip()]

    with open(args.input, 'r', encoding='utf-8') as f_in:
        rec_iter = parse_lines(
            f_in,
            skip_header_lines=args.skip_header_lines,
            max_lines=args.max_lines
        )
        samp_iter = build_sharegpt_samples(
            rec_iter,
            target_uid=args.target_id,
            include_names_in_user=include_names_in_user,
            add_system=args.add_system,
            system_prompt=args.system,
            time_window_seconds=args.time_window_seconds,
            max_context_msgs=args.max_context_msgs,
            min_chars=args.min_chars,
            drop_words=drop_words,
        )

        # 收集样本并在 human 端添加 value_lines（assistant 不加）
        samples = []
        for sample in samp_iter:
            for conv in sample['conversations']:
                if isinstance(conv.get('value'), str):
                    lines = conv['value'].split('\n')
                    if conv.get('from') == 'human' and len(lines) > 1:
                        conv['value_lines'] = lines  # 仅 human 添加
                        # 原 value 里可以保持换行（\n 字符）；不要用 indent 导致多行输出
                        conv['value'] = '\n'.join(lines)
                    else:
                        conv['value'] = '\n'.join(lines)
            samples.append(sample)
            if args.max_samples and len(samples) >= args.max_samples:
                break

    # 随机拆分（可复现）
    if args.split and 0 < args.split < 1:
        random.seed(args.seed)
        random.shuffle(samples)
        k = int(len(samples) * args.split)
        train_samples, val_samples = samples[:k], samples[k:]

        train_path = args.output + '.train.jsonl'
        val_path   = args.output + '.val.jsonl'
        write_jsonl(train_path, train_samples)
        write_jsonl(val_path,   val_samples)
        print(f'Wrote {len(train_samples)} train to {train_path}')
        print(f'Wrote {len(val_samples)} val   to {val_path}')

        if args.preview:
            write_preview(args.output + '.train.preview.json', train_samples)
            write_preview(args.output + '.val.preview.json',   val_samples)
            print('Preview files written (pretty JSON, not for training).')
    else:
        # 不拆分：单文件严格 JSONL
        out_path = args.output + '.jsonl'
        write_jsonl(out_path, samples)
        print(f'Wrote {len(samples)} samples to {out_path}')
        if args.preview:
            write_preview(args.output + '.preview.json', samples)
            print('Preview file written (pretty JSON, not for training).')

        
if __name__ == '__main__':
    main()
