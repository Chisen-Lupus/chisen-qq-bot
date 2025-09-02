# app.py
import os
import time
import json
import binascii
from datetime import datetime

import requests
from flask import Flask, request, jsonify
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from generate_ai_response import generate_ai_response

import sys

log_path = "/home/yichenliu/chisen-qq-bot/qqbot.log"
sys.stdout = open(log_path, "a", buffering=1)  # 行缓冲
sys.stderr = open(log_path, "a", buffering=1)


# ========= 环境变量 =========
APPID     = os.getenv('QQ_BOT_APPID', '').strip()
APPSECRET = os.getenv('QQ_BOT_APPSECRET', '').strip()  # 机器人密钥(Bot Secret)
if not APPID or not APPSECRET:
    raise SystemExit('缺少环境变量 QQ_BOT_APPID / QQ_BOT_APPSECRET')

app = Flask(__name__)

# ========= AccessToken 缓存 =========
_AT = {"val": None, "exp": 0.0}
def get_access_token() -> str:
    now = time.time()
    if _AT["val"] and now < _AT["exp"] - 60:
        return _AT["val"]
    r = requests.post(
        "https://bots.qq.com/app/getAppAccessToken",
        json={"appId": APPID, "clientSecret": APPSECRET},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    _AT["val"] = data["access_token"]
    _AT["exp"] = now + int(data.get("expires_in", 7000))
    print("[TOKEN] fetched, expires_in=", data.get("expires_in"))
    return _AT["val"]

# ========= 回调地址校验 =========
def _seed_from_secret(secret: str) -> bytes:
    s = (secret or "")
    while len(s) < 32:
        s = (s * 2)[:32]
    return s[:32].encode("utf-8")

def make_validation_signature(plain_token: str, event_ts: str) -> str:
    seed = _seed_from_secret(APPSECRET)
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    msg = (event_ts + plain_token).encode("utf-8")
    sig = sk.sign(msg)
    return binascii.hexlify(sig).decode("ascii")

# ========= 工具：清洗命令 =========
def normalize_command(content: str) -> str:
    """去 @ 标签，去空白，半角化斜杠。"""
    import re
    s = (content or "").strip().replace("／", "/")
    for p in (r"<@!?[0-9A-Za-z_]+>", r"<qqbot-at-user\s+id=\"[^\"]+\"\s*/>"):
        s = re.sub(p, "", s).strip()
    return s

# ========= 发送消息（频道 / 群） =========
def reply_channel_message(channel_id: str, content: str, msg_id: str):
    token = get_access_token()
    url = f"https://api.sgroup.qq.com/channels/{channel_id}/messages"
    headers = {"Authorization": f"QQBot {token}"}
    payload = {"content": content, "msg_id": msg_id}
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[SEND][GUILD]", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()

def reply_group_message(group_openid: str, content: str, msg_id: str):
    token = get_access_token()
    url = f"https://api.sgroup.qq.com/v2/groups/{group_openid}/messages"
    headers = {"Authorization": f"QQBot {token}"}
    payload = {
        "content": content,
        "msg_type": 0,   # 0=文本
        "msg_id": msg_id # 被动回复(5分钟/最多5次)
    }
    r = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[SEND][GROUP]", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()

# ========= 事件处理 =========
def handle_event(envelope: dict):
    t = envelope.get('t')
    d = envelope.get('d', {}) or {}
    print('[EVENT] t=', t)

    # 频道内 @ 机器人
    if t == 'AT_MESSAGE_CREATE':
        content    = d.get('content', '')
        channel_id = d.get('channel_id')
        msg_id     = d.get('id')
        cmd = normalize_command(content)
        print('[EVENT][GUILD] payload:', {'channel_id': channel_id, 'raw': content, 'cmd': cmd})

        # 1) 保留 /今天的日期
        if cmd.startswith('/今天的日期') and channel_id and msg_id:
            now = datetime.now()
            reply_channel_message(channel_id, f'今天是 {now:%Y-%m-%d %H:%M:%S}.', msg_id)
            return

        # 2) 非斜杠指令 => 走 AI 回复
        if channel_id and msg_id and not cmd.startswith('/'):
            try:
                ai_text = generate_ai_response(cmd)
            except Exception as e:
                print('[ERROR] ai_response:', repr(e))
                ai_text = '（赤弦打了个喷嚏，刚刚没接住思路…再说一遍？）'
            reply_channel_message(channel_id, ai_text, msg_id)
        return

    # QQ群内 @ 机器人
    if t == 'GROUP_AT_MESSAGE_CREATE':
        content       = d.get('content', '')
        group_openid  = d.get('group_openid') or d.get('group_id')
        msg_id        = d.get('id')
        cmd = normalize_command(content)
        print('[EVENT][GROUP] payload:', {'group_openid': group_openid, 'raw': content, 'cmd': cmd})

        # 1) 保留 /今天的日期
        if cmd.startswith('/今天的日期') and group_openid and msg_id:
            now = datetime.now()
            reply_group_message(group_openid, f'今天是 {now:%Y-%m-%d %H:%M:%S}.', msg_id)
            return

        # 2) 非斜杠指令 => 走 AI 回复
        if group_openid and msg_id and not cmd.startswith('/'):
            try:
                ai_text = generate_ai_response(cmd)
            except Exception as e:
                print('[ERROR] ai_response:', repr(e))
                ai_text = '（赤弦脑袋暂时短路了…要不你换种说法？）'
            reply_group_message(group_openid, ai_text, msg_id)
        return


# ========= 路由：/qq/webhook（保持与你的反代一致） =========
@app.route("/webhook", methods=["POST"])
def qq_webhook():
    raw = request.get_data(as_text=True)
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        try:
            payload = json.loads(raw) if raw else {}
        except Exception:
            payload = {}

    # 调试输出
    print("[HTTP] headers:", dict(request.headers))
    print("[HTTP] raw:", (raw[:500] + "...") if raw and len(raw) > 500 else raw)

    # 回调校验
    if payload.get("op") == 13:
        d = payload.get("d") or {}
        plain_token = (d.get("plain_token") or "").strip()
        event_ts    = (d.get("event_ts")   or "").strip()
        sig = make_validation_signature(plain_token, event_ts)
        print("[VALIDATION] ok;", event_ts[:8], plain_token[:8], sig[:16], "...")
        return jsonify({"plain_token": plain_token, "signature": sig})

    # 事件
    try:
        handle_event(payload)
    except Exception as e:
        print("[ERROR] handle_event:", repr(e))

    return jsonify({"code": 0})

if __name__ == "__main__":
    # 本地 5000；由 Nginx/宝塔把 https://bot.yliu.fit/qq/webhook 反代到这里
    app.run(host="127.0.0.1", port=5000)
