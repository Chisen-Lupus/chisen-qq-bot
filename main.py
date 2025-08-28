# main.py
import os
import re
import datetime as dt
import pytz
import botpy
from botpy.types.message import Message

APPID = os.getenv('QQ_BOT_APPID')
APPSECRET = os.getenv('QQ_BOT_APPSECRET')
DEFAULT_TZ = os.getenv('BOT_TZ', 'Asia/Shanghai')

# 兼容旧/新两种 @ 标签
AT_PATTERNS = [
    r'<@!?[0-9A-Za-z_]+>',
    r'<qqbot-at-user\s+id="[^"]+"\s*/>',
]

def normalize_command(s: str) -> str:
    """去掉 @ 机器人标签与多余空白，并把全角斜杠替换为半角。"""
    s = (s or '').strip().replace('／', '/')
    for p in AT_PATTERNS:
        s = re.sub(p, '', s).strip()
    return s

def is_date_cmd(content: str) -> bool:
    return normalize_command(content).startswith('/今天的日期')

def make_reply(tzname: str=DEFAULT_TZ) -> str:
    tz = pytz.timezone(tzname)
    now = dt.datetime.now(tz)
    return f'今天是 {now:%Y-%m-%d %H:%M:%S}（{now.tzname()}）'

class MyClient(botpy.Client):
    
    async def on_ready(self):
        # 启动后列出能看到的频道与子频道
        try:
            guilds = await self.api.me_guilds()         # 机器人加入的频道列表
            print('[READY] joined guilds =', [g.id for g in guilds])
            for g in guilds:
                chans = await self.api.get_channels(g.id)
                print(f'[READY] guild {g.id} channels =', [(c.id, c.name, c.type) for c in chans])
        except Exception as e:
            print('[READY][ERROR]', e)
    
    async def on_at_message_create(self, message: Message):
        # 频道里被 @ 的文本消息回调（最常用）
        print('[AT_MESSAGE_CREATE] raw =', repr(message.content))
        if is_date_cmd(message.content):
            try:
                # 被动回复：带 msg_id，成功率更稳
                await self.api.post_message(channel_id=message.channel_id,
                                            content=make_reply(),
                                            msg_id=message.id)
            except Exception as e:
                print('[AT_MESSAGE_CREATE][ERROR]', e)

    async def on_message_create(self, message: Message):
        # 某些场景可收到普通消息；是否能触达取决于权限与场景
        print('[MESSAGE_CREATE] raw =', repr(message.content))
        if is_date_cmd(message.content):
            try:
                await self.api.post_message(channel_id=message.channel_id,
                                            content=make_reply(),
                                            msg_id=message.id)
            except Exception as e:
                print('[MESSAGE_CREATE][ERROR]', e)

if __name__ == '__main__':
    # 只监听频道公域消息（含被 @ 的文本事件）
    intents = botpy.Intents(public_guild_messages=True)

    if not APPID or not APPSECRET:
        raise SystemExit('请先设置环境变量 QQ_BOT_APPID 与 QQ_BOT_APPSECRET')

    client = MyClient(intents=intents)
    client.run(appid=APPID, secret=APPSECRET)
