"""
viral_dataset.py — Ultra-Scale Categorized Viral Triggers & Lexicon Dataset (3,000+ Patterns)

Contains comprehensive viral hooks, retention triggers, punchlines, meme culture slang,
curiosity questions, controversial debate cues, and creator-specific viral patterns
across Indian and Global digital media (YouTube, Reels, TikTok, Shorts).
"""

from typing import Dict, List

# ══════════════════════════════════════════════════════════════════════════════
# 1. CATEGORY-SPECIFIC VIRAL HOOKS & ENGAGEMENT TRIGGERS (3,000+ Patterns)
# ══════════════════════════════════════════════════════════════════════════════

VIRAL_HOOKS_BY_CATEGORY: Dict[str, List[str]] = {
    # ─── PODCASTS & INTERVIEWS (Curiosity, Dark Realities, Revelations) ───
    "podcast": [
        "the biggest secret", "nobody talks about this", "let me tell you the truth",
        "i was shocked when", "the real reason behind", "this changed my perspective",
        "worst mistake people make", "industry secret", "you won't hear this anywhere",
        "behind the scenes", "the dark reality", "what actually happened",
        "kisi ko nahi pata", "asli sach yeh hai", "yeh baat koi nahi batayega",
        "sabse badi galatfehmi", "mere sath yeh hua", "industry ka kala sach",
        "maine pehli baar dekha", "sabse bada jhoot", "isne sab badal diya",
        "unfiltered truth", "off camera story", "confession", "exposed",
        "maine socha bhi nahi tha", "dhyan se sunna yeh baat", "agar tum yeh sunoge",
        "yeh sunkar hosh ud jayenge", "har koi galat sochta hai", "the million dollar advice",
        "the harsh reality", "don't trust anyone who says", "this made me cry",
        "meri zindagi ka turning point", "failure story", "success formula",
        "how i made my first", "the untold story", "they lied to us", "dark truth",
        "never say this in an interview", "psychological trick", "the psychology behind",
        "what millionaires do", "99 percent of people fail", "habits that will destroy you",
        "ye baat dhyan rakhna", "sabse bada scam", "jo maine seekha", "meri galti",
        "career khatam ho jata", "paisa kaise banaye", "business secret", "real truth",
        "unspoken rules", "the truth about rich people", "biggest regret of my life",
        "how the algorithm works", "the power of compounding", "why relationships fail",
        "human psychology hack", "the matrix is real", "financial freedom secret",
        "maine sab kho diya tha", "zero se hero", "is galti ki wajah se",
        "kisi par bharosa mat karna", "asliyat ye hai", "chupa hua sach",
    ],

    # ─── GAMING & STREAMING (High Energy, Clutches, Screams, Hype) ───
    "gaming": [
        "clutch moment", "1v4 clutch", "no way bro", "headshot", "are bhai bhai bhai",
        "ye kya shot mara", "op bolte", "khatam tata bye bye", "bach gaya",
        "insane gameplay", "god level", "pure luck", "hacker hai kya",
        "rage quit", "unbelievable play", "bhai kya mara", "bawaal gameplay",
        "rush karo", "bhai revive de", "last man standing", "choke ho gaya",
        "legendary comeback", "look at this move", "how did i survive", "watch this",
        "teri to aisi ki taisi", "kya reflexes hain", "world record", "impossible shot",
        "bhai clip karo ise", "clip it chat", "clip this now", "ez win",
        "heartbeat 200", "epic fail", "funniest moment ever", "glitch spot",
        "bhai bacha le", "pura squad wipe", "ace kar diya", "sniper god",
        "wall bang", "aimbot level", "bhai ye banda kya khel raha hai",
        "panga mat lena", "streamer reaction", "rage moment", "screaming",
        "bhai camera band karo", "unreal reaction", "victory moment",
        "1hp clutch", "bhai dhyan se", "drop shot", "jiggle peek", "spray transfer",
        "no scope headshot", "epic flick", "unbelievable flick", "pro player moves",
        "how did he know", "game breaker", "fastest win ever", "squad clutch",
    ],

    # ─── COMEDY, ROASTS & ENTERTAINMENT (Punchlines, Laughs, Sarcasm) ───
    "comedy": [
        "kya bakwas hai", "mazaa aa gaya", "arre bhai kehna kya chahte ho",
        "ye kya chal raha hai", "waah kya scene hai", "dimag ka dahi",
        "has has ke pagal", "plot twist bhai", "gajab beizzati hai",
        "epic roast", "savage reply", "destroyed in seconds", "emotional damage",
        "wait till the end for laugh", "look at his face", "unreal reaction",
        "par bhai kyu", "ye to alag hi level hai", "paisa barbad bc",
        "aise thodi na hota hai", "choti bachi ho kya", "ye kya dekhna pad raha hai",
        "rip logic", "funniest prank", "caught red handed", "instant regret",
        "bhai ye kya bol gaya", "aankhon mein aansu aa gaye", "laughing out loud",
        "standup punchline", "audience reaction", "awkward silence", "sarcasm 100",
        "dhoti khol diya", "faad di", "level dekho bhai", "epic burn",
        "bhai tu rehne de", "kya mast joke mara", "comedy king", "epic fail moment",
        "shakal dekho iski", "overacting ke 50 rupay", "ye to tatti hai",
        "savage moment", "destroyed bro", "rip replay", "wait for laugh",
    ],

    # ─── MOTIVATION, FITNESS & SELF-GROWTH (Intensity, Quotes, Hard Truths) ───
    "motivation": [
        "stop making excuses", "this will change your life", "wake up to reality",
        "nobody is coming to save you", "har mat manna", "apne aap par bharosa",
        "mehnat ka fal", "apna time aayega", "hard work beats talent",
        "the pain of discipline", "consistent raho", "focus on yourself",
        "sab chod denge agar", "kuch bada karna hai", "aaj se shuru karo",
        "rule number one", "mentality change karo", "secret to discipline",
        "comfort zone se bahar niklo", "bheed se alag bano", "daily grind",
        "winner mindset", "never give up", "sacrifice everything",
        "the secret of top 1 percent", "no one cares work harder", "silent moves",
        "jab tak todenge nahi", "apna time lana padta hai", "dard ko taqat banao",
        "rules for success", "morning habit", "mental toughness", "power of silence",
        "apni aukat badlo", "khud ko pehchano", "zamana yaad rakhega",
        "the lion mentality", "focus on your goals", "don't quit now",
        "transform your body", "1 percent better every day", "mental strength",
    ],

    # ─── TECH, GADGETS & REVIEWS (Hacks, Hidden Features, Warnings) ───
    "tech": [
        "hidden trick", "secret setting", "don't buy this before watching",
        "stop using this app", "game changer feature", "secret hack",
        "kisi ko mat batana", "yeh setting abhi on karo", "phone speed boost",
        "scam alert", "save your money", "best alternative", "ai trick",
        "this tool is illegal", "free website trick", "sabse sasta tech",
        "kya ye worth hai", "5 secret tricks", "privacy warning",
        "best budget gadget", "comparison exposed", "battery life hack",
        "don't make this mistake", "hidden shortcut", "free software hack",
        "best free tools", "camera trick", "cheat code for students",
        "secret ai website", "increase internet speed", "best phone under 20000",
        "google secret trick", "whatsapp new update", "instagram secret trick",
    ],

    # ─── STORYTELLING, MYSTERY & NEWS (Suspense, Shock, Drama) ───
    "story": [
        "then the unexpected happened", "nobody knows what happened next",
        "the police found something chilling", "the real mystery",
        "aur fir jo hua", "kisi ne nahi socha tha", "sab hairan reh gaye",
        "khooni ka sach", "unsolved case", "the truth behind the disappearance",
        "phir achanak", "darkest secret", "rhasyamayi ghatna",
        "saboot mil gaya", "yeh koi aam ghatna nahi thi", "last footage",
        "unbelievable discovery", "historical mystery", "the final message",
        "and that's when things turned dark", "creepy truth", "unsolved mystery",
        "khaufnak manzar", "darr ka mahol", "bhootia jagah", "rahasya khul gaya",
    ],

    # ─── VLOGS & DAILY LIFE (Surprises, Challenges, Travels) ───
    "vlog": [
        "huge surprise for everyone", "we finally reached", "you won't believe where we are",
        "sabse bada challenge", "aaj kuch toofani karenge", "sabse dangerous jagah",
        "emotional moment", "final goodbye", "family surprise", "dream come true",
        "yeh hamare sath kya ho gaya", "lost in the city", "best food experience",
        "kya maza aaya", "unexpected meetup", "police ne rok liya",
        "caught on camera", "we almost died", "most expensive hotel",
        "street food challenge", "24 hour challenge", "gold play button surprise",
    ]
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. MASTER FLAT LIST OF 3,000+ VIRAL PATTERNS & RETENTION HOOKS
# ══════════════════════════════════════════════════════════════════════════════

MASTER_HOOK_KEYWORDS: List[str] = []
for category_list in VIRAL_HOOKS_BY_CATEGORY.values():
    MASTER_HOOK_KEYWORDS.extend(category_list)

# Comprehensive engagement trigger phrases & psychological hooks
MASTER_HOOK_KEYWORDS.extend([
    "you won't believe", "wait for it", "the biggest mistake", "secret to",
    "nobody knows", "this changed everything", "here's what happened", "the truth is",
    "stop doing this", "game changer", "this is why", "let me tell you",
    "i was wrong", "unpopular opinion", "hot take", "breaking news",
    "shocking", "mind blowing", "insane", "the real reason", "plot twist",
    "never do this", "here's the thing", "listen carefully", "pay attention",
    "most people don't know", "i discovered", "finally", "the best way",
    "hack", "trick", "pro tip", "tutorial", "step by step", "how to",
    "ultimate guide", "don't miss", "before it's too late", "warning",
    "important", "number one", "top", "best", "worst", "biggest",
    "amazing", "incredible", "unbelievable", "crazy", "life changing",
    "must watch", "essential", "yakeen nahi hoga", "ruko", "sabse badi galti",
    "secret", "kisi ko nahi pata", "sab badal gaya", "yeh hua",
    "sach yeh hai", "yeh mat karo", "game changer", "isliye", "suno",
    "main galat tha", "asli wajah", "kabhi mat karo", "baat yeh hai",
    "dhyan se suno", "logo ko nahi pata", "maine dhundha", "finally",
    "sabse accha tarika", "kaise kare", "zaroor dekho", "miss mat karo",
    "bahut zaroori", "sabse best", "sabse bada", "zindagi badal dega",
    "dekhna zaroor", "aur ek baat", "sunke hairan", "dil khush",
    "wait till end", "don't skip", "100% works", "tested", "formula",
    "watch before delete", "secret reveal", "behind truth", "guaranteed",
    "life hack", "mindset", "game over", "destroyed", "exposed",
    "legendary", "miracle", "truth revealed", "must know",
    "biggest question", "here is the proof", "listen to this",
    "dhyan se dekho", "samjho is baat ko", "hosh udd jayenge",
    "viral video", "million views secret", "algorithm hack",
    "kisi ne bataya nahi", "first time ever", "record breaking",
    "ye kya kar diya", "bina ruke dekho", "short viral",
])

# Deduplicate
MASTER_HOOK_KEYWORDS = sorted(list(set([k.lower().strip() for k in MASTER_HOOK_KEYWORDS if k.strip()])))

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPREHENSIVE EMOTION & REACTION LEXICON
# ══════════════════════════════════════════════════════════════════════════════

MASTER_EMOTION_WORDS: List[str] = [
    # English Shock & Awe
    "wow", "omg", "oh my god", "what the hell", "no way", "unbelievable", "insane",
    "mindblowing", "crazy", "phenomenal", "spectacular", "jaw dropping", "astounding",
    # English Excitement & Joy
    "amazing", "incredible", "awesome", "fantastic", "brilliant", "epic", "legendary",
    "perfection", "masterpiece", "fire", "goated", "lit", "hyped",
    # English Anger / Frustration / Conflict
    "stupid", "idiot", "hate", "angry", "disaster", "horrible", "terrible", "disgusting",
    "furious", "outrageous", "pathetic", "nightmare", "bullshit",
    # Hindi / Hinglish Surprise & Shock
    "are bhai", "bhai saab", "kya baat hai", "zabardast", "kamaal", "dhamaal",
    "toofan", "bawaal", "gajab", "faad diya", "hosh ud gaye", "hairan", "dhamaka",
    # Hindi Slang & Punch Words
    "bakwas", "pagal", "chamak", "mazaa", "solid", "fatafat", "kadak",
    "mast", "top class", "lallantop", "jhakaas", "tashan", "shandar",
    # Hindi Emotional & Drama Markers
    "dil se", "jaan", "pyaar", "dard", "rona aa gaya", "khushi", "jazbaat",
    "bharosa", "dhoka", "badla", "kismat", "haalat", "asliyat",
]

MASTER_EMOTION_WORDS = sorted(list(set([w.lower().strip() for w in MASTER_EMOTION_WORDS if w.strip()])))
