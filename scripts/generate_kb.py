"""
Generates the domain knowledge base for the RAG pipeline.
Run once: python scripts/generate_kb.py
Produces 25 markdown documents (5 countries x 5 topics) under data/knowledge_base/.
Content is original summary text written for this project (not scraped), suitable
for a coursework RAG demo. Replace with authoritative sources for production use.
"""
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")

DOCS = {
    "sri_lanka": {
        "overview.md": """# Sri Lankan Rupee (LKR) — Overview
The Sri Lankan Rupee (code LKR, symbol Rs) is issued by the Central Bank of Sri Lanka (CBSL).
It is subdivided into 100 cents, though coins below Rs.1 are rarely used in circulation today.
Banknotes are issued in denominations of Rs.20, Rs.50, Rs.100, Rs.500, Rs.1000, Rs.5000.
The current note series, introduced from 2015 onward, features Sri Lankan wildlife, landmarks,
and cultural heritage sites unique to each denomination.
""",
        "denominations.md": """# LKR Banknote Denominations
- Rs.20 — Green, features a garden lizard and Kandy heritage motifs.
- Rs.50 — Violet, features Ridiyagama and coastal themes.
- Rs.100 — Crimson, features the Dutch fort of Galle.
- Rs.500 — Orange, features the Ruwanwelisaya stupa in Anuradhapura.
- Rs.1000 — Blue, features Sri Lankan elephants at Minneriya.
- Rs.5000 — Purple, features high-value transactions and Colombo skyline landmarks.
Each note carries a raised-print watermark portrait area and a security thread.
""",
        "security_features.md": """# LKR Security Features
Sri Lankan banknotes issued by CBSL contain: a windowed security thread that shifts color when
tilted, a watermark of a lotus flower visible under light, latent images that reveal denomination
numerals at an angle, micro-lettering in the border pattern, and raised intaglio printing that can
be felt by touch. Ultraviolet-reactive fibres embedded in the paper glow under UV light, which is
a common quick-check method used by banks and merchants.
""",
        "history.md": """# History of the Sri Lankan Rupee
The rupee replaced the Ceylonese pound in 1872 at a rate of 1 pound = 10 rupees. Sri Lanka
established its own central bank in 1950, taking over currency issuance from the earlier
currency board system. Since then, the CBSL has periodically redesigned notes, most recently in
2010-2015, to modernize security features and depict national heritage themes rather than
colonial-era imagery.
""",
        "travel_tips.md": """# Using LKR as a Traveler
Foreign currency can be exchanged at banks, licensed money changers, and airport counters;
banks generally offer better rates than street changers. Rs.5000 notes can be hard to break
for small purchases outside cities. It is illegal to take large amounts of LKR out of Sri Lanka
without declaration. Always count change carefully, as older and newer note designs currently
circulate together and worn notes are still valid legal tender.
""",
    },
    "india": {
        "overview.md": """# Indian Rupee (INR) — Overview
The Indian Rupee (code INR, symbol ₹) is issued by the Reserve Bank of India (RBI). It is
subdivided into 100 paise, though paise coins are now largely out of daily circulation.
Current banknotes belong to the Mahatma Gandhi New Series introduced from 2016, in denominations
of ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 (the ₹2000 note was withdrawn from circulation
in 2023 though it remains valid legal tender for exchange).
""",
        "denominations.md": """# INR Banknote Denominations
- ₹10 — Chocolate brown, depicts the Konark Sun Temple.
- ₹20 — Yellow-green, depicts the Ellora Caves.
- ₹50 — Fluorescent blue, depicts the Hampi chariot.
- ₹100 — Lavender, depicts the Rani ki Vav stepwell.
- ₹200 — Bright yellow, depicts the Sanchi Stupa.
- ₹500 — Stone grey, depicts the Red Fort.
Each note in the New Series is smaller than the previous series and uses a distinct color-coded
theme per denomination to assist quick visual identification.
""",
        "security_features.md": """# INR Security Features
RBI notes include a see-through register, a latent image showing the denomination when the
note is held flat, a windowed demetalized security thread inscribed with "Bharat" and "RBI" that
appears alternately in Hindi and English, an intaglio-printed Mahatma Gandhi portrait, micro
lettering, and color-shifting ink on higher denominations. Bleed lines and enlarged numerals on
the note edges assist visually impaired users in identifying denominations.
""",
        "history.md": """# History of the Indian Rupee
India adopted a decimal system for the rupee in 1957. The RBI has issued several note series
since independence, most notably the Mahatma Gandhi Series (1996) and the Mahatma Gandhi New
Series (2016), the latter launched alongside the 2016 demonetization of ₹500 and ₹1000 notes,
which was intended to curb counterfeit currency and undisclosed cash holdings.
""",
        "travel_tips.md": """# Using INR as a Traveler
Currency exchange for foreign visitors is best done at authorized dealers or banks; carrying
large amounts of cash is discouraged given digital payment (UPI) prevalence in urban India.
Old, torn, or heavily soiled notes can usually still be exchanged at bank branches. Travelers
should be cautious of counterfeit ₹500 notes in informal markets and verify security threads
when accepting large-denomination cash.
""",
    },
    "japan": {
        "overview.md": """# Japanese Yen (JPY) — Overview
The Japanese Yen (code JPY, symbol ¥) is issued by the Bank of Japan (BOJ). Unlike most
currencies, the yen has no subdivision in practical use today (the sen and rin subunits were
abolished from circulation in 1953). Circulating banknotes are ¥1000, ¥2000 (rare), ¥5000, and
¥10000, alongside coins from ¥1 to ¥500.
""",
        "denominations.md": """# JPY Banknote Denominations
- ¥1000 — Blue, features Kitasato Shibasaburo (2024 series) or Hideyo Noguchi (older series).
- ¥5000 — Purple, features Ichiyo Higuchi (2024 series) or Umeko Tsuda.
- ¥10000 — Brown, features Eiichi Shibusawa (2024 series) or Yukichi Fukuzawa (older series).
A new note series was issued starting July 2024 with upgraded 3D hologram portraits, while
older-series notes remain valid legal tender indefinitely.
""",
        "security_features.md": """# JPY Security Features
Bank of Japan notes use latent-image intaglio printing that reveals the denomination when
tilted, holographic strips or 3D holograms (2024 series) that rotate the portrait image, pearl
ink patterns visible at an angle, microprinting, and watermarks incorporating both a portrait and
a traditional pattern. The 2024 series added the world's first banknote 3D hologram at the time
of issue as an advanced anti-counterfeiting measure.
""",
        "history.md": """# History of the Japanese Yen
The yen was established in 1871, replacing the earlier Tokugawa-era coinage system, and pegged
originally to gold and silver standards. After WWII, the yen was fixed to the US dollar at
360:1 under the Bretton Woods system, before floating from 1973 onward. Japan periodically
redesigns its currency roughly every 20 years to counter forgery, with the most recent full
redesign completed in 2024.
""",
        "travel_tips.md": """# Using JPY as a Traveler
Japan remains a comparatively cash-reliant society outside major cities, so carrying yen is
useful even where cards are accepted. 7-Eleven, post office, and airport ATMs reliably accept
foreign cards. Coins are used heavily for small transactions, and it is common practice to place
change in a small tray at checkout rather than handing it directly to the cashier.
""",
    },
    "china": {
        "overview.md": """# Chinese Yuan / Renminbi (CNY) — Overview
The official currency of China is the Renminbi (RMB), with the yuan (¥ or 元) as its basic unit,
issued by the People's Bank of China (PBOC). It subdivides into 10 jiao and 100 fen. The current
banknote series in general circulation is the 2019 edition of the Fifth Series of RMB, covering
¥1, ¥5, ¥10, ¥20, ¥50, and ¥100 notes.
""",
        "denominations.md": """# CNY Banknote Denominations
- ¥1 — Olive green, portrait of Mao Zedong, orchid design.
- ¥5 — Brown-purple, Mount Tai theme.
- ¥10 — Blue-green, Three Gorges theme.
- ¥20 — Brown-yellow, Guilin landscape theme.
- ¥50 — Dark green, Potala Palace theme.
- ¥100 — Red, Great Hall of the People theme.
All Fifth Series notes since 1999 feature Mao Zedong's portrait on the obverse, a departure from
earlier series that depicted various ethnic groups and workers.
""",
        "security_features.md": """# CNY Security Features
Fifth Series (2019) RMB notes include an optically variable color-shifting numeral, a
watermark portrait of Mao Zedong, a magnetic security thread, intaglio printing giving raised
texture to the portrait and denomination numerals, and UV-reactive fibres. The ¥50 and ¥100
notes additionally use a gold-colored optical security strip that changes appearance when
tilted.
""",
        "history.md": """# History of the Renminbi
The Renminbi was introduced in 1948 by the People's Bank of China ahead of the founding of the
People's Republic in 1949. It has gone through five main banknote series, with major
redenominations and redesigns reflecting economic reform periods, most notably the 1955
currency reform and later note upgrades in 1999 and 2019 to standardize security features.
""",
        "travel_tips.md": """# Using CNY as a Traveler
Mobile payment apps (Alipay, WeChat Pay) dominate daily transactions in China, and foreign
visitors increasingly can link international cards to these apps. Cash remains accepted
everywhere and useful as backup, particularly in smaller towns. Always verify large notes (¥50,
¥100) at banks when exchanging, as counterfeit high-denomination notes have historically
circulated in informal markets.
""",
    },
    "thailand": {
        "overview.md": """# Thai Baht (THB) — Overview
The Thai Baht (code THB, symbol ฿) is issued by the Bank of Thailand (BOT), subdivided into 100
satang. Circulating banknotes are ฿20, ฿50, ฿100, ฿500, and ฿1000, each featuring a portrait of
the reigning monarch, King Maha Vajiralongkorn (Rama X), on the current series introduced from
2018 onward.
""",
        "denominations.md": """# THB Banknote Denominations
- ฿20 — Green, depicts King Rama I era history.
- ฿50 — Blue, depicts King Rama IV era history.
- ฿100 — Red, depicts King Rama V era history.
- ฿500 — Purple, depicts King Rama VII/VIII era history.
- ฿1000 — Grey-brown, depicts King Rama IX (Bhumibol Adulyadej) achievements.
Each note's reverse commemorates a different reign in the Chakri dynasty, forming a historical
narrative sequence across the denomination set.
""",
        "security_features.md": """# THB Security Features
Bank of Thailand notes contain a watermark portrait of the king visible when held to light, a
color-shifting security thread, intaglio-printed raised ink for the portrait and denomination
numerals, latent images visible at an angle, and, on the ฿100 and above, an optically variable
ink patch. Tactile marks (raised dots/dashes) assist blind and visually impaired users in
distinguishing denominations.
""",
        "history.md": """# History of the Thai Baht
The baht has been Thailand's currency since the 19th century, historically valued by weight of
silver before decimalization in 1897 (100 satang = 1 baht). Thailand floated the baht in 1997
during the Asian Financial Crisis after previously pegging it to the US dollar, an event widely
regarded as the trigger of the broader regional crisis that year.
""",
        "travel_tips.md": """# Using THB as a Traveler
Baht notes are widely accepted, and small vendors, markets, and taxis often prefer or require
cash. It is customary and legally expected to treat banknotes with respect, since they bear the
king's portrait; folding, defacing, or stepping on currency is considered highly disrespectful
in Thai culture. ATMs are widely available but typically charge a foreign transaction fee.
""",
    },
}

count = 0
for country, files in DOCS.items():
    d = os.path.join(BASE, country)
    os.makedirs(d, exist_ok=True)
    for fname, content in files.items():
        with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        count += 1

print(f"Generated {count} knowledge base documents across {len(DOCS)} countries.")
