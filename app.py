import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import re
from pathlib import Path
from urllib.parse import quote_plus
import requests
import feedparser
from yahooquery import Ticker as YahooQueryTicker

from bobcat_lab_core import (
    CRYPTO_OPTIONS,
    FALLBACK_COMPANY_PROFILES,
    REQUEST_HEADERS,
    SECTOR_EXPOSURES,
    STOCK_MARKET_BENCHMARK,
    TRADING_DAYS,
    build_prediction_signal,
    build_relevance_terms,
    build_sp500_screener,
    build_takeaway,
    classify_catalysts,
    classify_risk_profile,
    clean_search_tokens,
    compute_analysis,
    compute_ivol_percentile,
    deduplicate_articles,
    download_close_batches,
    extract_close,
    fetch_yahoo_headlines,
    format_market_caption,
    format_profile_line,
    get_company_profile,
    get_market_snapshot,
    get_stock_catalog,
    initialize_app_state,
    load_asset_vs_benchmark,
    render_date_messages,
)

TRADING_DAYS_PER_QUARTER = max(1, int(round(TRADING_DAYS / 4)))
PREDICTION_CHECK_HORIZONS = (
    (1, "1 day after"),
    (15, "15 days after"),
    (30, "30 days after"),
    (TRADING_DAYS_PER_QUARTER, f"3 months after (≈{TRADING_DAYS_PER_QUARTER} trading sessions)"),
)

st.set_page_config(page_title="Bob Cat Risk Lab", layout="wide")

gif_path = Path(__file__).with_name("finance.gif")

st.markdown("### Group: Bob Cat")
st.title("Bob Cat Risk Lab")
st.write(
    "An interactive market-versus-firm risk platform for large-cap stocks and cryptocurrencies. "
    "Use it to analyze one asset, compare two assets side by side, or screen the full S&P 500."
)

if gif_path.exists():
    st.image(str(gif_path), caption="Finance-themed intro", use_container_width=True)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CURATED_RSS_FEEDS = [
    {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "category": "policy"},
    {"name": "SEC Press Releases", "url": "https://www.sec.gov/news/pressreleases.rss", "category": "policy"},
    {"name": "IMF News", "url": "https://www.imf.org/en/News/RSS", "category": "macro"},
    {"name": "MarketWatch Top Stories", "url": "https://feeds.marketwatch.com/marketwatch/topstories/", "category": "market"},
]
GLOBAL_MAJOR_RSS_FEEDS = [
    {"name": "The New York Times (Business)", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "category": "macro"},
    {"name": "The New York Times (World)", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "category": "macro"},
    {"name": "CNN Top Stories", "url": "http://rss.cnn.com/rss/cnn_topstories.rss", "category": "macro"},
    {"name": "BBC News (World)", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "category": "macro"},
    {"name": "The Guardian (World)", "url": "https://www.theguardian.com/world/rss", "category": "macro"},
    {"name": "Al Jazeera English", "url": "https://www.aljazeera.com/xml/rss/all.xml", "category": "macro"},
    {"name": "The Washington Post (World)", "url": "https://feeds.washingtonpost.com/rss/world", "category": "macro"},
    {"name": "The Economist (World This Week)", "url": "https://www.economist.com/the-world-this-week/rss.xml", "category": "macro"},
    {"name": "France 24 (English)", "url": "https://www.france24.com/en/rss", "category": "macro"},
    {"name": "DW (English)", "url": "https://rss.dw.com/xml/rss-en-all", "category": "macro"},
    {"name": "BBC News (Africa)", "url": "http://feeds.bbci.co.uk/news/world/africa/rss.xml", "category": "macro"},
    {"name": "Africanews (pan-Africa)", "url": "https://www.africanews.com/feed/rss", "category": "macro"},
]
MAJOR_OUTLET_GOOGLE_SITES = [
    ("Reuters", "reuters.com"),
    ("The New York Times", "nytimes.com"),
    ("CNN", "cnn.com"),
    ("BBC News", "bbc.co.uk"),
    ("The Guardian", "theguardian.com"),
    ("The Washington Post", "washingtonpost.com"),
    ("Financial Times", "ft.com"),
    ("Bloomberg", "bloomberg.com"),
]
TOP_NEWSPAPER_RSS_BY_COUNTRY = {
    "united states": {"name": "USA Today (major US wire)", "url": "https://rssfeeds.usatoday.com/usatoday-NewsTopStories", "category": "local_major"},
    "united kingdom": {"name": "The Guardian UK", "url": "https://www.theguardian.com/uk/rss", "category": "local_major"},
    "india": [
        {"name": "Times of India", "url": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "category": "local_major"},
        {"name": "The Hindu (National)", "url": "https://www.thehindu.com/news/national/?service=rss", "category": "local_major"},
        {"name": "Indian Express", "url": "https://indianexpress.com/feed/", "category": "local_major"},
        {"name": "Economic Times", "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms", "category": "local_major"},
        {"name": "Hindustan Times (India News)", "url": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml", "category": "local_major"},
    ],
    "canada": {"name": "CBC News (Canada)", "url": "https://rss.cbc.ca/lineup/topstories.xml", "category": "local_major"},
    "japan": {"name": "The Japan Times", "url": "https://www.japantimes.co.jp/feed/topstories/", "category": "local_major"},
    "china": {"name": "South China Morning Post", "url": "https://www.scmp.com/rss/91/feed", "category": "local_major"},
    "germany": {"name": "Der Spiegel (English)", "url": "https://www.spiegel.de/international/index.rss", "category": "local_major"},
    "france": {"name": "Le Monde", "url": "https://www.lemonde.fr/rss/une.xml", "category": "local_major"},
    "australia": [
        {"name": "ABC News (Australia)", "url": "https://www.abc.net.au/news/feed/46182/rss.xml", "category": "local_major"},
        {"name": "The Sydney Morning Herald", "url": "https://www.smh.com.au/rss/feed.xml", "category": "local_major"},
        {"name": "News.com.au", "url": "https://www.news.com.au/content-feeds/latest-news-national/", "category": "local_major"},
    ],
    "argentina": [
        {"name": "Clarín", "url": "https://www.clarin.com/rss/lo-ultimo/", "category": "local_major"},
        {"name": "La Nación (Argentina)", "url": "https://www.lanacion.com.ar/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
        {"name": "Infobae", "url": "https://www.infobae.com/arc/outboundfeeds/rss/", "category": "local_major"},
    ],
    "bolivia": {"name": "Página Siete", "url": "https://www.paginasiete.bo/rss/", "category": "local_major"},
    "brazil": [
        {"name": "G1 Globo (Brazil)", "url": "https://g1.globo.com/rss/g1/", "category": "local_major"},
        {"name": "BBC Brasil (Portuguese)", "url": "https://feeds.bbci.co.uk/portuguese/rss.xml", "category": "local_major"},
    ],
    "chile": [
        {"name": "La Tercera", "url": "https://www.latercera.com/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
        {"name": "La Cuarta", "url": "https://www.lacuarta.com/feed/", "category": "local_major"},
    ],
    "colombia": {"name": "Semana", "url": "https://www.semana.com/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
    "ecuador": {"name": "El Universo (Ecuador)", "url": "https://www.eluniverso.com/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
    "paraguay": {"name": "ABC Color", "url": "https://www.abc.com.py/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
    "peru": [
        {"name": "El Comercio (Peru)", "url": "https://elcomercio.pe/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
        {"name": "Gestión (Peru)", "url": "https://gestion.pe/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
        {"name": "Andina (Peru state wire)", "url": "https://andina.pe/agencia/rss.aspx", "category": "local_major"},
    ],
    "uruguay": {"name": "El País (Uruguay)", "url": "https://www.elpais.com.uy/rss/", "category": "local_major"},
    "venezuela": {"name": "teleSUR", "url": "https://www.telesurtv.net/rss", "category": "local_major"},
    "south africa": [
        {"name": "Mail & Guardian (South Africa)", "url": "https://mg.co.za/feed/", "category": "local_major"},
        {"name": "Daily Maverick", "url": "https://www.dailymaverick.co.za/dmrss/", "category": "local_major"},
        {"name": "IOL", "url": "https://www.iol.co.za/rss", "category": "local_major"},
    ],
    "nigeria": [
        {"name": "Punch (Nigeria)", "url": "https://punchng.com/feed/", "category": "local_major"},
        {"name": "Vanguard (Nigeria)", "url": "https://www.vanguardngr.com/feed/", "category": "local_major"},
        {"name": "ThisDay (Nigeria)", "url": "https://www.thisdaylive.com/feed/", "category": "local_major"},
        {"name": "BusinessDay (Nigeria)", "url": "https://businessday.ng/feed/", "category": "local_major"},
    ],
    "kenya": [
        {"name": "Nation.Africa (Kenya)", "url": "https://nation.africa/kenya/rss.xml", "category": "local_major"},
        {"name": "The Standard (Kenya)", "url": "https://www.standardmedia.co.ke/rss/", "category": "local_major"},
        {"name": "Tuko (Kenya)", "url": "https://www.tuko.co.ke/rss/all.rss", "category": "local_major"},
        {"name": "The East African", "url": "https://www.theeastafrican.co.ke/rss.xml", "category": "local_major"},
    ],
    "ghana": {"name": "Joy News (Ghana)", "url": "https://www.myjoyonline.com/feed/", "category": "local_major"},
    "tanzania": [
        {"name": "The Citizen (Tanzania)", "url": "https://www.thecitizen.co.tz/rss.xml", "category": "local_major"},
        {"name": "Daily News (Tanzania)", "url": "https://www.dailynews.co.tz/feed/", "category": "local_major"},
    ],
    "uganda": {"name": "The Independent (Uganda)", "url": "https://www.independent.co.ug/feed/", "category": "local_major"},
    "rwanda": {"name": "The New Times (Rwanda)", "url": "https://www.newtimes.co.rw/rss", "category": "local_major"},
    "egypt": {"name": "Egypt Independent", "url": "https://www.egyptindependent.com/feed/", "category": "local_major"},
    "morocco": {"name": "Morocco World News", "url": "https://www.moroccoworldnews.com/feed/", "category": "local_major"},
    "tunisia": {"name": "Business News (Tunisia)", "url": "https://www.businessnews.com.tn/feed", "category": "local_major"},
    "algeria": {"name": "Algérie360", "url": "https://www.algerie360.com/feed/", "category": "local_major"},
    "senegal": [
        {"name": "APS (Senegal wire)", "url": "https://www.aps.sn/feed", "category": "local_major"},
        {"name": "Senego", "url": "https://senego.com/feed/", "category": "local_major"},
    ],
    "zambia": {"name": "Lusaka Times", "url": "https://www.lusakatimes.com/feed/", "category": "local_major"},
    "zimbabwe": {"name": "NewsDay (Zimbabwe)", "url": "https://www.newsday.co.zw/feed/", "category": "local_major"},
    "ethiopia": [
        {"name": "Fana Broadcasting (Ethiopia)", "url": "https://www.fanabc.com/feed/", "category": "local_major"},
        {"name": "Africanews (pan-Africa)", "url": "https://www.africanews.com/feed/rss", "category": "local_major"},
    ],
    "mozambique": {"name": "Club of Mozambique", "url": "https://clubofmozambique.com/feed/", "category": "local_major"},
    "malawi": {"name": "Nyasa Times (Malawi)", "url": "https://www.nyasatimes.com/feed/", "category": "local_major"},
    "afghanistan": {"name": "Pajhwok Afghan News", "url": "https://pajhwok.com/feed/", "category": "local_major"},
    "bangladesh": [
        {"name": "Prothom Alo (English)", "url": "https://en.prothomalo.com/stories.rss", "category": "local_major"},
        {"name": "The Daily Star (Bangladesh)", "url": "https://www.thedailystar.net/top-news/rss.xml", "category": "local_major"},
    ],
    "bhutan": {"name": "The Bhutanese", "url": "https://thebhutanese.bt/feed/", "category": "local_major"},
    "maldives": {"name": "Edition.mv (Maldives)", "url": "https://edition.mv/feed/", "category": "local_major"},
    "nepal": {"name": "The Kathmandu Post", "url": "https://kathmandupost.com/rss/", "category": "local_major"},
    "pakistan": [
        {"name": "The News International", "url": "https://www.thenews.com.pk/rss/1", "category": "local_major"},
        {"name": "Geo News", "url": "https://www.geo.tv/rss/1", "category": "local_major"},
    ],
    "sri lanka": {"name": "Ada Derana", "url": "https://www.adaderana.lk/rss.php", "category": "local_major"},
    "saudi arabia": {"name": "Arab News", "url": "https://www.arabnews.com/rss.xml", "category": "local_major"},
    "united arab emirates": {"name": "The National (UAE)", "url": "https://www.thenationalnews.com/arc/outboundfeeds/rss/?outputType=xml", "category": "local_major"},
    "qatar": {"name": "The Peninsula (Qatar)", "url": "https://thepeninsulaqatar.com/feed/", "category": "local_major"},
    "kuwait": {"name": "Arab Times (Kuwait)", "url": "https://www.arabtimesonline.com/feed/", "category": "local_major"},
    "bahrain": {"name": "News of Bahrain", "url": "https://www.newsofbahrain.com/feed/", "category": "local_major"},
    "oman": {"name": "Times of Oman", "url": "https://timesofoman.com/feed/", "category": "local_major"},
    "jordan": {"name": "The Jordan Times", "url": "https://jordantimes.com/rss", "category": "local_major"},
    "lebanon": {"name": "Middle East Eye (Lebanon & regional)", "url": "https://www.middleeasteye.net/rss", "category": "local_major"},
    "iraq": {"name": "Iraq News", "url": "https://www.iraqinews.com/feed/", "category": "local_major"},
    "iran": [
        {"name": "Tehran Times", "url": "https://www.tehrantimes.com/rss", "category": "local_major"},
        {"name": "IRNA (English)", "url": "https://en.irna.ir/rss.aspx", "category": "local_major"},
    ],
    "turkey": {"name": "Daily Sabah (English)", "url": "https://www.dailysabah.com/rss", "category": "local_major"},
    "yemen": {"name": "Saba News (Yemen)", "url": "https://www.sabanews.net/en/feed/", "category": "local_major"},
    "syria": {"name": "SANA (Syria)", "url": "https://sana.sy/en/?feed=rss2", "category": "local_major"},
    "palestine": {"name": "Middle East Eye (Palestine & regional)", "url": "https://www.middleeasteye.net/rss", "category": "local_major"},
    "jamaica": {"name": "Jamaica Gleaner", "url": "https://jamaica-gleaner.com/feed/rss.xml", "category": "local_major"},
    "trinidad and tobago": {"name": "CNC3 (Trinidad & Tobago)", "url": "https://cnc3.co.tt/feed/", "category": "local_major"},
    "barbados": {"name": "Barbados Today", "url": "https://www.barbadostoday.bb/feed/", "category": "local_major"},
    "bahamas": {"name": "Eyewitness News (Bahamas)", "url": "https://ewnews.com/feed/", "category": "local_major"},
    "cuba": {"name": "Granma (Cuba)", "url": "http://www.granma.cu/feed/", "category": "local_major"},
    "dominican republic": {"name": "Dominican Today", "url": "https://dominicantoday.com/feed/", "category": "local_major"},
    "haiti": {"name": "The Haitian Times", "url": "https://haitiantimes.com/feed/", "category": "local_major"},
    "puerto rico": {"name": "News is My Business (Puerto Rico)", "url": "https://www.newsismybusiness.com/feed/", "category": "local_major"},
    "cayman islands": {"name": "Cayman Compass", "url": "https://www.caymancompass.com/feed/", "category": "local_major"},
    "antigua and barbuda": {"name": "Antigua Observer", "url": "https://antiguaobserver.com/feed/", "category": "local_major"},
    "british virgin islands": {"name": "BVI News", "url": "https://bvinews.com/feed/", "category": "local_major"},
    "saint lucia": {"name": "St. Lucia Times", "url": "https://stluciatimes.com/feed/", "category": "local_major"},
    "grenada": {"name": "NOW Grenada", "url": "https://nowgrenada.com/feed/", "category": "local_major"},
    "philippines": [
        {"name": "Philippine Daily Inquirer", "url": "https://www.inquirer.net/feed/", "category": "local_major"},
        {"name": "Rappler", "url": "https://www.rappler.com/feed/", "category": "local_major"},
    ],
    "vietnam": [
        {"name": "Vietnam News (English)", "url": "https://vietnamnews.vn/rss/home.rss", "category": "local_major"},
        {"name": "VNExpress", "url": "https://vnexpress.net/rss/kinh-doanh.rss", "category": "local_major"},
    ],
    "malaysia": {"name": "Malay Mail", "url": "https://www.malaymail.com/rss/feed", "category": "local_major"},
    "indonesia": {"name": "CNN Indonesia", "url": "https://www.cnnindonesia.com/nasional/rss", "category": "local_major"},
    "new zealand": {"name": "NZ Herald", "url": "https://www.nzherald.co.nz/arc/outboundfeeds/rss/section/nz/?outputType=xml", "category": "local_major"},
    "fiji": {"name": "FBC News (Fiji)", "url": "https://www.fbcnews.com.fj/feed/", "category": "local_major"},
    "papua new guinea": {"name": "Post Courier (PNG)", "url": "https://postcourier.com.pg/feed/", "category": "local_major"},
    "marshall islands": {"name": "Marshall Islands Journal", "url": "https://marshallislandsjournal.com/feed/", "category": "local_major"},
    "palau": {"name": "Island Times (Palau)", "url": "https://islandtimes.org/feed/", "category": "local_major"},
    "northern mariana islands": {"name": "Marianas Variety", "url": "https://www.mvariety.com/feed/", "category": "local_major"},
    "guam": {"name": "KUAM (Guam)", "url": "https://www.kuam.com/feed/", "category": "local_major"},
    "cook islands": {"name": "Cook Islands News", "url": "https://www.cookislandsnews.com/feed/", "category": "local_major"},
    "solomon islands": {"name": "Solomon Star", "url": "https://www.solomonstarnews.com/feed/", "category": "local_major"},
    "south korea": {"name": "The Korea Herald", "url": "https://www.koreaherald.com/common/rss_xml.php?ct=0", "category": "local_major"},
    "mexico": {"name": "BBC Mundo (Latin America wire)", "url": "https://feeds.bbci.co.uk/mundo/rss.xml", "category": "local_major"},
    "italy": {"name": "ANSA English", "url": "https://www.ansa.it/english/english_rss.xml", "category": "local_major"},
    "spain": {"name": "El País (English)", "url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada", "category": "local_major"},
    "netherlands": {"name": "DutchNews.nl", "url": "https://www.dutchnews.nl/feed/", "category": "local_major"},
    "ireland": {"name": "RTÉ News", "url": "https://www.rte.ie/news/rss/news-headlines.xml", "category": "local_major"},
    "israel": {"name": "The Times of Israel", "url": "https://www.timesofisrael.com/feed/", "category": "local_major"},
    "singapore": {"name": "The Straits Times", "url": "https://www.straitstimes.com/news/world/rss.xml", "category": "local_major"},
    "russia": [
        {"name": "TASS English", "url": "https://tass.com/rss/v2.xml", "category": "local_major"},
        {"name": "The Moscow Times", "url": "https://www.themoscowtimes.com/rss/news", "category": "local_major"},
        {"name": "Interfax (Russia)", "url": "https://www.interfax.ru/rss.asp", "category": "local_major"},
    ],
    "kazakhstan": {"name": "Tengrinews (Kazakhstan)", "url": "https://tengrinews.kz/news.rss", "category": "local_major"},
    "uzbekistan": [
        {"name": "Gazeta.uz", "url": "https://www.gazeta.uz/uz/rss", "category": "local_major"},
        {"name": "Daryo (Uzbekistan, English)", "url": "https://daryo.uz/en/feed/", "category": "local_major"},
    ],
    "kyrgyzstan": [
        {"name": "AKIpress (English)", "url": "https://akipress.com/rss/en/", "category": "local_major"},
        {"name": "24.kg (Kyrgyzstan)", "url": "https://24.kg/rss/", "category": "local_major"},
    ],
    "tajikistan": {"name": "Asia-Plus (Tajikistan)", "url": "https://asiaplustj.info/rss.xml", "category": "local_major"},
    "turkmenistan": {"name": "Orient (Turkmenistan)", "url": "https://orient.tm/feed/", "category": "local_major"},
    "mongolia": [
        {"name": "news.mn (Mongolia)", "url": "https://news.mn/feed/", "category": "local_major"},
        {"name": "ikon.mn (Mongolia)", "url": "https://ikon.mn/rss", "category": "local_major"},
    ],
    "azerbaijan": {"name": "Trend News Agency (Azerbaijan)", "url": "https://en.trend.az/feeds/index.rss", "category": "local_major"},
    "armenia": {"name": "News.am (English)", "url": "https://news.am/eng/rss/", "category": "local_major"},
    "georgia": {"name": "Civil.ge", "url": "https://civil.ge/feed/", "category": "local_major"},
    "belarus": {"name": "Dev.by (Belarus)", "url": "https://dev.by/rss", "category": "local_major"},
    "switzerland": {"name": "France 24 (European wire)", "url": "https://www.france24.com/en/rss", "category": "local_major"},
    "hong kong": {"name": "South China Morning Post", "url": "https://www.scmp.com/rss/91/feed", "category": "local_major"},
}
COUNTRY_EXTRA_GOOGLE_SITES = {
    "taiwan": [
        ("Taipei Times", "taipeitimes.com"),
        ("Taiwan News", "taiwannews.com.tw"),
        ("Focus Taiwan", "focustaiwan.tw"),
    ],
    "hong kong": [("South China Morning Post", "scmp.com")],
    "india": [
        ("Times of India", "indiatimes.com"),
        ("The Hindu", "thehindu.com"),
        ("Indian Express", "indianexpress.com"),
        ("Economic Times", "economictimes.indiatimes.com"),
    ],
    "argentina": [
        ("Clarín", "clarin.com"),
        ("La Nación", "lanacion.com.ar"),
        ("Infobae", "infobae.com"),
    ],
    "brazil": [("G1", "g1.globo.com"), ("UOL", "uol.com.br"), ("Folha", "folha.uol.com.br")],
    "chile": [("La Tercera", "latercera.com"), ("La Cuarta", "lacuarta.com"), ("Emol", "emol.com")],
    "colombia": [("Semana", "semana.com"), ("El Tiempo", "eltiempo.com"), ("El Espectador", "elespectador.com")],
    "peru": [("El Comercio", "elcomercio.pe"), ("Gestión", "gestion.pe"), ("Andina", "andina.pe")],
    "ecuador": [("El Universo", "eluniverso.com")],
    "bolivia": [("Página Siete", "paginasiete.bo")],
    "paraguay": [("ABC Color", "abc.com.py")],
    "uruguay": [("El País Uruguay", "elpais.com.uy"), ("El Observador", "elobservador.com.uy")],
    "venezuela": [("teleSUR", "telesurtv.net")],
    "mexico": [("Reforma", "reforma.com"), ("El Universal", "eluniversal.com.mx")],
    "south africa": [
        ("Mail & Guardian", "mg.co.za"),
        ("News24", "news24.com"),
        ("IOL", "iol.co.za"),
    ],
    "nigeria": [
        ("Punch", "punchng.com"),
        ("Vanguard", "vanguardngr.com"),
        ("ThisDay", "thisdaylive.com"),
        ("BusinessDay Nigeria", "businessday.ng"),
    ],
    "kenya": [
        ("Nation.Africa", "nation.africa"),
        ("The Standard", "standardmedia.co.ke"),
        ("Tuko", "tuko.co.ke"),
    ],
    "ghana": [("Joy News", "myjoyonline.com"), ("Graphic", "graphic.com.gh")],
    "tanzania": [("The Citizen", "thecitizen.co.tz"), ("Daily News TZ", "dailynews.co.tz")],
    "uganda": [("The Independent Uganda", "independent.co.ug"), ("Monitor", "monitor.co.ug")],
    "rwanda": [("The New Times Rwanda", "newtimes.co.rw")],
    "egypt": [("Egypt Independent", "egyptindependent.com"), ("Ahram Online", "ahram.org.eg")],
    "morocco": [("Morocco World News", "moroccoworldnews.com"), ("Hespress", "hespress.com")],
    "tunisia": [("Business News Tunisia", "businessnews.com.tn")],
    "algeria": [("Algérie360", "algerie360.com")],
    "senegal": [("APS", "aps.sn"), ("Senego", "senego.com")],
    "zambia": [("Lusaka Times", "lusakatimes.com")],
    "zimbabwe": [("NewsDay Zimbabwe", "newsday.co.zw")],
    "ethiopia": [("The Reporter Ethiopia", "thereporterethiopia.com"), ("Fana Broadcasting", "fanabc.com")],
    "mozambique": [("Club of Mozambique", "clubofmozambique.com")],
    "malawi": [("Nyasa Times", "nyasatimes.com")],
    "russia": [
        ("TASS", "tass.com"),
        ("The Moscow Times", "themoscowtimes.com"),
        ("Interfax", "interfax.ru"),
        ("Kommersant", "kommersant.com"),
    ],
    "kazakhstan": [("Tengrinews", "tengrinews.kz"), ("Kazinform", "inform.kz")],
    "uzbekistan": [("Gazeta.uz", "gazeta.uz"), ("Daryo", "daryo.uz")],
    "kyrgyzstan": [("AKIpress", "akipress.com"), ("24.kg", "24.kg")],
    "tajikistan": [("Asia-Plus", "asiaplustj.info")],
    "turkmenistan": [("Orient", "orient.tm"), ("TDH", "tdh.gov.tm")],
    "mongolia": [("news.mn", "news.mn"), ("ikon.mn", "ikon.mn")],
    "azerbaijan": [("Trend", "trend.az"), ("Azernews", "azernews.az")],
    "armenia": [("News.am", "news.am"), ("Armenpress", "armenpress.am")],
    "georgia": [("Civil.ge", "civil.ge"), ("Agenda.ge", "agenda.ge")],
    "belarus": [("Dev.by", "dev.by"), ("BELTA", "belta.by")],
    "afghanistan": [("Pajhwok", "pajhwok.com"), ("TOLOnews", "tolonews.com")],
    "bangladesh": [
        ("Prothom Alo", "prothomalo.com"),
        ("The Daily Star", "thedailystar.net"),
        ("BDNews24", "bdnews24.com"),
    ],
    "bhutan": [("Kuensel", "kuenselonline.com"), ("The Bhutanese", "thebhutanese.bt")],
    "maldives": [("Edition.mv", "edition.mv"), ("Sun.mv", "sun.mv")],
    "nepal": [("The Kathmandu Post", "kathmandupost.com"), ("The Himalayan Times", "thehimalayantimes.com")],
    "pakistan": [
        ("Dawn", "dawn.com"),
        ("The News", "thenews.com.pk"),
        ("Express Tribune", "tribune.com.pk"),
    ],
    "sri lanka": [("Ada Derana", "adaderana.lk"), ("Daily Mirror", "dailymirror.lk")],
    "saudi arabia": [("Arab News", "arabnews.com"), ("Saudi Gazette", "saudigazette.com.sa")],
    "united arab emirates": [
        ("The National", "thenationalnews.com"),
        ("Khaleej Times", "khaleejtimes.com"),
        ("Gulf News", "gulfnews.com"),
    ],
    "qatar": [("The Peninsula", "thepeninsulaqatar.com"), ("Gulf Times", "gulf-times.com")],
    "kuwait": [("Arab Times", "arabtimesonline.com"), ("KUNA", "kuna.net.kw")],
    "bahrain": [("Gulf Daily News", "gdnonline.com"), ("News of Bahrain", "newsofbahrain.com")],
    "oman": [("Times of Oman", "timesofoman.com"), ("Oman Observer", "omanobserver.om")],
    "jordan": [("The Jordan Times", "jordantimes.com"), ("Petra", "petra.gov.jo")],
    "lebanon": [("Middle East Eye", "middleeasteye.net"), ("Naharnet", "naharnet.com")],
    "iraq": [("Iraq News", "iraqinews.com"), ("Rudaw", "rudaw.net")],
    "iran": [("Tehran Times", "tehrantimes.com"), ("IRNA", "irna.ir")],
    "turkey": [("Daily Sabah", "dailysabah.com"), ("Hürriyet Daily News", "hurriyetdailynews.com")],
    "yemen": [("Saba News", "sabanews.net")],
    "syria": [("SANA", "sana.sy")],
    "palestine": [("WAFA", "wafa.ps"), ("Middle East Eye", "middleeasteye.net")],
    "jamaica": [("Jamaica Gleaner", "jamaica-gleaner.com"), ("Jamaica Observer", "jamaicaobserver.com")],
    "trinidad and tobago": [("Trinidad Guardian", "guardian.co.tt"), ("CNC3", "cnc3.co.tt")],
    "barbados": [("Barbados Today", "barbadostoday.bb"), ("Nation News", "nationnews.com")],
    "bahamas": [("Eyewitness News", "ewnews.com"), ("Tribune242", "tribune242.com")],
    "cuba": [("Granma", "granma.cu"), ("Prensa Latina", "plenglish.com")],
    "dominican republic": [("Dominican Today", "dominicantoday.com"), ("Diario Libre", "diariolibre.com")],
    "haiti": [("The Haitian Times", "haitiantimes.com"), ("Haiti Libre", "haitilibre.com")],
    "puerto rico": [("El Nuevo Día", "elnuevodia.com"), ("News is my Business", "newsismybusiness.com")],
    "cayman islands": [("Cayman Compass", "caymancompass.com")],
    "antigua and barbuda": [("Antigua Observer", "antiguaobserver.com")],
    "british virgin islands": [("BVI News", "bvinews.com")],
    "saint lucia": [("St. Lucia Times", "stluciatimes.com")],
    "grenada": [("NOW Grenada", "nowgrenada.com")],
    "philippines": [("Inquirer", "inquirer.net"), ("Rappler", "rappler.com"), ("ABS-CBN", "abs-cbn.com")],
    "vietnam": [("Vietnam News", "vietnamnews.vn"), ("VNExpress", "vnexpress.net")],
    "malaysia": [("Malay Mail", "malaymail.com"), ("The Star", "thestar.com.my")],
    "indonesia": [("CNN Indonesia", "cnnindonesia.com"), ("Jakarta Post", "thejakartapost.com")],
    "brunei": [("Borneo Bulletin", "borneobulletin.com.bn"), ("Scoop", "scoop.co.bn")],
    "australia": [("ABC News", "abc.net.au"), ("Sydney Morning Herald", "smh.com.au"), ("News.com.au", "news.com.au")],
    "new zealand": [("NZ Herald", "nzherald.co.nz"), ("Stuff", "stuff.co.nz")],
    "fiji": [("FBC News", "fbcnews.com.fj"), ("Fiji Times", "fijitimes.com.fj")],
    "papua new guinea": [("Post Courier", "postcourier.com.pg"), ("The National PNG", "thenational.com.pg")],
    "marshall islands": [("Marshall Islands Journal", "marshallislandsjournal.com")],
    "palau": [("Island Times", "islandtimes.org")],
    "northern mariana islands": [("Marianas Variety", "mvariety.com"), ("Saipan Tribune", "saipantribune.com")],
    "guam": [("KUAM", "kuam.com"), ("Pacific Daily News", "guampdn.com")],
    "cook islands": [("Cook Islands News", "cookislandsnews.com")],
    "solomon islands": [("Solomon Star", "solomonstarnews.com")],
}
CENTRAL_BANK_TOPIC_TERMS = [
    "interest rate",
    "monetary policy",
    "policy rate",
    "benchmark rate",
    "central bank",
    "price stability",
    "inflation target",
    "foreign exchange",
    "exchange rate",
    "forward guidance",
    "policy decision",
    "basis points",
    "repo rate",
    "asset purchase",
    "quantitative easing",
    "balance sheet",
    "financial stability",
    "macroprudential",
    "discount rate",
    "open market",
    "fomc",
    "federal reserve",
    "ecb",
    "european central bank",
    "minutes of the",
    "press release",
    "communique",
]
CENTRAL_BANK_GOOGLE_QUERY_CORE = (
    "(monetary policy OR inflation OR interest rate OR policy rate OR benchmark OR central bank OR governor)"
)
ECB_CB_RSS = {"name": "ECB (press releases)", "url": "https://www.ecb.europa.eu/rss/press.html", "category": "policy_cb"}
FED_GOOGLE = ("Federal Reserve", "federalreserve.gov")
ECB_GOOGLE = ("ECB", "ecb.europa.eu")
CENTRAL_BANK_RSS_BY_COUNTRY = {
    "canada": [
        {"name": "Bank of Canada (news releases)", "url": "https://www.bankofcanada.ca/feed/", "category": "policy_cb"},
    ],
    "germany": [ECB_CB_RSS],
    "france": [ECB_CB_RSS],
    "italy": [ECB_CB_RSS],
    "spain": [ECB_CB_RSS],
    "netherlands": [ECB_CB_RSS],
    "ireland": [ECB_CB_RSS],
}
CENTRAL_BANK_GOOGLE_SITES_BY_COUNTRY = {
    "united states": [FED_GOOGLE],
    "united kingdom": [("Bank of England", "bankofengland.co.uk")],
    "india": [("Reserve Bank of India", "rbi.org.in")],
    "canada": [("Bank of Canada", "bankofcanada.ca")],
    "japan": [("Bank of Japan", "boj.or.jp")],
    "china": [("People's Bank of China", "pbc.gov.cn")],
    "germany": [ECB_GOOGLE],
    "france": [ECB_GOOGLE],
    "australia": [("Reserve Bank of Australia", "rba.gov.au")],
    "argentina": [("Banco Central de la República Argentina", "bcra.gob.ar")],
    "bolivia": [("Banco Central de Bolivia", "bcb.gob.bo")],
    "brazil": [("Banco Central do Brasil", "bcb.gov.br")],
    "chile": [("Banco Central de Chile", "bcentral.cl")],
    "colombia": [("Banco de la República", "banrep.gov.co")],
    "ecuador": [("Banco Central del Ecuador", "bce.fin.ec")],
    "paraguay": [("Banco Central del Paraguay", "bcp.gov.py")],
    "peru": [("Banco Central de Reserva del Perú", "bcrp.gob.pe")],
    "uruguay": [("Banco Central del Uruguay", "bcub.gub.uy")],
    "venezuela": [("Banco Central de Venezuela", "bcv.org.ve")],
    "south africa": [("South African Reserve Bank", "resbank.co.za")],
    "nigeria": [("Central Bank of Nigeria", "cbn.gov.ng")],
    "kenya": [("Central Bank of Kenya", "centralbank.go.ke")],
    "ghana": [("Bank of Ghana", "bog.gov.gh")],
    "tanzania": [("Bank of Tanzania", "bot.go.tz")],
    "uganda": [("Bank of Uganda", "bou.or.ug")],
    "rwanda": [("National Bank of Rwanda", "bnr.rw")],
    "egypt": [("Central Bank of Egypt", "cbe.org.eg")],
    "morocco": [("Bank Al-Maghrib", "bankal-maghrib.ma")],
    "tunisia": [("Central Bank of Tunisia", "bct.gov.tn")],
    "algeria": [("Bank of Algeria", "bank-of-algeria.dz")],
    "senegal": [("BCEAO (West African central bank)", "bceao.int")],
    "zambia": [("Bank of Zambia", "boz.zm")],
    "zimbabwe": [("Reserve Bank of Zimbabwe", "rbz.co.zw")],
    "ethiopia": [("National Bank of Ethiopia", "nbe.gov.et")],
    "mozambique": [("Banco de Moçambique", "bancomoc.mz")],
    "malawi": [("Reserve Bank of Malawi", "rbm.mw")],
    "afghanistan": [("Da Afghanistan Bank", "dab.gov.af")],
    "bangladesh": [("Bangladesh Bank", "bb.org.bd")],
    "bhutan": [("Royal Monetary Authority of Bhutan", "rma.org.bt")],
    "maldives": [("Maldives Monetary Authority", "mma.gov.mv")],
    "nepal": [("Nepal Rastra Bank", "nrb.org.np")],
    "pakistan": [("State Bank of Pakistan", "sbp.org.pk")],
    "sri lanka": [("Central Bank of Sri Lanka", "cbsl.gov.lk")],
    "saudi arabia": [("Saudi Central Bank (SAMA)", "sama.gov.sa")],
    "united arab emirates": [("Central Bank of the UAE", "centralbank.ae")],
    "qatar": [("Qatar Central Bank", "qcb.gov.qa")],
    "kuwait": [("Central Bank of Kuwait", "cbk.gov.kw")],
    "bahrain": [("Central Bank of Bahrain", "cbb.gov.bh")],
    "oman": [("Central Bank of Oman", "cbo.gov.om")],
    "jordan": [("Central Bank of Jordan", "centralbank.gov.jo")],
    "lebanon": [("Banque du Liban", "bdl.gov.lb")],
    "iraq": [("Central Bank of Iraq", "cbi.iq")],
    "iran": [("Central Bank of Iran", "cbi.ir")],
    "turkey": [("Central Bank of the Republic of Türkiye", "tcmb.gov.tr")],
    "yemen": [("Central Bank of Yemen", "cbye.org.ye")],
    "syria": [("Central Bank of Syria", "cbs.gov.sy")],
    "palestine": [("Palestine Monetary Authority", "pmna.gov.ps")],
    "jamaica": [("Bank of Jamaica", "boj.org.jm")],
    "trinidad and tobago": [("Central Bank of Trinidad and Tobago", "centralbank.org.tt")],
    "barbados": [("Central Bank of Barbados", "centralbank.org.bb")],
    "bahamas": [("Central Bank of The Bahamas", "centralbankbahamas.com")],
    "cuba": [("Banco Central de Cuba", "bc.gob.cu")],
    "dominican republic": [("Banco Central de la República Dominicana", "bancentral.gov.do")],
    "haiti": [("Banque de la République d'Haïti", "brh.ht")],
    "puerto rico": [FED_GOOGLE],
    "cayman islands": [("Cayman Islands Monetary Authority", "cima.ky")],
    "antigua and barbuda": [("Eastern Caribbean Central Bank", "eccb-centralbank.org")],
    "british virgin islands": [("Eastern Caribbean Central Bank", "eccb-centralbank.org")],
    "saint lucia": [("Eastern Caribbean Central Bank", "eccb-centralbank.org")],
    "grenada": [("Eastern Caribbean Central Bank", "eccb-centralbank.org")],
    "philippines": [("Bangko Sentral ng Pilipinas", "bsp.gov.ph")],
    "vietnam": [("State Bank of Vietnam", "sbv.gov.vn")],
    "malaysia": [("Bank Negara Malaysia", "bnm.gov.my")],
    "indonesia": [("Bank Indonesia", "bi.go.id")],
    "new zealand": [("Reserve Bank of New Zealand", "rbnz.govt.nz")],
    "fiji": [("Reserve Bank of Fiji", "reservebank.gov.fj")],
    "papua new guinea": [("Bank of Papua New Guinea", "bankpng.gov.pg")],
    "marshall islands": [FED_GOOGLE],
    "palau": [FED_GOOGLE],
    "northern mariana islands": [FED_GOOGLE],
    "guam": [FED_GOOGLE],
    "cook islands": [("Reserve Bank of New Zealand (NZD anchor)", "rbnz.govt.nz")],
    "solomon islands": [("Central Bank of Solomon Islands", "cbsi.com.sb")],
    "south korea": [("Bank of Korea", "bok.or.kr")],
    "mexico": [("Banco de México", "banxico.org.mx")],
    "italy": [ECB_GOOGLE],
    "spain": [ECB_GOOGLE],
    "netherlands": [ECB_GOOGLE],
    "ireland": [ECB_GOOGLE],
    "israel": [("Bank of Israel", "boi.org.il")],
    "singapore": [("Monetary Authority of Singapore", "mas.gov.sg")],
    "russia": [("Bank of Russia", "cbr.ru")],
    "kazakhstan": [("National Bank of Kazakhstan", "nationalbank.kz")],
    "uzbekistan": [("Central Bank of Uzbekistan", "cbu.uz")],
    "kyrgyzstan": [("National Bank of the Kyrgyz Republic", "nbkr.kg")],
    "tajikistan": [("National Bank of Tajikistan", "nbt.tj")],
    "turkmenistan": [("Central Bank of Turkmenistan", "cbt.tm")],
    "mongolia": [("Bank of Mongolia", "mongolbank.mn")],
    "azerbaijan": [("Central Bank of Azerbaijan", "cbar.az")],
    "armenia": [("Central Bank of Armenia", "cba.am")],
    "georgia": [("National Bank of Georgia", "nbg.gov.ge")],
    "belarus": [("National Bank of Belarus", "nbrb.by")],
    "switzerland": [("Swiss National Bank", "snb.ch")],
    "hong kong": [("Hong Kong Monetary Authority", "hkma.gov.hk")],
    "taiwan": [("Central Bank of the Republic of China (Taiwan)", "cbc.gov.tw")],
    "brunei": [("Autoriti Monetari Brunei Darussalam", "ambd.gov.bn")],
}
PREDICTION_SUPPORTED_MODES = {"Single Company", "Single Crypto"}


def normalize_country_key(country):
    if not country or not isinstance(country, str):
        return None
    c = country.strip().lower()
    aliases = {
        "usa": "united states",
        "u.s.": "united states",
        "u.s.a.": "united states",
        "us": "united states",
        "u.k.": "united kingdom",
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",
        "scotland": "united kingdom",
        "south korea": "south korea",
        "korea, republic of": "south korea",
        "korea (south)": "south korea",
        "republic of korea": "south korea",
        "russian federation": "russia",
        "taiwan, province of china": "taiwan",
        "hong kong sar, china": "hong kong",
        "hong kong sar": "hong kong",
        "republic of argentina": "argentina",
        "argentine republic": "argentina",
        "republic of chile": "chile",
        "republic of colombia": "colombia",
        "republic of peru": "peru",
        "perú": "peru",
        "republic of ecuador": "ecuador",
        "plurinational state of bolivia": "bolivia",
        "republic of bolivia": "bolivia",
        "republic of paraguay": "paraguay",
        "oriental republic of uruguay": "uruguay",
        "bolivarian republic of venezuela": "venezuela",
        "republic of venezuela": "venezuela",
        "federative republic of brazil": "brazil",
        "republic of south africa": "south africa",
        "federal republic of nigeria": "nigeria",
        "republic of kenya": "kenya",
        "republic of ghana": "ghana",
        "united republic of tanzania": "tanzania",
        "republic of uganda": "uganda",
        "republic of rwanda": "rwanda",
        "arab republic of egypt": "egypt",
        "kingdom of morocco": "morocco",
        "republic of tunisia": "tunisia",
        "people's democratic republic of algeria": "algeria",
        "republic of senegal": "senegal",
        "republic of zambia": "zambia",
        "republic of zimbabwe": "zimbabwe",
        "federal democratic republic of ethiopia": "ethiopia",
        "republic of mozambique": "mozambique",
        "republic of malawi": "malawi",
        "republic of kazakhstan": "kazakhstan",
        "republic of uzbekistan": "uzbekistan",
        "kyrgyz republic": "kyrgyzstan",
        "republic of kyrgyzstan": "kyrgyzstan",
        "republic of tajikistan": "tajikistan",
        "republic of azerbaijan": "azerbaijan",
        "republic of armenia": "armenia",
        "republic of belarus": "belarus",
        "republic of georgia": "georgia",
        "islamic republic of afghanistan": "afghanistan",
        "people's republic of bangladesh": "bangladesh",
        "kingdom of bhutan": "bhutan",
        "republic of maldives": "maldives",
        "federal democratic republic of nepal": "nepal",
        "islamic republic of pakistan": "pakistan",
        "democratic socialist republic of sri lanka": "sri lanka",
        "kingdom of saudi arabia": "saudi arabia",
        "uae": "united arab emirates",
        "the united arab emirates": "united arab emirates",
        "state of qatar": "qatar",
        "state of kuwait": "kuwait",
        "kingdom of bahrain": "bahrain",
        "sultanate of oman": "oman",
        "hashemite kingdom of jordan": "jordan",
        "lebanese republic": "lebanon",
        "republic of iraq": "iraq",
        "islamic republic of iran": "iran",
        "republic of turkey": "turkey",
        "türkiye": "turkey",
        "republic of yemen": "yemen",
        "syrian arab republic": "syria",
        "state of palestine": "palestine",
        "palestinian territory": "palestine",
        "occupied palestinian territory": "palestine",
        "republic of the philippines": "philippines",
        "socialist republic of viet nam": "vietnam",
        "viet nam": "vietnam",
        "republic of indonesia": "indonesia",
        "brunei darussalam": "brunei",
        "independent state of papua new guinea": "papua new guinea",
        "republic of the marshall islands": "marshall islands",
        "republic of palau": "palau",
        "commonwealth of the northern mariana islands": "northern mariana islands",
        "republic of trinidad and tobago": "trinidad and tobago",
        "commonwealth of the bahamas": "bahamas",
        "the bahamas": "bahamas",
        "republic of cuba": "cuba",
        "republic of haiti": "haiti",
        "st. lucia": "saint lucia",
        "virgin islands (british)": "british virgin islands",
        "commonwealth of puerto rico": "puerto rico",
    }
    return aliases.get(c, c)


def build_news_queries(asset_name, asset_ticker, sector=None, industry=None):
    exposures = SECTOR_EXPOSURES.get(sector or "", [])[:3]
    company_query = f'"{asset_name}" {asset_ticker} earnings OR guidance OR merger OR lawsuit OR CEO'
    sector_query = f'"{sector or "market"}" "{industry or sector or asset_name}" demand OR competition OR regulation'
    macro_query = f'"{asset_name}" {asset_ticker} {" OR ".join(exposures or ["rates", "inflation", "regulation"])}'
    return [
        {"query": company_query, "category": "company"},
        {"query": sector_query, "category": "sector"},
        {"query": macro_query, "category": "policy_macro"},
    ]


def format_entry_published(entry):
    for key in ("published", "updated", "pubDate"):
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _parallel_google_queries(query_configs, max_items=4):
    if not query_configs:
        return []

    def run_one(qc):
        return fetch_google_news(qc["query"], qc["category"], max_items=max_items)

    with ThreadPoolExecutor(max_workers=min(3, len(query_configs))) as executor:
        futures = [executor.submit(run_one, qc) for qc in query_configs]
        merged = []
        for fut in as_completed(futures):
            try:
                merged.extend(fut.result() or [])
            except Exception:
                pass
    return merged


def fetch_google_news(query, query_category, max_items=4):
    try:
        response = requests.get(
            f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception:
        return []

    articles = []
    for entry in feed.entries[:max_items]:
        articles.append(
            {
                "title": entry.get("title"),
                "summary": re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip(),
                "source": getattr(getattr(entry, "source", {}), "title", None) or "Google News",
                "published": format_entry_published(entry),
                "url": entry.get("link"),
                "source_type": "google",
                "query_category": query_category,
            }
        )
    return articles


def fetch_filtered_rss_articles(feed_configs, relevance_terms, max_total, max_per_feed=None):
    articles = []
    for feed_config in feed_configs:
        per_feed = 0
        try:
            response = requests.get(feed_config["url"], headers=REQUEST_HEADERS, timeout=12)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception:
            continue

        for entry in feed.entries:
            summary = re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip()
            text = f"{entry.get('title', '')} {summary}".lower()
            if relevance_terms and not any(term in text for term in relevance_terms):
                continue
            articles.append(
                {
                    "title": entry.get("title"),
                    "summary": summary,
                    "source": feed_config["name"],
                    "published": format_entry_published(entry),
                    "url": entry.get("link"),
                    "source_type": "rss",
                    "query_category": feed_config["category"],
                }
            )
            per_feed += 1
            if max_per_feed is not None and per_feed >= max_per_feed:
                break
            if len(articles) >= max_total:
                return articles
    return articles


def fetch_curated_rss_items(relevance_terms, max_items=6):
    return fetch_filtered_rss_articles(CURATED_RSS_FEEDS, relevance_terms, max_total=max_items, max_per_feed=None)


def fetch_global_major_rss(relevance_terms, max_total=12, max_per_feed=2):
    return fetch_filtered_rss_articles(GLOBAL_MAJOR_RSS_FEEDS, relevance_terms, max_total=max_total, max_per_feed=max_per_feed)


def fetch_country_newspaper_rss(relevance_terms, country, max_total=4, max_per_feed=4):
    key = normalize_country_key(country)
    if not key:
        return []
    cfg = TOP_NEWSPAPER_RSS_BY_COUNTRY.get(key)
    if not cfg:
        return []
    feeds = cfg if isinstance(cfg, list) else [cfg]
    if len(feeds) > 1:
        return fetch_filtered_rss_articles(feeds, relevance_terms, max_total=max(8, max_total), max_per_feed=2)
    return fetch_filtered_rss_articles(feeds, relevance_terms, max_total=max_total, max_per_feed=max_per_feed)


def fetch_rss_cb_policy_feeds(feed_configs, relevance_terms, max_total, max_per_feed=None):
    articles = []
    topic_terms = CENTRAL_BANK_TOPIC_TERMS
    seen_feed_urls = set()
    unique_feeds = []
    for fc in feed_configs:
        u = fc["url"]
        if u in seen_feed_urls:
            continue
        seen_feed_urls.add(u)
        unique_feeds.append(fc)

    for feed_config in unique_feeds:
        per_feed = 0
        try:
            response = requests.get(feed_config["url"], headers=REQUEST_HEADERS, timeout=12)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception:
            continue

        for entry in feed.entries:
            summary = re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip()
            text = f"{entry.get('title', '')} {summary}".lower()
            company_hit = bool(relevance_terms and any(term in text for term in relevance_terms))
            topic_hit = any(term in text for term in topic_terms)
            if not (company_hit or topic_hit):
                continue
            articles.append(
                {
                    "title": entry.get("title"),
                    "summary": summary,
                    "source": feed_config["name"],
                    "published": format_entry_published(entry),
                    "url": entry.get("link"),
                    "source_type": "rss",
                    "query_category": feed_config.get("category", "policy_cb"),
                }
            )
            per_feed += 1
            if max_per_feed is not None and per_feed >= max_per_feed:
                break
            if len(articles) >= max_total:
                return articles
    return articles


def fetch_central_bank_policy_rss(relevance_terms, country, max_total=6, max_per_feed=2):
    key = normalize_country_key(country)
    if not key:
        return []
    cfg = CENTRAL_BANK_RSS_BY_COUNTRY.get(key)
    if not cfg:
        return []
    feeds = cfg if isinstance(cfg, list) else [cfg]
    return fetch_rss_cb_policy_feeds(feeds, relevance_terms, max_total=max_total, max_per_feed=max_per_feed)


def fetch_central_bank_policy_google(country, max_per_site=2, max_total=8):
    key = normalize_country_key(country)
    if not key:
        return []
    sites = CENTRAL_BANK_GOOGLE_SITES_BY_COUNTRY.get(key)
    if not sites:
        return []
    base_q = CENTRAL_BANK_GOOGLE_QUERY_CORE

    def pull_cb(label_domain):
        label, domain = label_domain
        query = f"{base_q} site:{domain}"
        batch = []
        try:
            response = requests.get(
                f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
                headers=REQUEST_HEADERS,
                timeout=10,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception:
            return batch
        for entry in feed.entries[:max_per_site]:
            batch.append(
                {
                    "title": entry.get("title"),
                    "summary": re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip(),
                    "source": label,
                    "published": format_entry_published(entry),
                    "url": entry.get("link"),
                    "source_type": "google_central_bank",
                    "query_category": "policy_cb",
                }
            )
        return batch

    articles = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sites)))) as executor:
        futures = [executor.submit(pull_cb, pair) for pair in sites]
        for fut in as_completed(futures):
            try:
                articles.extend(fut.result() or [])
            except Exception:
                pass
            if len(articles) >= max_total:
                break
    seen_urls = set()
    deduped = []
    for item in articles:
        link = (item.get("url") or "").strip()
        if link:
            lk = link.lower()
            if lk in seen_urls:
                continue
            seen_urls.add(lk)
        deduped.append(item)
        if len(deduped) >= max_total:
            break
    return deduped


def fetch_google_news_site_scoped(
    asset_name, asset_ticker, extra_site_tuples=None, max_per_site=2, max_major_sites=None
):
    base = f'"{asset_name}" OR {asset_ticker}'
    sites = list(MAJOR_OUTLET_GOOGLE_SITES)
    if max_major_sites is not None:
        sites = sites[:max_major_sites]
    if extra_site_tuples:
        sites.extend(extra_site_tuples)

    def pull_one(label_domain):
        label, domain = label_domain
        query = f"{base} site:{domain}"
        items = []
        try:
            response = requests.get(
                f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
                headers=REQUEST_HEADERS,
                timeout=10,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception:
            return items

        for entry in feed.entries[:max_per_site]:
            link = entry.get("link")
            items.append(
                {
                    "title": entry.get("title"),
                    "summary": re.sub(r"<[^>]+>", " ", entry.get("summary", "")).strip(),
                    "source": label,
                    "published": format_entry_published(entry),
                    "url": link,
                    "source_type": "google_outlet",
                    "query_category": "macro",
                }
            )
        return items

    articles = []
    workers = min(12, max(1, len(sites)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(pull_one, pair) for pair in sites]
        for fut in as_completed(futures):
            try:
                articles.extend(fut.result() or [])
            except Exception:
                pass

    seen_urls = set()
    deduped = []
    for item in articles:
        link = item.get("url")
        if link:
            lk = link.lower()
            if lk in seen_urls:
                continue
            seen_urls.add(lk)
        deduped.append(item)
    return deduped


def fetch_gdelt_items(asset_name, asset_ticker, sector=None, max_items=6):
    query = f'"{asset_name}" OR {asset_ticker}'
    if sector:
        query = f"{query} OR {sector}"
    try:
        response = requests.get(
            GDELT_API_URL,
            params={
                "query": query,
                "mode": "ArtList",
                "maxrecords": max_items,
                "format": "json",
                "sort": "DateDesc",
            },
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    articles = []
    for item in payload.get("articles", [])[:max_items]:
        articles.append(
            {
                "title": item.get("title"),
                "summary": item.get("seendate") or item.get("domain"),
                "source": item.get("source") or item.get("domain") or "GDELT",
                "published": item.get("seendate"),
                "url": item.get("url"),
                "source_type": "gdelt",
                "query_category": "macro",
            }
        )
    return articles


@st.cache_data(ttl=1200, show_spinner=False)
def get_news_context(
    asset_ticker, asset_name, sector=None, industry=None, country=None, max_items=24, fast_news=True
):
    relevance_terms = build_relevance_terms(asset_name, asset_ticker, sector, industry)
    all_articles = []
    source_counts = {}
    coverage_notes = []

    queries = build_news_queries(asset_name, asset_ticker, sector, industry)
    query_slice = queries[:2] if fast_news else queries
    g_max = 2 if fast_news else 4
    max_items_eff = 18 if fast_news else max_items

    country_key = normalize_country_key(country)
    extra_sites = list(COUNTRY_EXTRA_GOOGLE_SITES.get(country_key, [])) if country_key else []
    if fast_news:
        extra_sites = extra_sites[:3]

    feed_slice = GLOBAL_MAJOR_RSS_FEEDS[: (6 if fast_news else len(GLOBAL_MAJOR_RSS_FEEDS))]
    glob_max_total = 6 if fast_news else 12
    glob_per_feed = 1 if fast_news else 2
    yahoo_max = 4 if fast_news else 5
    curated_cap = 4 if fast_news else 6

    def task_google():
        return _parallel_google_queries(query_slice, max_items=g_max)

    def task_outlets():
        return fetch_google_news_site_scoped(
            asset_name,
            asset_ticker,
            extra_site_tuples=extra_sites or None,
            max_per_site=1 if fast_news else 2,
            max_major_sites=6 if fast_news else None,
        )

    def task_global():
        return fetch_filtered_rss_articles(
            feed_slice, relevance_terms, max_total=glob_max_total, max_per_feed=glob_per_feed
        )

    def task_curated():
        return fetch_curated_rss_items(relevance_terms, max_items=curated_cap)

    def task_country_cb():
        if fast_news:
            return []
        a = fetch_country_newspaper_rss(relevance_terms, country)
        b = fetch_central_bank_policy_rss(relevance_terms, country)
        c = fetch_central_bank_policy_google(country)
        return (a or []) + (b or []) + (c or [])

    def task_gdelt():
        if fast_news:
            return []
        return fetch_gdelt_items(asset_name, asset_ticker, sector)

    def task_yahoo():
        return fetch_yahoo_headlines(asset_ticker, max_items=yahoo_max)

    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            "google": executor.submit(task_google),
            "outlets": executor.submit(task_outlets),
            "global": executor.submit(task_global),
            "curated": executor.submit(task_curated),
            "country_cb": executor.submit(task_country_cb),
            "gdelt": executor.submit(task_gdelt),
            "yahoo": executor.submit(task_yahoo),
        }
        results = {}
        for name, fut in futures.items():
            try:
                results[name] = fut.result(timeout=45)
            except Exception:
                results[name] = []

    google_articles = results["google"]
    if google_articles:
        all_articles.extend(google_articles)
        source_counts["Google News"] = source_counts.get("Google News", 0) + len(google_articles)

    outlet_articles = results["outlets"]
    if outlet_articles:
        all_articles.extend(outlet_articles)
        for item in outlet_articles:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1

    global_major = results["global"]
    if global_major:
        all_articles.extend(global_major)
        for item in global_major:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1

    curated = results["curated"]
    if curated:
        all_articles.extend(curated)
        for item in curated:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
    else:
        coverage_notes.append("Curated policy and market RSS feeds did not return relevant matches.")

    for item in results["country_cb"]:
        all_articles.append(item)
        src = item.get("source") or item.get("source_type") or "News"
        source_counts[src] = source_counts.get(src, 0) + 1

    gdelt_articles = results["gdelt"]
    if gdelt_articles:
        all_articles.extend(gdelt_articles)
        source_counts["GDELT"] = source_counts.get("GDELT", 0) + len(gdelt_articles)
    elif not fast_news:
        coverage_notes.append(
            "GDELT world-event coverage was unavailable for this run; major wires (Reuters, FT, Bloomberg, etc.) "
            "are still queried via Google News site search, alongside NYTimes, CNN, BBC, Guardian, and other RSS feeds."
        )

    yahoo_articles = results["yahoo"]
    if yahoo_articles:
        all_articles.extend(yahoo_articles)
        source_counts["Yahoo Finance"] = source_counts.get("Yahoo Finance", 0) + len(yahoo_articles)
    else:
        coverage_notes.append("Yahoo Finance fallback headlines were unavailable.")

    deduped_articles = deduplicate_articles(all_articles, max_items=max_items_eff)
    if not deduped_articles:
        coverage_notes.append("No recent public headlines were available, so the prediction leans more heavily on the quantitative signals.")

    return {
        "articles": deduped_articles,
        "coverage_notes": coverage_notes,
        "source_counts": source_counts,
        "relevance_terms": relevance_terms,
        "country": country,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def download_asset_close_slice(asset_ticker, start, end):
    raw = yf.download(asset_ticker, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return None
    close = extract_close(raw, asset_ticker)
    if close is None or close.empty:
        return None
    return close.sort_index()


def _directional_outcome_label(action, forward_return, band=0.01):
    if forward_return is None or not np.isfinite(forward_return):
        return "n/a", "No data."
    if action == "Buy":
        if forward_return > band:
            return "consistent", "Price rose over this horizon—**directionally consistent** with Buy."
        if forward_return < -band:
            return "conflict", "Price fell—**conflicts** with Buy as a directional call."
        return "muted", "Move was small; **inconclusive** for judging Buy on price alone."
    if action in ("Sell", "Avoid"):
        if forward_return < -band:
            return "consistent", "Price fell—**directionally consistent** with a cautious stance."
        if forward_return > band:
            return "conflict", "Price rose—**conflicts** with Sell/Avoid as a directional bet."
        return "muted", "Move was small; **inconclusive** for Sell/Avoid."
    if action == "Hold":
        if abs(forward_return) < 2 * band:
            return "consistent", "Price stayed relatively **muted**, which fits a non-directional Hold."
        return "mixed", "Price moved materially; Hold is **not** a directional forecast, so this is only partly informative."
    return "muted", "—"


def _answer_right_or_wrong(action, forward_rows, forward_error):
    """
    Single headline verdict vs realized price path after the sample end.
    Buy/Sell/Avoid: directional. Hold: "right" if price stayed relatively muted.
    """
    if not forward_rows:
        detail = (
            forward_error
            or "No post-sample prices were available—pick an **End date** before today to compare the call to what happened next."
        )
        return {
            "label": "Cannot check yet",
            "short": "Cannot check yet",
            "detail": detail,
            "kind": "not_verifiable",
        }

    by_days = {r["trading_days"]: r for r in forward_rows}
    primary = (
        by_days.get(30)
        or by_days.get(TRADING_DAYS_PER_QUARTER)
        or by_days.get(15)
        or by_days.get(1)
        or forward_rows[0]
    )
    tags = [r["tag"] for r in forward_rows]

    if "conflict" in tags and "consistent" in tags:
        return {
            "label": "Inconclusive",
            "short": "Inconclusive",
            "detail": "Different forward horizons **disagree** (one matches the call, one does not), so there is no single yes/no.",
            "kind": "inconclusive",
        }

    tag = primary["tag"]
    h = primary["trading_days"]
    r = primary["return"]
    hl = primary.get("horizon_label") or f"{h} trading sessions after sample end"

    if tag == "consistent":
        return {
            "label": "Yes — directionally",
            "short": "Yes",
            "detail": (
                f"At **{hl}** (trading sessions from sample end), the price path **matched** a directional read of **{action}** "
                f"(cumulative return **{r:+.2%}**). This only scores direction, not magnitude or risk."
            ),
            "kind": "yes",
        }
    if tag == "conflict":
        return {
            "label": "No — directionally",
            "short": "No",
            "detail": (
                f"At **{hl}** (trading sessions from sample end), the price path **went the other way** vs a simple directional read of **{action}** "
                f"(cumulative return **{r:+.2%}**)."
            ),
            "kind": "no",
        }

    return {
        "label": "Inconclusive",
        "short": "Inconclusive",
        "detail": (
            f"At **{hl}** the move was **{r:+.2%}**—too small or ambiguous for a clear right/wrong on price alone."
        ),
        "kind": "inconclusive",
    }


def build_prediction_verdict(prediction, metrics, asset_ticker):
    action = prediction["action"]
    ann = float(metrics["annualized_return"])
    total_score = float(prediction["total_score"])
    actual_end = metrics["actual_end"]
    today = pd.Timestamp.today().normalize().date()

    internal = []
    if action == "Buy":
        if ann < -0.08:
            internal.append(
                f"**Tension:** Annualized return over your sample is quite negative ({ann:.1%}) while the model prefers **Buy**. "
                "Stabilizing volatility or positive headlines may be outweighing the historical trend."
            )
        elif ann > 0.03:
            internal.append(
                f"The sample’s annualized return trend is positive ({ann:.1%}), which **aligns** with a Buy lean."
            )
        else:
            internal.append(
                f"Return trend over the sample is moderate ({ann:.1%}); it **neither strongly confirms nor contradicts** Buy."
            )
    elif action == "Hold":
        internal.append(
            f"**Hold** is mostly neutral. Sample return was {ann:.1%} annualized—use catalyst and confidence notes before acting."
        )
    elif action in ("Sell", "Avoid"):
        if ann > 0.08:
            internal.append(
                f"**Tension:** Sample return trend is strongly positive ({ann:.1%}) while the model leans **{action}**. "
                "Risk metrics or headline risks may dominate the raw trend."
            )
        elif ann < -0.03:
            internal.append(
                f"Negative sample return trend ({ann:.1%}) **fits** a defensive **{action}** reading."
            )
        else:
            internal.append(
                f"Sample return ({ann:.1%}) is **mixed** relative to **{action}**—weigh the evidence lists."
            )

    if abs(total_score) < 2 and prediction.get("confidence") == "High":
        internal.append(
            "**Score check:** Combined score is modest but confidence was labeled high—treat as **low robustness** if evidence is thin."
        )

    forward_rows = []
    forward_error = None
    if pd.Timestamp(actual_end).date() >= today:
        forward_error = (
            "The sample ends on the latest calendar day in this run, so **completed** post-sample closes are not available yet."
        )
    else:
        fwd_end = max(
            pd.Timestamp(today) + pd.Timedelta(days=5),
            pd.Timestamp(actual_end) + pd.Timedelta(days=110),
        )
        close = download_asset_close_slice(asset_ticker, actual_end, fwd_end)
        if close is None or len(close) < 2:
            forward_error = "Could not load enough **post-sample** prices to score what happened next."
        else:
            base_idx = None
            for i, dt in enumerate(close.index):
                if hasattr(dt, "date"):
                    d = dt.date()
                else:
                    d = pd.Timestamp(dt).date()
                if d >= actual_end:
                    base_idx = i
                    break
            if base_idx is None:
                forward_error = "Could not align the sample end date to extended price data."
            else:
                for h, horizon_label in PREDICTION_CHECK_HORIZONS:
                    if base_idx + h >= len(close):
                        continue
                    r = float(close.iloc[base_idx + h] / close.iloc[base_idx] - 1.0)
                    tag, explain = _directional_outcome_label(action, r)
                    forward_rows.append(
                        {
                            "trading_days": h,
                            "horizon_label": horizon_label,
                            "return": r,
                            "tag": tag,
                            "detail": explain,
                        }
                    )
                if not forward_rows:
                    forward_error = (
                        "Too few trading days after the sample end in the downloaded window—try an earlier **End date** "
                        "so more forward history can be measured."
                    )

    summary = None
    if forward_rows:
        tags = [r["tag"] for r in forward_rows]
        if all(t == "consistent" for t in tags):
            summary = "Post-sample price moves were **mostly consistent** with the directional lean across horizons shown below."
        elif any(t == "conflict" for t in tags):
            summary = "At least one horizon **conflicted** with a simple directional read—the call is not uniformly “right” or “wrong” on price alone."
        elif all(t == "muted" for t in tags):
            summary = "Post-sample moves were **small**; the label is hard to score as right or wrong from price alone."
        else:
            summary = "Post-sample evidence is **mixed** by horizon; see the table."

    answer = _answer_right_or_wrong(action, forward_rows, forward_error)

    return {
        "internal": internal,
        "forward_rows": forward_rows,
        "forward_error": forward_error,
        "summary": summary,
        "answer": answer,
    }


def render_prediction_sense_check(prediction, metrics, asset_ticker, *, show_heading=True):
    verdict = build_prediction_verdict(prediction, metrics, asset_ticker)
    if show_heading:
        st.markdown("#### Is the prediction right? (Sense-check)")
    st.caption(
        "Educational check only—not a guarantee. **In-sample** tension compares the call to your window’s return trend. "
        "**Horizons** (1 / 15 / 30 / 3-month) count **trading sessions** after the sample end, not calendar days. "
        "**Out-of-sample** rows use closes *after* the sample end date when that history exists."
    )

    ans = verdict.get("answer") or {}
    headline = f"**Was it the right answer?** **{ans.get('label', '—')}**"
    if ans.get("kind") == "yes":
        st.success(f"{headline}\n\n{ans.get('detail', '')}")
    elif ans.get("kind") == "no":
        st.warning(f"{headline}\n\n{ans.get('detail', '')}")
    elif ans.get("kind") == "inconclusive":
        st.info(f"{headline}\n\n{ans.get('detail', '')}")
    else:
        st.info(f"{headline}\n\n{ans.get('detail', '')}")

    for line in verdict["internal"]:
        st.markdown(line)
    if verdict["forward_error"]:
        st.info(verdict["forward_error"])
    if verdict["summary"]:
        if "conflicted" in verdict["summary"]:
            st.warning(verdict["summary"])
        elif "mostly consistent" in verdict["summary"]:
            st.success(verdict["summary"])
        else:
            st.info(verdict["summary"])
    if verdict["forward_rows"]:
        st.markdown("**After the sample end (by trading days)**")
        tbl = pd.DataFrame(
            [
                {
                    "Horizon": r.get("horizon_label", f"{r['trading_days']} sessions"),
                    "Sessions ahead": r["trading_days"],
                    "Cumulative return": f"{r['return']:.2%}",
                    "Versus call": r["detail"],
                }
                for r in verdict["forward_rows"]
            ]
        )
        st.dataframe(tbl, use_container_width=True)
    st.caption("Markets can reverse for reasons the model does not observe; use this alongside your own judgment.")


def render_prediction_button():
    if st.button("View Prediction Details", type="primary"):
        st.session_state["analysis_view"] = "prediction"
        st.rerun()


def render_grouped_articles(grouped_articles):
    if not grouped_articles:
        st.info("No source-backed headlines were available for this run.")
        return

    for catalyst, items in grouped_articles.items():
        with st.expander(f"{catalyst.upper()} Headlines ({len(items)})"):
            for item in items[:5]:
                title = item.get("title") or "Untitled headline"
                source = item.get("source") or item.get("source_type", "Source")
                published = item.get("published") or "Recent"
                if item.get("url"):
                    st.markdown(f"- [{title}]({item['url']})")
                else:
                    st.markdown(f"- {title}")
                st.caption(f"{source} | {published}")


def render_prediction_detail_page(asset_name, asset_ticker, profile, benchmark_label, metrics, risk_profile, market_snapshot, prediction):
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("Back to Dashboard"):
            st.session_state["analysis_view"] = "dashboard"
            st.rerun()
    with top_right:
        st.subheader("Prediction Details")
        st.caption("Best-effort public-data model. This is an educational support tool, not guaranteed investment advice.")

    st.markdown(f"**Selected Asset:** {asset_name} (`{asset_ticker}`)")
    profile_line = format_profile_line(profile)
    if profile_line:
        st.markdown(profile_line)

    hero1, hero2, hero3, hero4 = st.columns(4)
    hero1.metric("Preferred Action", prediction["action"])
    hero2.metric("Confidence", prediction["confidence"])
    hero3.metric("Quant Score", f"{prediction['quant_score']:.1f}")
    hero4.metric("News Score", f"{prediction['news_score']:.1f}")

    st.info(prediction["summary"])
    if market_snapshot:
        st.caption(format_market_caption(market_snapshot))

    render_prediction_sense_check(prediction, metrics, asset_ticker, show_heading=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Why Buy")
        for reason in prediction["why_buy"]:
            st.markdown(f"- {reason}")
    with right:
        st.markdown("#### Why Sell or Wait")
        for reason in prediction["why_sell"]:
            st.markdown(f"- {reason}")

    risk_left, risk_right = st.columns(2)
    with risk_left:
        st.markdown("#### Risk of Losing")
        st.write(prediction["risk_text"])
        st.caption(f"Risk profile: {risk_profile} | Annualized total volatility: {metrics['total_vol_annual']:.2%}")
    with risk_right:
        st.markdown("#### Chance of Winning")
        st.write(prediction["upside_text"])
        st.caption(f"Annualized return trend: {metrics['annualized_return']:.2%} | Benchmark: {benchmark_label}")

    st.markdown("#### What News Was Used")
    render_grouped_articles(prediction["grouped_articles"])

    policy_macro_items = {
        label: items
        for label, items in prediction["grouped_articles"].items()
        if label in {"policy", "macro", "sector"}
    }
    st.markdown("#### Policy and Macro Watch")
    if policy_macro_items:
        render_grouped_articles(policy_macro_items)
    else:
        st.info("No policy or macro headlines were classified for this run.")

    if prediction["source_counts"]:
        st.markdown("#### Coverage Summary")
        coverage_line = " | ".join(f"{source}: {count}" for source, count in prediction["source_counts"].items())
        st.caption(coverage_line)

    if prediction.get("headquarters_country"):
        st.caption(
            f"Country-focused RSS uses Yahoo’s headquarters field when available ({prediction['headquarters_country']}). "
            "Headlines are still relevance-filtered to the company, ticker, sector, or macro terms so the list stays on-topic."
        )

    if prediction["coverage_notes"]:
        st.warning("Coverage guardrails: " + " ".join(prediction["coverage_notes"]))


def render_single_asset_dashboard(
    asset_name, asset_ticker, profile, benchmark_label, frame, metrics, prediction
):
    ivol_percentile = compute_ivol_percentile(frame["rolling_iv"])
    risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
    takeaway = build_takeaway(asset_name, risk_profile, metrics, benchmark_label)
    market_snapshot = get_market_snapshot(asset_ticker)
    idio_share = metrics["idio_vol_annual"] / metrics["total_vol_annual"] if metrics["total_vol_annual"] > 0 else np.nan

    st.markdown(f"**Selected Asset:** {asset_name} (`{asset_ticker}`)")
    profile_line = format_profile_line(profile)
    if profile_line:
        st.markdown(profile_line)

    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Beta", f"{metrics['beta']:.3f}")
    col2.metric("Alpha (daily)", f"{metrics['alpha_daily']:.5f}")
    col3.metric("R-squared", f"{metrics['r_squared']:.3f}")
    col4.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col5.metric("Annualized Total Volatility", f"{metrics['total_vol_annual']:.2%}")
    col6.metric("Annualized Idiosyncratic Volatility", f"{metrics['idio_vol_annual']:.2%}")

    if market_snapshot:
        st.subheader("Current Market Snapshot")
        snap1, snap2, snap3 = st.columns(3)
        price = market_snapshot.get("price")
        change_pct = market_snapshot.get("change_pct")
        market_time = market_snapshot.get("market_time")
        snap1.metric("Latest Market Price", f"{price:.2f}" if price is not None else "N/A")
        snap2.metric("Latest Change", f"{change_pct:.2%}" if change_pct is not None else "N/A")
        snap3.metric("Market Timestamp", str(market_time) if market_time is not None else "N/A")

    st.subheader("Historical Risk Context")
    ctx1, ctx2, ctx3, ctx4 = st.columns(4)
    ctx1.metric("Current Rolling IVOL", f"{metrics['rolling_iv_current']:.2%}" if pd.notna(metrics["rolling_iv_current"]) else "N/A")
    ctx2.metric("Average Rolling IVOL", f"{metrics['rolling_iv_average']:.2%}" if pd.notna(metrics["rolling_iv_average"]) else "N/A")
    ctx3.metric("IVOL Percentile", f"{ivol_percentile:.0f}th" if pd.notna(ivol_percentile) else "N/A")
    ctx4.metric("Firm-Specific Share of Risk", f"{idio_share:.0%}" if pd.notna(idio_share) else "N/A")

    st.subheader("Risk Interpretation")
    st.info(f"**Risk Profile:** {risk_profile}")
    st.success(f"**Investment Takeaway:** {takeaway}")
    st.subheader("Educational Outlook")
    st.warning(f"**Preferred Action:** {prediction['action']} ({prediction['confidence']} confidence)")
    st.write(prediction["summary"])
    st.caption(prediction["risk_text"])
    render_prediction_button()
    with st.expander("Is the prediction right? (sense-check)"):
        render_prediction_sense_check(prediction, metrics, asset_ticker, show_heading=False)

    if prediction["grouped_articles"]:
        with st.expander("Latest Evidence Used for Context"):
            for catalyst, items in prediction["grouped_articles"].items():
                st.markdown(f"**{catalyst.upper()}**")
                for item in items[:3]:
                    if item.get("url"):
                        st.markdown(f"- [{item['title']}]({item['url']})")
                    else:
                        st.markdown(f"- {item['title']}")
                    st.caption(item.get("source") or item.get("source_type", "Source"))
    elif prediction["coverage_notes"]:
        st.info(" | ".join(prediction["coverage_notes"]))

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Price Chart", "Return Scatter", "Rolling IVOL"])
    with chart_tab1:
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=frame.index, y=frame["asset_close"], mode="lines", name=asset_ticker))
        price_fig.add_trace(go.Scatter(x=frame.index, y=frame["benchmark_close"], mode="lines", name=benchmark_label))
        price_fig.update_layout(xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(price_fig, use_container_width=True)
    with chart_tab2:
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=frame["benchmark_return"],
                y=frame["asset_return"],
                mode="markers",
                name="Daily Returns",
            )
        )
        scatter_fig.update_layout(
            title=f"{asset_ticker} Return vs {benchmark_label} Return",
            xaxis_title=f"{benchmark_label} Return",
            yaxis_title=f"{asset_ticker} Return",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    with chart_tab3:
        rolling_fig = go.Figure()
        rolling_fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["rolling_iv"],
                mode="lines",
                name="Rolling Idiosyncratic Volatility",
            )
        )
        rolling_fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Idiosyncratic Volatility")
        st.plotly_chart(rolling_fig, use_container_width=True)

    st.subheader("Model Summary")
    results_df = pd.DataFrame(
        {
            "Metric": [
                "Alpha (daily)",
                "Beta",
                "R-squared",
                "Daily Idiosyncratic Volatility",
                "Current Rolling IVOL",
                "IVOL Percentile",
            ],
            "Value": [
                metrics["alpha_daily"],
                metrics["beta"],
                metrics["r_squared"],
                metrics["idio_vol_daily"],
                metrics["rolling_iv_current"],
                ivol_percentile / 100 if pd.notna(ivol_percentile) else np.nan,
            ],
        }
    )
    st.dataframe(results_df, use_container_width=True)


def render_compare_dashboard(name_a, ticker_a, profile_a, frame_a, metrics_a, name_b, ticker_b, profile_b, frame_b, metrics_b, benchmark_label):
    st.subheader("Side-by-Side Comparison")
    left, right = st.columns(2)
    snapshot_a = get_market_snapshot(ticker_a)
    snapshot_b = get_market_snapshot(ticker_b)
    with left:
        st.markdown(f"### {name_a} (`{ticker_a}`)")
        line_a = format_profile_line(profile_a)
        if line_a:
            st.markdown(line_a)
        if snapshot_a:
            st.caption(format_market_caption(snapshot_a))
    with right:
        st.markdown(f"### {name_b} (`{ticker_b}`)")
        line_b = format_profile_line(profile_b)
        if line_b:
            st.markdown(line_b)
        if snapshot_b:
            st.caption(format_market_caption(snapshot_b))

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Beta",
                "Alpha (daily)",
                "R-squared",
                "Annualized Return",
                "Annualized Total Volatility",
                "Annualized Idiosyncratic Volatility",
                "Current Rolling IVOL",
            ],
            name_a: [
                metrics_a["beta"],
                metrics_a["alpha_daily"],
                metrics_a["r_squared"],
                metrics_a["annualized_return"],
                metrics_a["total_vol_annual"],
                metrics_a["idio_vol_annual"],
                metrics_a["rolling_iv_current"],
            ],
            name_b: [
                metrics_b["beta"],
                metrics_b["alpha_daily"],
                metrics_b["r_squared"],
                metrics_b["annualized_return"],
                metrics_b["total_vol_annual"],
                metrics_b["idio_vol_annual"],
                metrics_b["rolling_iv_current"],
            ],
        }
    )
    st.dataframe(comparison_df, use_container_width=True)

    normalized_a = frame_a["asset_close"] / frame_a["asset_close"].iloc[0]
    normalized_b = frame_b["asset_close"] / frame_b["asset_close"].iloc[0]
    benchmark_norm_a = frame_a["benchmark_close"] / frame_a["benchmark_close"].iloc[0]

    tabs = st.tabs(["Normalized Performance", "Rolling IVOL Comparison", "Quick Interpretation"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_a.index, y=normalized_a, mode="lines", name=ticker_a))
        fig.add_trace(go.Scatter(x=normalized_b.index, y=normalized_b, mode="lines", name=ticker_b))
        fig.add_trace(go.Scatter(x=benchmark_norm_a.index, y=benchmark_norm_a, mode="lines", name=benchmark_label))
        fig.update_layout(xaxis_title="Date", yaxis_title="Normalized Value (Start = 1)")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_a.index, y=frame_a["rolling_iv"], mode="lines", name=ticker_a))
        fig.add_trace(go.Scatter(x=frame_b.index, y=frame_b["rolling_iv"], mode="lines", name=ticker_b))
        fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Idiosyncratic Volatility")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        profile_label_a = classify_risk_profile(metrics_a["r_squared"], metrics_a["idio_vol_annual"], metrics_a["total_vol_annual"])
        profile_label_b = classify_risk_profile(metrics_b["r_squared"], metrics_b["idio_vol_annual"], metrics_b["total_vol_annual"])
        st.info(
            f"{name_a} is classified as **{profile_label_a}**, while {name_b} is classified as **{profile_label_b}**."
        )
        if metrics_a["idio_vol_annual"] > metrics_b["idio_vol_annual"]:
            st.warning(f"{name_a} has the higher annualized idiosyncratic volatility in this sample.")
        else:
            st.warning(f"{name_b} has the higher annualized idiosyncratic volatility in this sample.")


stock_catalog = get_stock_catalog()
stock_labels = stock_catalog["label"].tolist()
stock_label_to_ticker = dict(zip(stock_catalog["label"], stock_catalog["ticker"]))
default_stock_label = next((label for label in stock_labels if label.endswith("(AAPL)")), stock_labels[0] if stock_labels else "")

initialize_app_state()

st.sidebar.header("Controls")
st.sidebar.caption("Extended build: multi-source news, prediction details, and forward sense-check. For the original Yahoo-only lab, run `app_classic.py` separately.")

asset_universe = st.sidebar.radio("Asset Universe", ["S&P 500 Stocks", "Cryptocurrencies"])

if asset_universe == "S&P 500 Stocks":
    analysis_mode = st.sidebar.radio(
        "Stock Mode",
        ["Single Company", "Compare Two Companies", "Full S&P 500 Screener"],
    )

    if analysis_mode == "Single Company":
        selected_label = st.sidebar.selectbox("Search company name", stock_labels, index=stock_labels.index(default_stock_label))
    elif analysis_mode == "Compare Two Companies":
        selected_label_a = st.sidebar.selectbox("Company 1", stock_labels, index=stock_labels.index(default_stock_label))
        default_b = next((label for label in stock_labels if label.endswith("(MSFT)")), stock_labels[1] if len(stock_labels) > 1 else stock_labels[0])
        selected_label_b = st.sidebar.selectbox("Company 2", stock_labels, index=stock_labels.index(default_b))
    else:
        st.sidebar.caption("Runs a broad cross-sectional screen across the current S&P 500 list. This can take longer.")
else:
    analysis_mode = st.sidebar.radio(
        "Crypto Mode",
        ["Single Crypto", "Compare Two Cryptos"],
    )
    crypto_labels = list(CRYPTO_OPTIONS.keys())
    if analysis_mode == "Single Crypto":
        selected_crypto_label = st.sidebar.selectbox("Choose cryptocurrency", crypto_labels, index=0)
        benchmark_crypto_options = [label for label in crypto_labels if label != selected_crypto_label]
        selected_crypto_benchmark = st.sidebar.selectbox("Crypto benchmark", benchmark_crypto_options, index=0)
    else:
        selected_crypto_a = st.sidebar.selectbox("Crypto 1", crypto_labels, index=0)
        remaining = [label for label in crypto_labels if label != selected_crypto_a]
        selected_crypto_b = st.sidebar.selectbox("Crypto 2", remaining, index=0)
        benchmark_options = [label for label in crypto_labels if label not in {selected_crypto_a, selected_crypto_b}]
        selected_crypto_benchmark = st.sidebar.selectbox("Comparison benchmark", benchmark_options, index=0 if benchmark_options else 0)

with st.sidebar.form("analysis_form"):
    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-02"))
    end_date = st.date_input("End date", value=pd.Timestamp.today().date())
    rolling_window = st.slider("Rolling window (days)", 20, 120, 30)
    fast_news = st.checkbox(
        "Faster loading (lighter news)",
        value=True,
        help="Parallel fetches with fewer sources: skips GDELT, country newspapers, and central-bank extras. Uncheck for fuller coverage (slower).",
    )
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    request_state = {
        "asset_universe": asset_universe,
        "analysis_mode": analysis_mode,
        "start_date": start_date,
        "end_date": end_date,
        "rolling_window": rolling_window,
        "fast_news": fast_news,
    }
    if asset_universe == "S&P 500 Stocks":
        if analysis_mode == "Single Company":
            request_state["selected_label"] = selected_label
        elif analysis_mode == "Compare Two Companies":
            request_state["selected_label_a"] = selected_label_a
            request_state["selected_label_b"] = selected_label_b
    else:
        if analysis_mode == "Single Crypto":
            request_state["selected_crypto_label"] = selected_crypto_label
            request_state["selected_crypto_benchmark"] = selected_crypto_benchmark
        else:
            request_state["selected_crypto_a"] = selected_crypto_a
            request_state["selected_crypto_b"] = selected_crypto_b
            request_state["selected_crypto_benchmark"] = selected_crypto_benchmark

    st.session_state["analysis_request"] = request_state
    st.session_state["analysis_view"] = "dashboard"

analysis_request = st.session_state.get("analysis_request")
if not analysis_request:
    st.info("Use the sidebar to choose stocks or crypto, set the date range, and click Run Analysis.")
    st.stop()

asset_universe = analysis_request["asset_universe"]
analysis_mode = analysis_request["analysis_mode"]
start_date = analysis_request["start_date"]
end_date = analysis_request["end_date"]
rolling_window = analysis_request["rolling_window"]
fast_news = analysis_request.get("fast_news", True)

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

if asset_universe == "S&P 500 Stocks":
    if analysis_mode == "Single Company":
        ticker = stock_label_to_ticker[analysis_request["selected_label"]]
        profile = get_company_profile(ticker)
        frame = load_asset_vs_benchmark(ticker, STOCK_MARKET_BENCHMARK, start_date, end_date)
        if frame is None or frame.empty:
            st.error("No stock data was downloaded. Please try another date range.")
            st.stop()
        _, enriched, metrics = compute_analysis(frame, rolling_window)
        risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
        market_snapshot = get_market_snapshot(ticker)
        with st.spinner("Gathering public news and policy context..."):
            news_context = get_news_context(
                ticker,
                profile["name"],
                profile.get("sector"),
                profile.get("industry"),
                profile.get("country"),
                fast_news=fast_news,
            )
        classified_context = classify_catalysts(news_context, sector=profile.get("sector"), industry=profile.get("industry"))
        prediction = build_prediction_signal(profile["name"], metrics, risk_profile, market_snapshot, news_context, classified_context)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Company:** {profile['name']}")
        if profile.get("sector"):
            st.sidebar.markdown(f"**Sector:** {profile['sector']}")
        if profile.get("industry"):
            st.sidebar.markdown(f"**Industry:** {profile['industry']}")
        render_date_messages(metrics, start_date, end_date)
        if st.session_state.get("analysis_view") == "prediction":
            render_prediction_detail_page(profile["name"], ticker, profile, STOCK_MARKET_BENCHMARK, metrics, risk_profile, market_snapshot, prediction)
        else:
            render_single_asset_dashboard(
                profile["name"], ticker, profile, STOCK_MARKET_BENCHMARK, enriched, metrics, prediction
            )

    elif analysis_mode == "Compare Two Companies":
        ticker_a = stock_label_to_ticker[analysis_request["selected_label_a"]]
        ticker_b = stock_label_to_ticker[analysis_request["selected_label_b"]]
        if ticker_a == ticker_b:
            st.error("Please choose two different companies to compare.")
            st.stop()
        profile_a = get_company_profile(ticker_a)
        profile_b = get_company_profile(ticker_b)
        frame_a = load_asset_vs_benchmark(ticker_a, STOCK_MARKET_BENCHMARK, start_date, end_date)
        frame_b = load_asset_vs_benchmark(ticker_b, STOCK_MARKET_BENCHMARK, start_date, end_date)
        if frame_a is None or frame_a.empty or frame_b is None or frame_b.empty:
            st.error("One of the selected companies could not be loaded for the chosen date range.")
            st.stop()
        _, enriched_a, metrics_a = compute_analysis(frame_a, rolling_window)
        _, enriched_b, metrics_b = compute_analysis(frame_b, rolling_window)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Company 1:** {profile_a['name']}")
        st.sidebar.markdown(f"**Company 2:** {profile_b['name']}")
        render_date_messages(metrics_a, start_date, end_date)
        render_compare_dashboard(
            profile_a["name"], ticker_a, profile_a, enriched_a, metrics_a,
            profile_b["name"], ticker_b, profile_b, enriched_b, metrics_b,
            STOCK_MARKET_BENCHMARK,
        )

    else:
        st.subheader("Full S&P 500 Screener")
        st.write(
            "This section screens the current S&P 500 list and ranks companies by annualized idiosyncratic volatility "
            "for the selected date range."
        )
        with st.spinner("Running the S&P 500 screen. This may take a moment..."):
            screener_df = build_sp500_screener(start_date, end_date, rolling_window)
        if screener_df.empty:
            st.error("The S&P 500 screener could not be generated for the chosen date range.")
            st.stop()
        top_high_ivol = screener_df.head(10)
        top_low_ivol = screener_df.tail(10).sort_values("Annualized Idiosyncratic Volatility")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Top 10 Highest IVOL Stocks")
            st.dataframe(top_high_ivol, use_container_width=True)
        with col_right:
            st.markdown("#### Top 10 Lowest IVOL Stocks")
            st.dataframe(top_low_ivol, use_container_width=True)
        st.markdown("#### Full S&P 500 Results")
        st.dataframe(screener_df, use_container_width=True)
        st.download_button(
            "Download full S&P 500 screener as CSV",
            screener_df.to_csv(index=False).encode("utf-8"),
            file_name="sp500_ivol_screener.csv",
            mime="text/csv",
        )

else:
    if analysis_mode == "Single Crypto":
        crypto_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_label"]]
        benchmark_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_benchmark"]]
        frame = load_asset_vs_benchmark(crypto_ticker, benchmark_ticker, start_date, end_date)
        if frame is None or frame.empty:
            st.error("Crypto data could not be loaded for the chosen date range.")
            st.stop()
        _, enriched, metrics = compute_analysis(frame, rolling_window)
        profile = {"name": analysis_request["selected_crypto_label"].replace(f" ({crypto_ticker})", ""), "sector": "Digital Asset", "industry": "Cryptocurrency"}
        risk_profile = classify_risk_profile(metrics["r_squared"], metrics["idio_vol_annual"], metrics["total_vol_annual"])
        market_snapshot = get_market_snapshot(crypto_ticker)
        with st.spinner("Gathering public news and policy context..."):
            news_context = get_news_context(
                crypto_ticker,
                profile["name"],
                profile.get("sector"),
                profile.get("industry"),
                profile.get("country"),
                fast_news=fast_news,
            )
        classified_context = classify_catalysts(news_context, sector=profile.get("sector"), industry=profile.get("industry"))
        prediction = build_prediction_signal(profile["name"], metrics, risk_profile, market_snapshot, news_context, classified_context)
        render_date_messages(metrics, start_date, end_date)
        if st.session_state.get("analysis_view") == "prediction":
            render_prediction_detail_page(profile["name"], crypto_ticker, profile, benchmark_ticker, metrics, risk_profile, market_snapshot, prediction)
        else:
            render_single_asset_dashboard(
                analysis_request["selected_crypto_label"],
                crypto_ticker,
                profile,
                benchmark_ticker,
                enriched,
                metrics,
                prediction,
            )

    else:
        crypto_ticker_a = CRYPTO_OPTIONS[analysis_request["selected_crypto_a"]]
        crypto_ticker_b = CRYPTO_OPTIONS[analysis_request["selected_crypto_b"]]
        benchmark_ticker = CRYPTO_OPTIONS[analysis_request["selected_crypto_benchmark"]]
        frame_a = load_asset_vs_benchmark(crypto_ticker_a, benchmark_ticker, start_date, end_date)
        frame_b = load_asset_vs_benchmark(crypto_ticker_b, benchmark_ticker, start_date, end_date)
        if frame_a is None or frame_a.empty or frame_b is None or frame_b.empty:
            st.error("One of the selected cryptocurrencies could not be loaded for the chosen date range.")
            st.stop()
        _, enriched_a, metrics_a = compute_analysis(frame_a, rolling_window)
        _, enriched_b, metrics_b = compute_analysis(frame_b, rolling_window)
        profile_a = {"name": analysis_request["selected_crypto_a"].replace(f" ({crypto_ticker_a})", ""), "sector": "Digital Asset", "industry": "Cryptocurrency"}
        profile_b = {"name": analysis_request["selected_crypto_b"].replace(f" ({crypto_ticker_b})", ""), "sector": "Digital Asset", "industry": "Cryptocurrency"}
        render_date_messages(metrics_a, start_date, end_date)
        render_compare_dashboard(
            analysis_request["selected_crypto_a"], crypto_ticker_a, profile_a, enriched_a, metrics_a,
            analysis_request["selected_crypto_b"], crypto_ticker_b, profile_b, enriched_b, metrics_b,
            benchmark_ticker,
        )

st.caption(
    "Method: CAPM-style regression of asset daily returns on a chosen benchmark return. "
    "Idiosyncratic volatility is measured as the standard deviation of regression residuals. "
    "The S&P 500 screener uses the current constituent list and may take longer to load."
)