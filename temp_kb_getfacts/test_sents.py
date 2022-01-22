# coding: utf-8
import en_core_web_md
from extract_facts import OpenRE_get_facts
from IPython import embed

txt = 'NVIDIA Shares Higher Tuesday After Company Reports Pegasus Platform For Autonomous Cars NVIDIA (NVDA) shares are higher on Tuesday morning after the company unveiled a Pegasus computing platform for autonomous cars and a partnership with logistics firms DHL and ZF to power their self-driving truck fleet starting in 2019, according to reports.\r\n\r\nInsider at Oracle Acquires Stock Via Option/Derivative Security Sells Portion to Pay Tax, Buy Trend Intact On Oct 05, 2017, Dorian Daley, EVP, Genl Counsel, Secretary, exercised.\r\n\r\noptions/derivative securities for 12,500 shares of Oracle (ORCL) and sold 6,523 shares in the company for $318,975 to meet tax obligations. After accounting for tax obligations this transaction resulted in a net acquisition of 5,977 company shares.  Subsequent to the Form 4 filing with the SEC, Daley has 36,954 shares of the company, with a market value, based on the prior-day closing price, of $1.79 million.\r\n\r\nGeneral Motors Insider Continues 90-Day Selling Trend Alicia S Boler-Davis, EVP, reported a sale of 10,000 shares in General Motors (GM) on Oct 06, 2017, for $450,000.  Following the Form 4 filing with the SEC, Boler-Davis has 33,912 shares of the company, which have a market value of $1.54 million as of the prior-day closing price.'

nlp = en_core_web_md.load()
doc = nlp(txt)
for i, sent in enumerate(doc.sents):
    print(i, sent)

embed()
