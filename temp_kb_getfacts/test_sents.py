# coding: utf-8
import en_core_web_md
from spacy.language import Language
from extract_facts import OpenRE_get_facts
from IPython import embed

txt = 'NVIDIA Shares Higher Tuesday After Company Reports Pegasus Platform For Autonomous Cars NVIDIA (NVDA) shares are higher on Tuesday morning after the company unveiled a Pegasus computing platform for autonomous cars and a partnership with logistics firms DHL and ZF to power their self-driving truck fleet starting in 2019, according to reports.\r\n\r\nInsider at Oracle Acquires Stock Via Option/Derivative Security Sells Portion to Pay Tax, Buy Trend Intact On Oct 05, 2017, Dorian Daley, EVP, Genl Counsel, Secretary, exercised.\r\n\r\noptions/derivative securities for 12,500 shares of Oracle (ORCL) and sold 6,523 shares in the company for $318,975 to meet tax obligations. After accounting for tax obligations this transaction resulted in a net acquisition of 5,977 company shares.  Subsequent to the Form 4 filing with the SEC, Daley has 36,954 shares of the company, with a market value, based on the prior-day closing price, of $1.79 million.\r\n\r\nGeneral Motors Insider Continues 90-Day Selling Trend Alicia S Boler-Davis, EVP, reported a sale of 10,000 shares in General Motors (GM) on Oct 06, 2017, for $450,000.  Following the Form 4 filing with the SEC, Boler-Davis has 33,912 shares of the company, which have a market value of $1.54 million as of the prior-day closing price.'

txt = """
Q2 Consensus Forecast for Newfield Exploration's Earnings Trends Higher The Q2 forecasted earnings estimate for Newfield Exploration's (NFX) for the quarter ending June 30, 2018 has been scaled up. The current consensus of $0.81 per share is higher than the previous consensus of $0.79 per share.

90-Day Insider Buying Trend Reduced with Disposition of JM Smucker Shares On Jun 19, 2018, SVP, Joseph Stanziano, executed a sale of 970 shares in JM Smucker (SJM) for $101,850.  After the Form 4 filing with the SEC, Stanziano has control over a total of 12,700 shares of the company, with 10,537 shares held directly and 2,163 controlled indirectly. The market value of the direct and indirect holdings, based on the prior-day closing price, is approximately $1.35 million.

 (KR) KROGER CO Fiscal Year EPS Range $2.00 - $2.15 

Wyndham Hotels &amp; Resorts Insider Sale for Taxes Adds to 90-Day Selling Trend Geoffrey A Ballotti, Director and President & Chief Exec Officer, made a sale of 16,278 shares of Wyndham Hotels & Resorts (WH) on Jun 19, 2018, for approximately $1,000,283 to satisfy tax obligations.  Ballotti, following the transactions defined in the Form 4 SEC filing, has 240,461 shares of the company, which have a market value of $14.78 million as of the prior-day closing price.

 Insider at Wyndham Hotels &amp; Resorts Sells for Tax Adds to 90-Day Selling Trend On Jun 19, 2018, Mary R Falvey, Chief Admin. Officer, completed a sale of 10,570 Wyndham Hotels & Resorts (WH) shares for approximately $649,527 to satisfy tax obligations.  After the Form 4 filing with the SEC, Falvey has 153,202 company shares, with a market value, based on the prior-day closing price, of $9.41 million.
"""

@Language.component('sent')
def set_custom_sentence_end_points(doc):
    for token in doc[:-1]:
        if token.text.startswith('\n') or token.text.startswith('\r'):
            doc[token.i + 1].is_sent_start = True
    return doc


nlp = en_core_web_md.load()
nlp.add_pipe('sent', before='parser')
doc = nlp(txt)
for i, sent in enumerate(doc.sents):
    print(i, sent)

embed()
