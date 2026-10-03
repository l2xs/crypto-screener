# crypto-screener
Helps find coins/stocks with available entry points for trades

Disclaimer:
This software is for educational purposes only. Do not risk money you cannot afford to lose.
Use the software at your own risk. The authors and affiliates assume no responsibility for your trading results. The signals produced shall not be followed blindly. It has no use if you don't have the necessary experience and knowledge.

How to use
1. Install Python from python.org
2. Open Windows PowerShell
3. cd "path to screener.py"

The following outputs can be used with LLMs to sanity-check it
   
python screener.py       # gives coins with triggers to enter trades

python screener.py --min-score 2       # see more candidates, looser bar

python screener.py --min-score 4       # only perfect convergence

python screener.py --signal       # full deterministic trade setup

Support further development:
USDT TRC20: TZ3Xc2j2Qu9DpLP6VnCJ2GzLM8D9VN85XW
