Projekt: Predikcija cijene Bitcoina korištenjem neuronskih mreža i strojnog učenja

Ciljevi:
Glavni cilj ovog projekta bio je izraditi modele strojnog učenja za predikciju cijene Bitcoina na temelju povijesnih podataka o cijeni i volumenu trgovanja. Osnovna ideja bila je ispitati i usporediti performanse različitih pristupa: jednostavne feedforward neuronske mreže, LSTM mreže, GRU mreže te XGBoost regresijskog modela. Željeli smo provjeriti koji model najpreciznije predviđa cijenu Bitcoina, s posebnim naglaskom na tjednu i mjesečnu vremensku rezoluciju. Uz to, cilj je bio dobiti uvid u prednosti i nedostatke svake od korištenih arhitektura u kontekstu predikcije financijskih vremenskih serija.

Podaci su preuzeti u CSV formatu, sadržavali su dnevne vrijednosti cijene Bitcoina (Open, High, Low, Close) i volumena trgovanja.

Podaci su agregirani na tjednoj i mjesečnoj razini za potrebe izrade različitih modela.

README
Za pokretanje ovog projekta potrebne su sljedeće biblioteke i alati:

Python 3.x

PyTorch

Pandas

NumPy

Matplotlib

Scikit-learn

XGBoost

Instalacija potrebnih paketa:

pip install torch pandas numpy matplotlib scikit-learn xgboost

Pokretanje pojedinačnih modela:

Svaki model se nalazi u zasebnoj Python datoteci (npr. feedforward_7.py, lstm_m.py itd.).

Svaki model učitava CSV datoteku BTC-USD.csv s povijesnim podacima o cijeni Bitcoina.

Nakon izvršavanja svakog modela, generiraju se CSV datoteke s predikcijama koje se koriste za usporedne grafičke prikaze.

Pokretanje usporedbe grafova:

Datoteka usporedba_grafova.py učitava predikcije iz CSV datoteka i prikazuje grafičku usporedbu predikcija svih modela.

Podaci:

Povijesni podaci o Bitcoinu preuzeti su s platforme Kaggle, iz skupa podataka "https://www.kaggle.com/datasets/meetnagadia/bitcoin-stock-data-sept-17-2014-august-24-2021"

Napomena: Prilikom pokretanja osigurati da se sve datoteke nalaze u istom direktoriju kako bi učitavanje CSV podataka i predikcija bilo ispravno.

