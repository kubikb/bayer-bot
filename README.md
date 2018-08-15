# Bayer Zsolt Bot
Manapság egyre több hírt olvashatunk a mesterséges intelligencia legutóbbi vívmányairól, melyek emberi tevékenységeket próbálnak több-kevesebb sikerrel ellátni. Szöveggeneráló robotra is több példát láttunk már. Készült már robot a [Trónok Harca legújabb kötetének megírására](https://motherboard.vice.com/en_us/article/evvq3n/game-of-thrones-winds-of-winter-neural-network), [Donald Trump stílusában tweetelésre](https://twitter.com/deepdrumpf), vagy [tudományos cikkek generálására](https://pdos.csail.mit.edu/archive/scigen/#examples). Kevés példát találhatunk azonban magyar nyelvű szöveget író robotokra, bár az ehhez szükséges eszközök ingyen bárki rendelkezésére állnak.

A dokumentum bemutatja, hogy miképp hoztam létre a robotot, aki a [BayerBot blogon](https://bayerbot.blogspot.com/) osztja meg a "gondolatait" az alapján, amit Bayer Zsolt blogján olvasottakból tanult. A célom ezzel az, hogy bemutassam, mennyire egyszerű is egy hasonló, magyar nyelven "beszélő" robot létrehozása. Egy neurális hálót (pontosabban egy többrétegű Recurrent Neural Network-öt) fogunk az alábbiakban építeni, mely Bayer Zsolt blogposztjai alapján próbálja meg egy karakterlánc következő karakterét megjósolni. A program futtatásához szükségünk lesz a Python 3.6-os verziójára. Javaslom az [Anaconda](https://www.anaconda.com/download/) legújabb verziójának használatát.

## Dependenciák
A neurális háló építését a [textgenrnn](https://github.com/minimaxir/textgenrnn) Python csomaggal fogjuk végezni, melyhez a [Google Tensorflow](https://www.tensorflow.org/) ökoszisztémát is telepítenünk kell. Amennyiben van Nvidia videókártyánk, mely támogatja a CUDA-t, jelentősen felgyorsíthatjuk a neurális háló becslését a `tensorflow-gpu` dependencia használatával.

````
pip install tensorflow textgenrnn
````

## Adat beolvasása
Gyűjtsük be Bayer Zsolt blogjának posztjait az internetről. Az adatfájlt megtaláljuk egy [régi hobbiprojektem Github repozitóriumában](https://github.com/kubikb/bayer_blog_parser).

````
import pandas as pd

text_df = pd.read_csv(
    "https://raw.githubusercontent.com/kubikb/bayer_blog_parser/master/data/posts_lemmatized.tsv",
    sep = "\t",
    encoding = "UTF-8"
)
````

1359 blogposzttal fogunk dolgozni, melyek 2018. június 14-e előtt kerültek fel.

Számunkra a `full_content` oszlop lesz érdekes, mely a blogposzt teljes tartalma mellett a poszt címét is tartalmazza. Gyűjtsük ki ezt a változót lista formátumban.

````
texts = text_df.full_content.tolist()
````

## Modellezés
Mint írtam, a neurális háló építéséhez a `textgenrnn` csomagot fogjuk használni, mely nagyon megkönnyíti a folyamatot és jó eredményt produkál rövid modelltréning után is. Hozzunk létre egy új modellt `bayer_bot` néven.

````
from textgenrnn import textgenrnn

model = textgenrnn(name="bayer_bot")
````

Indítsuk el a modellbecslést az összes blogposzton. A paraméterek a következőek.
- `train_size` - Az adatmennyiség mekkora aránya legyen a véletlenszerűen kiválasztott tréning halmaz. A maradék lesz a validációs halmaz.
- `max_length` - Legfeljebb ennyi előző karaktert vegyünk figyelembe a következő jóslásához
- `num_epochs` - Összesen hányszor haladjunk végig az adatmennyiségen a modellbecslés során.
- `gen_epochs` - Minden második körben írassunk ki a programmal általa generált példaszövegeket, hogy így felügyeljük a folyamatot
- `batch_size` - A tréningre használt egységek mérete. Ha videókártyát használunk a neurális hálózat építéséhez, vegyük magasabbra az értéket.

GPU-s tréning esetén kb. **1,5 órát** fut. CPU-n futtatva lényegesen lassabb lesz.

````
model.train_on_texts(
    new_model=True,
    texts=texts,
    train_size=0.7,
    max_length=40,
    num_epochs=10,
    gen_epochs=2,
    batch_size=1024 # Csak tensorflow-gpu esetén
)
````

A modell futás közben visszajelzést ad a jelenlegi állapotról. A `loss` érték mutatja a modellilleszkedést, illetve láthatjuk, hogy várhatóan mennyi idő van még vissza a jelenlegi lépés végrehajtásából.

````
Training new model w/ 2-layer, 128-cell LSTMs
Training on 3,775,063 character sequences.
Epoch 1/10
  11/3686 [..............................] - ETA: 0:32:37 - loss: 3.8728
````

Időközönként kapunk példaszövegeket is:

````
####################
Temperature: 0.2
####################
Ez el a határok kis és a kormány a felesége a felelős szerint a szíriai között , a magyar politikai előtt a magyar is terroristák , a halálot , a megválasztások eléget . A szabadság a szerint a kis és az egy oldalán meg a szerint a magyar kormány a magyar kormány a terrori szerint a kormány az ell

A HÍR 12. „ Magyar Idők a kormány szerint a magyar kormány szerint a katolikus előtt , az olasz szerint a magyar kormány a felmentésére . A szerint a magyar választások , a szerint az egy kormány a szabadság , a szerint a szerint a kis évek , a magyar nem tudom , hogy a belegelete a felesége , a s

EGY NÉPMŰVÉSZET MESTEREI 197.
````

## A neurális hálózat használata
A modellezés eredménye elérhető lesz az alábbi fájlokban: `bayer_bot_config.json`, `bayer_bot_vocab.json`, `bayer_bot_weights.hdf5`. Ez lehetővé teszi, hogy később a modell újbóli építése nélkül használjuk azt és tetszőleges mennyiségű új szöveget generálhassunk. Egy korábban megépített modellt így tudunk használni:

````
model = textgenrnn(
    weights_path="bayer_bot_weights.hdf5",
    vocab_path="bayer_bot_vocab.json",
    config_path="bayer_bot_config.json"
)
````

Egy példaszöveg generálásánál több beállítást is megadhatunk:
- `temperature` - A modell "kreativitásának" foka.
- `max_gen_length` - A generálandó szöveg maximális hossza
- `prefix` - Tartalom a szöveglánc elejére, amihez a modell hozzáfűzi a többit az általa tanultak alapján
- `return_as_list` - Egy új lista elemeként adja vissza a generált szöveget

Próbáljuk meg 0.3-mas "kreativitással".

Egy példaszöveg generálásánál több beállítást is megadhatunk:
- `temperature` - A modell "kreativitásának" foka.
- `max_gen_length` - A generálandó szöveg maximális hossza
- `prefix` - Tartalom a szöveglánc elejére, amihez a modell hozzáfűzi a többit az általa tanultak alapján
- `return_as_list` - Egy új lista elemeként adja vissza a generált szöveget

Próbáljuk meg 0.3-mas "kreativitással".

````
model.generate(
    temperature = 0.3,
    prefix = "Liberális"
)

# Liberális Szabad Times , a német kormány a magyar kormány terrorizmus ellen , amit az amerikai kormány a vallásról . A megfelelő felelősségét valami szerint a közösség megszervezését . Az emberek meg kell többen kimenni . Az egyetlen testvérei ellen kapcsolatban , hogy a történetek szerint a magya
````

Magasabb `temperature` már nehezebben értelmezhető, zagyva szöveget eredményez.

````
model.generate(
    temperature = 1.0,
    prefix = "Liberális"
)

# Liberálisan mai vádgalt : „ Egy nagy szép példánatkia a pop (a tanulmány : bennerült már meg nem . - ugyanis mindenfele bírált hegyi , aki 23-as kismilleszkedéseket az akadályait Orbán Viktor mellépályásában . Ugyanakkor már a zavart , ezzel annyira azért itt a felelősségében ! Köszönjük az elmúlt
````

## Posztolás a blogra
Egy Blogger blogra való posztolás is automatizálható. Ehhez szükségünk lesz egy [Google Cloud](https://console.cloud.google.com/apis) profilra és projektre, melyben engedélyezzük a [Blogger API](https://developers.google.com/blogger/docs/3.0/using) használatát. Példakód [ITT](./post_to_blog.py)