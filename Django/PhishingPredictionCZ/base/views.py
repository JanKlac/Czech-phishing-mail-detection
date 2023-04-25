from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import FormView
from .forms import *
import pickle
import simplemma
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re
import sys


import pytesseract
from PIL import Image
def home(request):
    return render(request, "home.html")

def modellabel(request):
    return render(request, "modellabel.html")

def modeltext(request):
    return render(request, "modeltext.html")

def getPredictions(nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance):
    model = pickle.load(open("/home/JanKlac/PhishingPredictionCZ/ml_model.sav", 'rb'))
    scaled = pickle.load(open("/home/JanKlac/PhishingPredictionCZ/scaler.sav", 'rb'))

    prediction = model.predict(scaled.transform([
        [nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance]
    ]))

    if prediction == 0:
        return 'no'
    elif prediction == 1:
        return 'yes'
    else:
        return 'error'

def cz_stem(word, aggressive=False):
    if not re.match("^\\w+$", word):
        return word
    if not word.islower() and not word.istitle() and not word.isupper():
        print("warning: skipping word with mixed case: {}".format(word),
              file=sys.stderr)
        return word
    s = word.lower() # all our pattern matching is done in lowercase
    s = _remove_case(s)
    s = _remove_possessives(s)
    if aggressive:
        s = _remove_comparative(s)
        s = _remove_diminutive(s)
        s = _remove_augmentative(s)
        s = _remove_derivational(s)
    if word.isupper():
        return s.upper()
    if word.istitle():
        return s.title()
    return s

def _remove_case(word):
    if len(word) > 7 and word.endswith("atech"):
        return word[:-5]
    if len(word) > 6:
        if word.endswith("ětem"):
            return _palatalise(word[:-3])
        if word.endswith("atům"):
            return word[:-4]
    if len(word) > 5:
        if word[-3:] in {"ech", "ich", "ích", "ého", "ěmi", "emi", "ému",
                         "ete", "eti", "iho", "ího", "ími", "imu"}:
            return _palatalise(word[:-2])
        if word[-3:] in {"ách", "ata", "aty", "ých", "ama", "ami",
                         "ové", "ovi", "ými"}:
            return word[:-3]
    if len(word) > 4:
        if word.endswith("em"):
            return _palatalise(word[:-1])
        if word[-2:] in {"es", "ém", "ím"}:
            return _palatalise(word[:-2])
        if word[-2:] in {"ům", "at", "ám", "os", "us", "ým", "mi", "ou"}:
            return word[:-2]
    if len(word) > 3:
        if word[-1] in "eiíě":
            return _palatalise(word)
        if word[-1] in "uyůaoáéý":
            return word[:-1]
    return word

def _remove_possessives(word):
    if len(word) > 5:
        if word[-2:] in {"ov", "ův"}:
            return word[:-2]
        if word.endswith("in"):
            return _palatalise(word[:-1])
    return word

def _remove_comparative(word):
    if len(word) > 5:
        if word[-3:] in {"ejš", "ějš"}:
            return _palatalise(word[:-2])
    return word

def _remove_diminutive(word):
    if len(word) > 7 and word.endswith("oušek"):
        return word[:-5]
    if len(word) > 6:
        if word[-4:] in {"eček", "éček", "iček", "íček", "enek", "ének",
                         "inek", "ínek"}:
            return _palatalise(word[:-3])
        if word[-4:] in {"áček", "aček", "oček", "uček", "anek", "onek",
                         "unek", "ánek"}:
            return _palatalise(word[:-4])
    if len(word) > 5:
        if word[-3:] in {"ečk", "éčk", "ičk", "íčk", "enk", "énk",
                         "ink", "ínk"}:
            return _palatalise(word[:-3])
        if word[-3:] in {"áčk", "ačk", "očk", "učk", "ank", "onk",
                         "unk", "átk", "ánk", "ušk"}:
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in {"ek", "ék", "ík", "ik"}:
            return _palatalise(word[:-1])
        if word[-2:] in {"ák", "ak", "ok", "uk"}:
            return word[:-1]
    if len(word) > 3 and word[-1] == "k":
        return word[:-1]
    return word

def _remove_augmentative(word):
    if len(word) > 6 and word.endswith("ajzn"):
        return word[:-4]
    if len(word) > 5 and word[-3:] in {"izn", "isk"}:
        return _palatalise(word[:-2])
    if len(word) > 4 and word.endswith("ák"):
        return word[:-2]
    return word

def _remove_derivational(word):
    if len(word) > 8 and word.endswith("obinec"):
        return word[:-6]
    if len(word) > 7:
        if word.endswith("ionář"):
            return _palatalise(word[:-4])
        if word[-5:] in {"ovisk", "ovstv", "ovišt", "ovník"}:
            return word[:-5]
    if len(word) > 6:
        if word[-4:] in {"ásek", "loun", "nost", "teln", "ovec", "ovík",
                         "ovtv", "ovin", "štin"}:
            return word[:-4]
        if word[-4:] in {"enic", "inec", "itel"}:
            return _palatalise(word[:-3])
    if len(word) > 5:
        if word.endswith("árn"):
            return word[:-3]
        if word[-3:] in {"ěnk", "ián", "ist", "isk", "išt", "itb", "írn"}:
            return _palatalise(word[:-2])
        if word[-3:] in {"och", "ost", "ovn", "oun", "out", "ouš",
                         "ušk", "kyn", "čan", "kář", "néř", "ník",
                         "ctv", "stv"}:
            return word[:-3]
    if len(word) > 4:
        if word[-2:] in {"áč", "ač", "án", "an", "ář", "as"}:
            return word[:-2]
        if word[-2:] in {"ec", "en", "ěn", "éř", "íř", "ic", "in", "ín",
                         "it", "iv"}:
            return _palatalise(word[:-1])
        if word[-2:] in {"ob", "ot", "ov", "oň", "ul", "yn", "čk", "čn",
                         "dl", "nk", "tv", "tk", "vk"}:
            return word[:-2]
    if len(word) > 3 and word[-1] in "cčklnt":
        return word[:-1]
    return word

def _palatalise(word):
    if word[-2:] in {"ci", "ce", "či", "če"}:
        return word[:-2] + "k"

    if word[-2:] in {"zi", "ze", "ži", "že"}:
        return word[:-2] + "h"

    if word[-3:] in {"čtě", "čti", "čtí"}:
        return word[:-3] + "ck"

    if word[-3:] in {"ště", "šti", "ští"}:
        return word[:-3] + "sk"
    return word[:-1]

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ("light", "aggressive"):
        sys.exit("usage: {} light|aggressive".format(sys.argv[0]))
    aggressive = sys.argv[1] == "aggressive"
    for line in sys.stdin:
        print(*[cz_stem(word, aggressive=aggressive)
                for word in line.split()])

def image_to_text(request):
    if request.method == 'POST':
        image = request.FILES['image']
        image = Image.open(image)
        text = pytesseract.image_to_string(image, lang ="ces")
        textedited = text.replace("\n", " ").replace("\r", " ").replace(".","").replace(",","").split()

        stopwords = ['a', 'aby', 'ahoj', 'aj', 'ale', 'anebo', 'ani', 'aniž', 'ano', 'asi', 'aspoň', 'atd',
                     'atp', 'az', 'ačkoli', 'až', 'bez', 'beze', 'blízko', 'bohužel', 'brzo', 'bude', 'budem',
                     'budeme', 'budes', 'budete', 'budeš', 'budou', 'budu', 'by', 'byl', 'byla', 'byli', 'bylo',
                     'byly', 'bys', 'byt', 'být', 'během', 'chce', 'chceme', 'chcete', 'chceš', 'chci', 'chtít',
                     'chtějí', "chut'", 'chuti', 'ci', 'clanek', 'clanku', 'clanky', 'co', 'coz', 'což', 'cz', 'daleko',
                     'dalsi', 'další', 'den', 'deset', 'design', 'devatenáct', 'devět', 'dnes', 'do', 'dobrý', 'docela',
                     'dva', 'dvacet', 'dvanáct', 'dvě', 'dál', 'dále', 'děkovat', 'děkujeme', 'děkuji', 'email', 'ho',
                     'hodně', 'i', 'jak', 'jakmile', 'jako', 'jakož', 'jde', 'je', 'jeden', 'jedenáct', 'jedna', 'jedno',
                     'jednou', 'jedou', 'jeho', 'jehož', 'jej', 'jeji', 'jejich', 'její', 'jelikož', 'jemu', 'jen', 'jenom',
                     'jenž', 'jeste', 'jestli', 'jestliže', 'ještě', 'jež', 'ji', 'jich', 'jimi', 'jinak', 'jine', 'jiné',
                     'jiz', 'již', 'jsem', 'jses', 'jseš', 'jsi', 'jsme', 'jsou', 'jste', 'já', 'jí', 'jím', 'jíž', 'jšte',
                     'k', 'kam', 'každý', 'kde', 'kdo', 'kdy', 'kdyz', 'když', 'ke', 'kolik', 'kromě', 'ktera', 'ktere', 'kteri',
                     'kterou', 'ktery', 'která', 'které', 'který', 'kteři', 'kteří', 'ku', 'kvůli', 'ma', 'mají', 'mate', 'me', 'mezi',
                     'mi', 'mit', 'mne', 'mnou', 'mně', 'moc', 'mohl', 'mohou', 'moje', 'moji', 'možná', 'muj', 'musí', 'muze', 'my',
                     'má', 'málo', 'mám', 'máme', 'máte', 'máš', 'mé', 'mí', 'mít', 'mě', 'můj', 'může', 'na', 'nad', 'nade', 'nam',
                     'napiste', 'napište', 'naproti', 'nas', 'nasi', 'načež', 'naše', 'naši', 'ne', 'nebo', 'nebyl', 'nebyla',
                     'nebyli', 'nebyly', 'nechť', 'nedělají', 'nedělá', 'nedělám', 'neděláme', 'neděláte', 'neděláš', 'neg', 'nejsi',
                     'nejsou', 'nemají', 'nemáme', 'nemáte', 'neměl', 'neni', 'není', 'nestačí', 'nevadí', 'nez', 'než', 'nic',
                     'nich', 'nimi', 'nove', 'novy', 'nové', 'nový', 'nula', 'ná', 'nám', 'námi', 'nás', 'náš', 'ní', 'ním', 'ně',
                     'něco', 'nějak', 'někde', 'někdo', 'němu', 'němuž', 'o', 'od', 'ode', 'on', 'ona', 'oni', 'ono', 'ony', 'osm',
                     'osmnáct', 'pak', 'patnáct', 'po', 'pod', 'podle', 'pokud', 'potom', 'pouze', 'pozdě', 'pořád', 'prave', 'pravé',
                     'pred', 'pres', 'pri', 'pro', 'proc', 'prostě', 'prosím', 'proti', 'proto', 'protoze', 'protože', 'proč', 'prvni',
                     'první', 'práve', 'pta', 'pět', 'před', 'přede', 'přes', 'přese', 'při', 'přičemž', 're', 'rovně', 's', 'se', 'sedm',
                     'sedmnáct', 'si', 'sice', 'skoro', 'smí', 'smějí', 'snad', 'spolu', 'sta', 'sto', 'strana', 'sté', 'sve', 'svych',
                     'svym', 'svymi', 'své', 'svých', 'svým', 'svými', 'svůj', 'ta', 'tady', 'tak', 'take', 'takhle', 'taky', 'takze',
                     'také', 'takže', 'tam', 'tamhle', 'tamhleto', 'tamto', 'tato', 'te', 'tebe', 'tebou', "ted'", 'tedy', 'tema', 'ten',
                     'tento', 'teto', 'ti', 'tim', 'timto', 'tipy', 'tisíc', 'tisíce', 'to', 'tobě', 'tohle', 'toho', 'tohoto', 'tom',
                     'tomto', 'tomu', 'tomuto', 'toto', 'trošku', 'tu', 'tuto', 'tvoje', 'tvá', 'tvé', 'tvůj', 'ty', 'tyto', 'téma', 'této',
                     'tím', 'tímto', 'tě', 'těm', 'těma', 'těmu', 'třeba', 'tři', 'třináct', 'u', 'určitě', 'uz', 'už', 'v', 'vam', 'vas',
                     'vase', 'vaše', 'vaši', 've', 'vedle', 'večer', 'vice', 'vlastně', 'vsak', 'vy', 'vám', 'vámi', 'vás', 'váš', 'více',
                     'však', 'všechen', 'všechno', 'všichni', 'vůbec', 'vždy', 'z', 'za', 'zatímco', 'zač', 'zda', 'zde', 'ze', 'zpet',
                     'zpravy', 'zprávy', 'zpět', 'čau', 'či', 'článek', 'článku', 'články', 'čtrnáct', 'čtyři', 'šest', 'šestnáct', 'že']

        def lemmatizeandstop(inputtext):
            textstopped = []
            finaltext = []

            for i in inputtext:
                if i not in stopwords and len(i) > 1 and not any(c.isdigit() for c in i):
                    textstopped.append(cz_stem(simplemma.lemmatize(i, lang='cs').lower(), aggressive=False))

            finaltext.append(' '.join(textstopped))

            return(finaltext)

        textlemmastop = lemmatizeandstop(textedited)

        # coun_vect = CountVectorizer(lowercase=False)

        # count_matrix = coun_vect.fit_transform(textlemmastop)
        # count_array = count_matrix.toarray()
        # df = pd.DataFrame(data=count_array,columns=coun_vect.get_feature_names_out())

        def getPredictionsText(input):
            cv = CountVectorizer()
            model = joblib.load("/home/JanKlac/PhishingPredictionCZ/logreg.pkl")
            train = open("/home/JanKlac/PhishingPredictionCZ/train2", "rb")
            X_train = pickle.load(train)
            cv.fit(X_train)

            # model = pickle.load(open('/home/JanKlac/PhishingPredictionCZ/ml_modeltext.sav', 'rb'))
            # scaled = pickle.load(open('/home/JanKlac/PhishingPredictionCZ/scalertext.sav', 'rb'))
            sentence = cv.transform(input).toarray()
            prediction = model.predict(sentence)[0]
            # prediction = model.predict(scaled.transform([[input]]))

            if prediction == 0:
                return 'no'
            elif prediction == 1:
                return 'yes'
            else:
                return 'error'


        return render(request, 'result2.html', {'text': text, "textlemmastop": textlemmastop, "prediction": getPredictionsText(textlemmastop)})

    return render(request, 'modeltext.html')

def result(request):
    nalehani = str(request.GET.get('nalehani', 0))
    lukrativni = str(request.GET.get('lukrativni', 0))
    hyperlink = str(request.GET.get('hyperlink', 0))
    priloha = str(request.GET.get('priloha', 0))
    vizual = str(request.GET.get('podezrela', 0))
    gramatika = str(request.GET.get('gramatika', 0))
    finance = str(request.GET.get('finance', 0))

    if nalehani == "yes":
        nalehani = 1
    else:
        nalehani = 0

    if lukrativni == "yes":
        lukrativni = 1
    else:
        lukrativni = 0

    if hyperlink == "yes":
        hyperlink = 1
    else:
        hyperlink = 0

    if priloha == "yes":
        priloha = 1
    else:
        priloha = 0

    if vizual == "yes":
        vizual = 1
    else:
        vizual = 0

    if gramatika == "yes":
        gramatika = 1
    else:
        gramatika = 0

    if finance == "yes":
        finance = 1
    else:
        finance = 0

    result = getPredictions(nalehani, lukrativni, hyperlink, priloha, vizual, gramatika, finance)

    return render(request, 'result.html', {'result': result})

class HomeView(FormView):
    form_class = UploadForm
    template_name = 'index.html'
    success_url = '/'

    def form_valid(self, form):
        upload = self.request.FILES['file']
        return super().form_valid(form)
