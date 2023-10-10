from flask import Flask
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request,send_file
import pickle
import json

flask_app = Flask(__name__)
#1 VARIABLES
NITRAT = pickle.load(open("models/1v/NITRAT", "rb"))
PHSPHT = pickle.load(open("models/1v/PHSPHT", "rb"))
SILCAT = pickle.load(open("models/1v/SILCAT", "rb"))
TCARBN = pickle.load(open("models/1v/TCARBN", "rb"))
TMP = pickle.load(open("models/1v/TMP", "rb"))
#2VARIABLES
NITRAT_PHSPHT= pickle.load(open("models/2v/NITRAT PHSPHT", "rb"))
NITRAT_TMP= pickle.load(open("models/2v/NITRAT TMP", "rb"))
PHSPHT_TMP= pickle.load(open("models/2v/PHSPHT TMP", "rb"))
SILCAT_NITRAT= pickle.load(open("models/2v/SILCAT NITRAT", "rb"))
SILCAT_PHSPHT= pickle.load(open("models/2v/SILCAT PHSPHT", "rb"))
SILCAT_TCARBN= pickle.load(open("models/2v/SILCAT TCARBN", "rb"))
SILCAT_TMP= pickle.load(open("models/2v/SILCAT TMP", "rb"))
TCARBN_NITRAT= pickle.load(open("models/2v/TCARBN NITRAT", "rb"))
TCARBN_PHSPHT= pickle.load(open("models/2v/TCARBN PHSPHT", "rb"))
TCARBN_TMP= pickle.load(open("models/2v/TCARBN TMP", "rb"))
#3Variables
NITRAT_PHSPHT_TMP= pickle.load(open("models/3v/NITRAT PHSPHT TMP", "rb"))
SILCAT_NITRAT_PHSPHT= pickle.load(open("models/3v/SILCAT NITRAT PHSPHT", "rb"))
SILCAT_NITRAT_TMP= pickle.load(open("models/3v/SILCAT NITRAT TMP", "rb"))
SILCAT_PHSPHT_TMP= pickle.load(open("models/3v/SILCAT PHSPHT TMP", "rb"))
SILCAT_TCARBN_NITRAT= pickle.load(open("models/3v/SILCAT TCARBN NITRAT", "rb"))
SILCAT_TCARBN_PHSPHT= pickle.load(open("models/3v/SILCAT TCARBN PHSPHT", "rb"))
SILCAT_TCARBN_TMP= pickle.load(open("models/3v/SILCAT TCARBN TMP", "rb"))
TCARBN_NITRAT_PHSPHT= pickle.load(open("models/3v/TCARBN NITRAT PHSPHT", "rb"))
TCARBN_NITRAT_TMP= pickle.load(open("models/3v/TCARBN NITRAT TMP", "rb"))
TCARBN_PHSPHT_TMP= pickle.load(open("models/3v/TCARBN PHSPHT TMP", "rb"))
#4Variables
SILCAT_NITRAT_PHSPHT_TMP = pickle.load(open("models/4v/SILCAT NITRAT PHSPHT TMP", "rb"))
SILCAT_TCARBN_NITRAT_PHSPHT = pickle.load(open("models/4v/SILCAT TCARBN NITRAT PHSPHT", "rb"))
SILCAT_TCARBN_NITRAT_TMP = pickle.load(open("models/4v/SILCAT TCARBN NITRAT TMP", "rb"))
SILCAT_TCARBN_PHSPHT_TMP = pickle.load(open("models/4v/SILCAT TCARBN PHSPHT TMP", "rb"))
TCARBN_NITRAT_PHSPHT_TMP = pickle.load(open("models/4v/TCARBN NITRAT PHSPHT TMP", "rb"))
#5Variables
_5v= pickle.load(open("models/5v/5v", "rb"))


@flask_app.route("/")
def Home():

    return render_template('index.html')



@flask_app.route("/predict", methods=["POST"])
def predict():
    prediction=[0.0]
    def features(float_features):
        ft = list(map(float, float_features))
        fts = np.array(ft)
        fts = fts.reshape(1, -1)
        return fts

    input1 = request.json['input1']
    input2 = request.json['input2']
    input3 = request.json['input3']
    input4 = request.json['input4']
    input5 = request.json['input5']

    inputs = [input1,input4,input5,input2,input3]



    vars=[]
    for i  in inputs:

        if (i!={}):
            vars.append(int(i))


    #vars.reverse()
    print(vars)
    if (vars == [1]):
        tmp = request.json['tmp']
        float_features = [tmp]
        prediction = TMP.predict(features(float_features))
    if (vars == [2]):
        tcarb = request.json['tcarb']
        print(tcarb)
        float_features = [tcarb]
        prediction = TCARBN.predict(features(float_features))
    if (vars == [3]):
        silc = request.json['silc']
        print(silc)
        float_features = [silc]
        prediction = SILCAT.predict(features(float_features))
    if (vars == [4]):
        phspht = request.json['phspht']
        float_features = [phspht]
        prediction = PHSPHT.predict(features(float_features))
    if (vars == [5]):
        nit = request.json['nit']
        print(nit)
        float_features = [nit]
        prediction = NITRAT.predict(features(float_features))
#2Variables
    if (vars == [1,2]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        float_features = [tcarb,tmp]
        prediction = TCARBN_TMP.predict(features(float_features))
    if (vars == [1, 3]):
        tmp = request.json['tmp']
        silc = request.json['silc']
        float_features = [silc,tmp]
        prediction = SILCAT_TMP.predict(features(float_features))
    if (vars == [1, 4]):
        tmp = request.json['tmp']
        phspht = request.json['phspht']
        float_features = [phspht,tmp]
        prediction = PHSPHT_TMP.predict(features(float_features))
    if (vars == [1, 5]):
        tmp = request.json['tmp']
        nit = request.json['nit']
        float_features = [nit,tmp]
        prediction = NITRAT_TMP.predict(features(float_features))
    if (vars == [2, 3]):
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        float_features = [silc,tcarb]
        prediction = SILCAT_TCARBN.predict(features(float_features))
    if (vars == [2, 4]):
        tcarb = request.json['tcarb']
        phspht = request.json['phspht']
        float_features = [phspht,tcarb]
        prediction = TCARBN_PHSPHT.predict(features(float_features))
    if (vars == [2, 5]):
        tcarb = request.json['tcarb']
        nit = request.json['nit']
        float_features = [nit,tcarb]
        prediction = TCARBN_NITRAT.predict(features(float_features))
    if (vars == [3, 4]):
        silc = request.json['silc']
        phspht = request.json['phspht']
        float_features = [phspht,silc]
        prediction = SILCAT_PHSPHT.predict(features(float_features))
    if (vars == [3, 5]):
        silc = request.json['silc']
        nit = request.json['nit']
        float_features = [nit,silc]
        prediction = SILCAT_NITRAT.predict(features(float_features))
    if (vars == [4, 5]):
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [nit,phspht]
        prediction = NITRAT_PHSPHT.predict(features(float_features))
    # 3Variables
    if (vars == [1, 2,3]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        float_features = [silc,tcarb, tmp]
        prediction = SILCAT_TCARBN_TMP.predict(features(float_features))
    if (vars == [1, 3,4]):
        tmp = request.json['tmp']
        silc = request.json['silc']
        phspht = request.json['phspht']
        float_features = [silc,phspht, tmp]
        prediction = SILCAT_PHSPHT_TMP.predict(features(float_features))
    if (vars == [1, 4,5]):
        tmp = request.json['tmp']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [nit,phspht, tmp]
        prediction = NITRAT_PHSPHT_TMP.predict(features(float_features))
    if (vars == [1,3, 5]):
        tmp = request.json['tmp']
        silc = request.json['silc']
        nit = request.json['nit']
        float_features = [silc,nit, tmp]
        prediction = SILCAT_NITRAT_TMP.predict(features(float_features))
    if (vars == [2, 3,4]):
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        phspht = request.json['phspht']
        float_features = [silc, tcarb,phspht]
        prediction = SILCAT_TCARBN_PHSPHT.predict(features(float_features))
    if (vars == [1,2, 4]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        phspht = request.json['phspht']
        float_features = [tcarb,phspht, tmp]
        prediction = TCARBN_PHSPHT_TMP.predict(features(float_features))
    if (vars == [1,2, 5]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        nit = request.json['nit']
        float_features = [tcarb,nit,tmp]
        prediction = TCARBN_NITRAT_TMP.predict(features(float_features))
    if (vars == [3, 4,5]):
        silc = request.json['silc']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [silc,nit,phspht]
        prediction = SILCAT_NITRAT_PHSPHT.predict(features(float_features))
    if (vars == [2,3, 5]):
        silc = request.json['silc']
        nit = request.json['nit']
        tcarb = request.json['tcarb']
        float_features = [ silc,tcarb,nit]
        prediction = SILCAT_TCARBN_NITRAT.predict(features(float_features))
    if (vars == [2,4, 5]):
        phspht = request.json['phspht']
        nit = request.json['nit']
        tcarb = request.json['tcarb']
        float_features = [tcarb,nit, phspht]
        prediction = TCARBN_NITRAT_PHSPHT.predict(features(float_features))
    #4Variables
    if (vars == [1, 2,3,4]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        phspht = request.json['phspht']
        float_features = [silc,tcarb,phspht, tmp]
        prediction = SILCAT_TCARBN_PHSPHT_TMP.predict(features(float_features))
    if (vars == [1, 2,3,5]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        nit = request.json['nit']
        float_features = [silc,tcarb,nit, tmp]
        prediction = SILCAT_TCARBN_NITRAT_TMP.predict(features(float_features))
    if (vars == [1,2, 4,5]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [tcarb,nit,phspht, tmp]
        prediction = TCARBN_NITRAT_PHSPHT_TMP.predict(features(float_features))
    if (vars == [1, 3,4,5]):
        tmp = request.json['tmp']
        silc = request.json['silc']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [silc,nit,phspht, tmp]
        prediction = SILCAT_NITRAT_PHSPHT_TMP.predict(features(float_features))
    if (vars == [2, 3,4,5]):
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [silc, tcarb,nit,phspht]
        prediction = SILCAT_TCARBN_NITRAT_PHSPHT.predict(features(float_features))
    if (vars == [1,2, 3,4,5]):
        tmp = request.json['tmp']
        tcarb = request.json['tcarb']
        silc = request.json['silc']
        phspht = request.json['phspht']
        nit = request.json['nit']
        float_features = [silc, tcarb,nit,phspht,tmp]
        prediction = _5v.predict(features(float_features))




    return jsonify({'result': input1, 'predict' : round(float(prediction[0]),2) })


@flask_app.route("/predict2", methods=["POST"])
def predict2():
    f = request.files["file_upload"]
    jsondata = request.form['variables']

    input=json.loads(jsondata)

    input1 = input['input1']
    input2 = input['input2']
    input3 = input['input3']
    input4 = input['input4']
    input5 = input['input5']

    inputs = [input1, input4, input5, input2, input3]
    print(inputs)
    data = pd.read_csv(f,sep=',')
    prediction = [0.0]
    print(data.head())

    vars = []
    for i in inputs:
        if (i != {}):
            vars.append(int(i))

    # vars.reverse()
    print(vars)
    if (vars == [1]):
        df=data.loc[:,["TMP"]]
        prediction = TMP.predict(df)
    if (vars == [2]):
        df = data.loc[:, ["TCARBN"]]
        prediction = TCARBN.predict(df)

    if (vars == [3]):
        df = data.loc[:, ["SILCAT"]]

        prediction = SILCAT.predict(df)

    if (vars == [4]):
        df = data.loc[:, ["PHSPHT"]]
        prediction = PHSPHT.predict(df)
    if (vars == [5]):
        df = data.loc[:, [ "NITRAT"]]
        prediction = NITRAT.predict(df)
    # 2Variables
    if (vars == [1, 2]):
        df = data.loc[:, ["TCARBN","TMP"]]
        prediction = TCARBN_TMP.predict(df)
    if (vars == [1, 3]):
        df = data.loc[:, ["SILCAT","TMP"]]
        prediction = SILCAT_TMP.predict(df)
    if (vars == [1, 4]):

        df = data.loc[:, ["PHSPHT", "TMP"]]
        prediction = PHSPHT_TMP.predict(df)
    if (vars == [1, 5]):

        df = data.loc[:, ["NITRAT", "TMP"]]
        prediction = NITRAT_TMP.predict(df)
    if (vars == [2, 3]):

        df = data.loc[:, ["SILCAT", "TCARBN"]]
        prediction = SILCAT_TCARBN.predict(df)
    if (vars == [2, 4]):
        df = data.loc[:, ["TCARBN","PHSPHT"]]
        prediction = TCARBN_PHSPHT.predict(df)
    if (vars == [2, 5]):

        df = data.loc[:, ["TCARBN", "NITRAT"]]
        prediction = TCARBN_NITRAT.predict(df)
    if (vars == [3, 4]):

        df = data.loc[:, ["SILCAT", "PHSPHT"]]
        prediction = SILCAT_PHSPHT.predict(df)
    if (vars == [3, 5]):

        df = data.loc[:, ["SILCAT", "NITRAT"]]
        prediction = SILCAT_NITRAT.predict(df)

    if (vars == [4, 5]):

        df = data.loc[:, [ "NITRAT", "PHSPHT"]]
        prediction = NITRAT_PHSPHT.predict(df)

    # 3Variables
    if (vars == [1, 2, 3]):

        df = data.loc[:, ["SILCAT", "TCARBN", "TMP"]]
        prediction = SILCAT_TCARBN_TMP.predict(df)
    if (vars == [1, 3, 4]):

        df = data.loc[:, ["SILCAT",  "PHSPHT", "TMP"]]
        prediction = SILCAT_PHSPHT_TMP.predict(df)
    if (vars == [1, 4, 5]):

        df = data.loc[:, [ "NITRAT", "PHSPHT", "TMP"]]
        prediction = NITRAT_PHSPHT_TMP.predict(df)
    if (vars == [1, 3, 5]):
        df = data.loc[:, ["SILCAT",  "NITRAT", "TMP"]]
        prediction = SILCAT_NITRAT_TMP.predict(df)
    if (vars == [2, 3, 4]):

        df = data.loc[:, ["SILCAT", "TCARBN",  "PHSPHT"]]
        prediction = SILCAT_TCARBN_PHSPHT.predict(df)
    if (vars == [1, 2, 4]):

        df = data.loc[:, [ "TCARBN",  "PHSPHT", "TMP"]]
        prediction = TCARBN_PHSPHT_TMP.predict(df)
    if (vars == [1, 2, 5]):

        df = data.loc[:, [ "TCARBN", "NITRAT","TMP"]]
        prediction = TCARBN_NITRAT_TMP.predict(df)
    if (vars == [3, 4, 5]):

        df = data.loc[:, ["SILCAT", "NITRAT", "PHSPHT"]]
        prediction = SILCAT_NITRAT_PHSPHT.predict(df)
    if (vars == [2, 3, 5]):

        df = data.loc[:, ["SILCAT", "TCARBN", "NITRAT"]]
        prediction = SILCAT_TCARBN_NITRAT.predict(df)

    if (vars == [2, 4, 5]):

        df = data.loc[:, [ "TCARBN", "NITRAT", "PHSPHT"]]
        prediction = TCARBN_NITRAT_PHSPHT.predict(df)
    # 4Variables
    if (vars == [1, 2, 3, 4]):

        df = data.loc[:, ["SILCAT", "TCARBN",  "PHSPHT", "TMP"]]
        prediction = SILCAT_TCARBN_PHSPHT_TMP.predict(df)
    if (vars == [1, 2, 3, 5]):

        df = data.loc[:, ["SILCAT", "TCARBN", "NITRAT", "TMP"]]
        prediction = SILCAT_TCARBN_NITRAT_TMP.predict(df)
    if (vars == [1, 2, 4, 5]):

        df = data.loc[:, [ "TCARBN", "NITRAT", "PHSPHT", "TMP"]]
        prediction = TCARBN_NITRAT_PHSPHT_TMP.predict(df)
    if (vars == [1, 3, 4, 5]):

        df = data.loc[:, ["SILCAT",  "NITRAT", "PHSPHT", "TMP"]]
        prediction = SILCAT_NITRAT_PHSPHT_TMP.predict(df)
    if (vars == [2, 3, 4, 5]):
        df = data.loc[:, ["SILCAT","TCARBN","NITRAT","PHSPHT"]]
        prediction = SILCAT_TCARBN_NITRAT_PHSPHT.predict(df)
    if (vars == [1,2, 3, 4, 5]):
        df = data.loc[:, ["TMP", "PHSPHT", "NITRAT", "TCARBN", "SILCAT"]]

        prediction = _5v.predict(df)




    ph=list(prediction.astype(np.float64))
    ph_rounded=[]
    for n in ph:
        ph_rounded.append(round(n,2))
    date=list(data['mon/day/yr'])
    depth=list( data['DEPTH'])
    long=list(data['Longitude'])
    lat=list(data['Latitude'])
    data['PH']=ph_rounded
    print(ph)
    print(type(ph))
    print(type(depth))

    csv_file = "data.csv"
    data.to_csv(csv_file, index=False)

    return jsonify({ "download_url": f"/download-csv/{csv_file}",'date':date,'depth':depth,'long': long ,'lat' : lat,'ph':ph_rounded})

@flask_app.route("/download-csv/<path:filename>", methods=["GET"])
def download_csv(filename):
    try:
        return send_file(filename, as_attachment=True)
    except FileNotFoundError:
        print("error")
if __name__ == '__main__':
    flask_app.run(debug=True)