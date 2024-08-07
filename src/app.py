from flask import Flask,request, render_template
import numpy as np
import pickle
import os
import json

#loading models
Jmodel = pickle.load(open('jmodel.pkl','rb'))
Wmodel = pickle.load(open('wmodel.pkl','rb'))
Cmodel = pickle.load(open('cmodel.pkl','rb'))
Smodel = pickle.load(open('smodel.pkl','rb'))
Bmodel = pickle.load(open('bmodel.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

#flask app
app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('homepage.html')    


@app.route('/predict')
def predict():
    return render_template('predict.html') 


@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        commoditytype = request.form['commodityname']
        month  = request.form['month']
        Year = request.form['year']
        NextYear = int(Year) + 1
        average_rain_fall = request.form['average_rain_fall'] 

        avgPriceyear = []
        mspyear = []
        mspnextyear = []
        avgPriceNextyear = []
        months_labels = ["jan","Feb","March","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
        monthcount = 1
        yearcount = 2021

        rainfall2024 = [1,2,3,1,8,673,1318,779,408,106,44,8]
        rainfall2023 = [90,7.2,29.9,41.4,67.6,148.6,315.9,162.7,190.3,50.8,9.3,8]
        x_count =0 

        cropimges = ['/static/images/jowarlogo.webp',
                     '/static/images/wheatlogo.avif',
                     '/static/images/cottonlogo.jpg',
                     '/static/images/sugarcanelogo.jpg',
                     '/static/images/bajralogo.jpg'
                    ]
        

        features = np.array([[month,Year,average_rain_fall]],dtype=object)
        transformed_features = preprocessor.transform(features)

        if(commoditytype == "Jowar"):
            cropface = cropimges[0]
            prediction = Jmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1550)/100,2)
            max_value = round((predicted_value*2970)/100,2)
            avg_value = round((min_value + max_value) / 2,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Jmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1550)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Jmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1550)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1

        elif(commoditytype == "Wheat"):
            cropface = cropimges[1]
            prediction = Wmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1350)/100,2)
            max_value = round((predicted_value*2125)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Wmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2125)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1350)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Wmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2125)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Cotton"):
            cropface = cropimges[2]
            prediction = Cmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*3600)/100,2)
            max_value = round((predicted_value*6080)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Cmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*6080)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*3600)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Cmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*6080)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*3600)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Sugarcane"):
            cropface = cropimges[3]
            prediction = Smodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*2250)/100,2)
            max_value = round((predicted_value*2775)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Smodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2775)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2250)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Smodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2775)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Bajara"):
            cropface = cropimges[4]
            prediction = Bmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1175)/100,2)
            max_value = round((predicted_value*2350)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Bmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2350)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1175)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Bmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2350)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1



        maxmspyear = max(mspyear)
        maxavgPriceyear = max(avgPriceyear)

        goldmonthindex = mspyear.index(maxmspyear) + 1

        minmspyear = min(mspyear)
        minavgPriceyear = min(avgPriceyear)

        silvermonthindex = mspyear.index(minmspyear) + 1
        


        return render_template('result.html',prediction = predicted_value,
                                             cropface = cropface,
                                             min_value = min_value,
                                             max_value = max_value,
                                             avg_value= avg_value,
                                             year = Year,
                                             NextYear = NextYear,
                                             month = month,
                                             maxhigh = maxmspyear,
                                             maxlow = maxavgPriceyear,
                                             minhigh = minmspyear,
                                             minlow = minavgPriceyear,
                                             goldmonth = goldmonthindex,
                                             silvermonth = silvermonthindex,
                                             months_labels = months_labels,
                                             mspyear = json.dumps(mspyear),
                                             minPriceYear = json.dumps(avgPriceyear),
                                             mspnextyear = json.dumps(mspnextyear),
                                             minPriceNextYear = json.dumps(avgPriceNextyear)
                                             )
        

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)