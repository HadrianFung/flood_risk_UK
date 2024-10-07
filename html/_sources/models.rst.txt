====================================
Regression and Classification models
====================================



\ **Flood Risk Assessment**\ : Predicts flood risk based on postcodes ,
OSGB36 locations (easting, northings) or
WGS84_locations(latitudes,longitudes). Specify the model name you want
to use when do training or prediction.

When you predict from postcode, you can use: 

=========================== ============================
model name                  meaning
=========================== ============================
RF_riskLabel_from_postcode  RandomForestClassifier
LR_riskLabel_from_postcode  Logistic Regression
KNN_riskLabel_from_postcode K-Nearest Neighbours (KNN)
=========================== ============================

When you predict from location, you can use: 

=========================== ============================
model name                  meaning
=========================== ============================
RF_riskLabel_from_location  RandomForestClassifier
LR_riskLabel_from_location  Logistic Regression
KNN_riskLabel_from_location K-Nearest Neighbours (KNN)
=========================== ============================


.. code:: python

   # example
   tool.train(models=['RF_riskLabel_from_postcode'])

   tool.predict_flood_class_from_postcode(postcodes=['NE29 7EN', 'S31 8QF', 'YO19 6HT'],method='RF_riskLabel_from_postcode')

\ **Historic Flooding Prediction**\ : Assesses the boolean of historic
flooding based on postcodes.

===================== ============================
model name            meaning
===================== ============================
MLP_historic_flooding Multi-layer Perceptron (MLP)
RF_historic_flooding  Random Forest (RF)
KNN_historic_flooding K-Nearest Neighbours (KNN)
===================== ============================

.. code:: python

   # example
   tool.train(models=['KNN_historic_flooding'])

   tool.predict_historic_flooding(postcodes=['NE29 7EN', 'S31 8QF', 'YO19 6HT'],method='KNN_historic_flooding')

\ **House Price Estimation**\ : Estimates median house prices based on
postcodes.

================ ================================
model name       meaning
================ ================================
GBR_median_price GradientBoosting Regressor (GBR)
KNR_median_price KNeighbors Regressor (KNR)
RFR_median_price RandomForest Regressor (RFR)
================ ================================

.. code:: python

   # example
   tool.train(models=['GBR_median_price'])

   tool.predict_median_house_price(postcodes=['NE29 7EN', 'S31 8QF', 'YO19 6HT'],method='GBR_median_price')

\ **Local Authority Prediction**\ : Estimates the local authority for
OSGB36 locations (easting, northings).

+-----------------------------------+-----------------------------------+
| model name                        | meaning                           |
+===================================+===================================+
| NCA_KNN_local_authority           | K-Nearest Neighbours (KNN) with   |
|                                   | Neighbourhood Component Analysis  |
|                                   | (NCA)                             |
+-----------------------------------+-----------------------------------+
| SVC_local_authority               | Support Vector Classifier (SVC)   |
+-----------------------------------+-----------------------------------+
| RF_local_authority                | Random Forest Classifier (RF)     |
+-----------------------------------+-----------------------------------+

.. code:: python

   # example
   tool.train(models=['SVC_local_authority'])

   tool.predict_local_authority(eastings = [445771, 395560],
       northings = [515362, 397900],method='SVC_local_authority')
