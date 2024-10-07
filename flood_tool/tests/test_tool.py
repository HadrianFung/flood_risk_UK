"""Test flood tool."""

import numpy as np
import pandas as pd

from pytest import mark

import flood_tool.tool as tool


testtool = tool.Tool()

def test_lookup_easting_northing():
    """Check"""

    data = testtool.lookup_easting_northing(["M34 7QL"])

    assert np.isclose(data.iloc[0].easting, 393470).all()
    assert np.isclose(data.iloc[0].northing, 394371).all()


def test_lookup_lat_long():
    """Check"""

    data = testtool.lookup_lat_long(["M34 7QL"])

    assert np.isclose(data.iloc[0].latitude, 53.4461, rtol=1.0e-3).all()
    assert np.isclose(data.iloc[0].longitude, -2.0997, rtol=1.0e-3).all()

def test_predict_flood_class_from_postcode():
    testtool.train(models=['RF_riskLabel_from_postcode'])
    postcodes=['OL9 7NS']
    method = "RF_riskLabel_from_postcode"

    result = testtool.predict_flood_class_from_postcode(postcodes,method)

    assert isinstance(result, pd.Series) 

def test_predict_flood_class_from_OSGB36_location():
    testtool.train(models=['RF_riskLabel_from_location'])
    eastings = [445771]
    northings = [515362]
    method = "RF_riskLabel_from_location"

    result = testtool.predict_flood_class_from_OSGB36_location(eastings, northings, method)

    assert isinstance(result, pd.Series) 

def test_predict_flood_class_from_WGS84_location():
    testtool.train(models=['RF_riskLabel_from_location'])
    longitudes = [-2.0997]
    latitudes = [53.4461]
    method = "RF_riskLabel_from_location"

    result = testtool.predict_flood_class_from_WGS84_locations(longitudes,latitudes, method)

    assert isinstance(result, pd.Series) 

def test_predict_median_house_price():
    testtool.train(models=['GBR_median_price'])
    postcodes=['OL9 7NS']
    method = "GBR_median_price"

    result = testtool.predict_median_house_price(postcodes, method)

    assert isinstance(result, pd.Series)

def test_predict_local_authority():
    testtool.train(models=['SVC_local_authority'])
    eastings = [445771]
    northings = [515362]
    method = "SVC_local_authority"

    result = testtool.predict_local_authority(eastings,northings,method)
    assert isinstance(result, pd.Series)

def test_predict_historic_flooding():
    testtool.train(models=['RF_historic_flooding'])
    postcodes=['NE65 7NU']
    method = "RF_historic_flooding"

    result = testtool.predict_historic_flooding(postcodes, method)

    assert isinstance(result, pd.Series)

if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
    test_predict_flood_class_from_postcode()
    test_predict_flood_class_from_OSGB36_location()
    test_predict_median_house_price()
    test_predict_local_authority()
    test_predict_historic_flooding()