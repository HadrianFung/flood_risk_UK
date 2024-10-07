#############################
The Team Thames Flood Tool
#############################


This package implements a flood risk prediction and visualization tool.

.. figure:: ../media/prediction.JPG
   :alt: Prediction

Installation Instructions
-------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to install package.

Run the command below in the directory where flood_tool directory and setup.py are

.. code:: bash

   pip install .
   pip install imbalanced-learn  


Quick Usage guide
-----------------

The ``tool.py`` module offers prediction tools for flood risk, property values, and local authorities etc.


.. code:: python

   import flood_tool as ft

   tool=ft.Tool()

Further Documentation
---------------------

.. toctree::
   :maxdepth: 2

   models
   coordinates
   visualization


Function APIs
-------------

.. automodule:: flood_tool
  :members:
  :imported-members:

Contributing
-------------

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate. # License

`MIT <https://choosealicense.com/licenses/mit/>`__
