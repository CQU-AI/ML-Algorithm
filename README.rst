ML-Algorithm
==============

 |Code quality|_ |PyPI|_ |Downloads| 

Introduction
------------

Implement machine learning algorithms with python without sklearn. Some classes are design especially for CQU-ML course by Professor.He


Install
-------

-  With pip : ``pip3 install cquai-ml``
-  With src : Clone or fork this project, then build it with
   ``python3 setup.py install``


Usage
------------
In most cases, API in this project is similar to scikit-learn project.

For example, if you want to run a decision tree classifier based on C4.5 (While scikit-learn use opt-CART instead of C4.5)

.. code:: python

    from cquai_ml import DecisionTreeClassifier
    from sklearn.datasets import load_breast_cancer # get a dataset

    X, y = load_breast_cancer(return_X_y=True)

    clf1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
    pred1 = clf1.predict(X)


Contributing
------------
Everyone is welcomed to contribute!

We currently provides:
 - DatasetSpace
 - UnionHypothesisSpace
 - LinearRegression
 - LogisticRegression
 - LinearDiscriminantAnalysis
 - KNeighborsClassifier
 - DecisionTreeRegressor
 - DecisionTreeClassifier

.. |Code quality| image:: https://api.codacy.com/project/badge/Grade/29ea1d846d5c433bbb9a5183a908f0a5
.. _Code quality: https://app.codacy.com/manual/loopyme/ML-Algorithm/dashboard

.. |PyPI| image:: https://img.shields.io/pypi/v/cquai-ml
.. _PyPI: https://pypi.org/project/cquai-ml/

.. |Downloads| image:: https://pepy.tech/badge/cquai-ml

