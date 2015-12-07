# pandas-learn

[![Travis](https://img.shields.io/travis/richlewis42/pandas-learn.svg?style=flat-square)](https://travis-ci.org/richlewis42/pandas-learn) [![Coveralls](https://img.shields.io/coveralls/richlewis42/pandas-learn.svg?style=flat-square)](https://coveralls.io/github/richlewis42/pandas-learn)

A python package adapting scikit-learn models to intelligently use pandas
data structures.

## Installation

To install package using pip, clone the repo, and from its root:

```bash
pip install .
```

## Usage

The project adapts scikit-learn models to take in and then output appropriately
indexed pandas objects, allowing for more seamless integration of these
excellent libraries. A minimal example:

```python
In [1]: from pdlearn.ensemble import RandomForestClassifier

In [2]: data = pd.read_csv('titanic-train.csv')           \
                 .join(pd.read_csv('titanic-test.csv'))   \
                 .set_index('name')
In [3]: data['sex'] = data.sex == 'male'
In [4]: X = data[['sex', 'pclass']]
In [5]: y = data['survived']
In [6]: train = y.notnull()
In [7]: rf = RandomForestClassifier()
In [8]: # index of test features, and target label included
In [9]: rf.fit(X[train], y[train]).predict(X[~train]).head()
Out[9]:
                                              survived
name                                                  
Kelly, Mr. James                                     0
Wilkes, Mrs. James (Ellen Needs)                     1
Myles, Mr. Thomas Francis                            0
Wirz, Mr. Albert                                     0
Hirvonen, Mrs. Alexander (Helga E Lindqvist)         1

In [10]: #columns are multiindexed by target-class pairs
In [11]: rf.predict_proba(X[~train]).head()
Out[11]:
                                              survived          
                                                     0         1
name                                                            
Kelly, Mr. James                              0.861595  0.138405
Wilkes, Mrs. James (Ellen Needs)              0.489031  0.510969
Myles, Mr. Thomas Francis                     0.854857  0.145143
Wirz, Mr. Albert                              0.861595  0.138405
Hirvonen, Mrs. Alexander (Helga E Lindqvist)  0.489031  0.510969

In [12]: rf.feature_importances_ #fitted model properties are also pandas
Out[12]:
sex       0.735979
pclass    0.264021
Name: feature_importances, dtype: float64
```

## Development

Install the dev dependencies with:

```bash
pip install -r requirements_dev.txt
```

[Pyinvoke](http://www.pyinvoke.org) is used to run the development tasks.

You can lint the project using [Pylint](pylint.org) with:
```bash
invoke linter
```

And run the tests using [pytest](pytest.org) and [nose](nose.readthedocs.org) with:
```bash
invoke tests
```
