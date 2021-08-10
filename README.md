# Steam Review Analyzer
## Dependencies
 + vue
 + npm
 + flask
 + sklearn
## Repositroy
 + Remove joke.py and Test.py
 + Analyzer.py: the main file for the project conatining several sub classes
 + Aquisitions.py: the file for building a training data set from the Steam API
 + VUE contains a client and a server folder
    + Server:
        + App.py: A flask app router
        + API.py: A class for the flask app to call
    + Client:
        + src: contains the vue app code
        + src/components: conatins all page components
## Plan
Create a data processing pipeline to clean steam reviews for NLP
Create a model to predict if a review will be liked
Create a model to show trends in text*
## Results
Random Forest:
<br />
    On Validation:
    |  A | B  |  C | D | E  |
    |---|---|---|---|---|
    |  1 | 1  | 1  | 1  |  1 |
    |  1 | 1  | 1  | 1  |  2 |
    |  1 | 1  | 1  | 1  |  3 |
    
    <br />
    On Test:
    |   |   |   |   |   |
    |---|---|---|---|---|
    |   |   |   |   |   |
    |   |   |   |   |   |
    |   |   |   |   |   |
XGB:
    <br />
    On Validation:
    |  A | B  |  C | D | E  |
    |---|---|---|---|---|
    |  1 | 1  | 1  | 1  |  1 |
    |  1 | 1  | 1  | 1  |  2 |
    |  1 | 1  | 1  | 1  |  3 |
    
    <br />
    On Test:
    |   |   |   |   |   |
    |---|---|---|---|---|
    |   |   |   |   |   |
    |   |   |   |   |   |
    |   |   |   |   |   |

## Misc.
word cloud
graphs
### Todo:
clean up website
expand model site interactivity
add live calls to site
