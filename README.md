Iris Classifier

This repository provides a Python script to train the neural network model for classifying Iris Flower. The dataset is used for this classification task. 

# Project Installation & Installation Depedencies:

. Clone the repo: `https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course.git`

. Change the directory: `cd IrisClassifier`

. Install dependencies: `pip install -r requirements.txt`

# 1. Git 

usage of GitHub for the whole project time

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/commits/main/">See the commit history </a>

# 2. UML

A UML diagram is a diagram based on the UML (Unified Modeling Language) with the purpose of visually representing a system along with its main actors, roles, actions, artifacts or classes, in order to better understand, alter,maintain, or document information about the system. For this, I have chosen the following three diagrams

                          1. Activity Diagram
                          2. Sequence Diagram
                          3. Use Case Diagram

Click those link for displaying the diagram:


<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/UML/activity_diagram.pdf">See the Activity Diagram </a>


<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/UML/sequence_diagram%20(1).pdf">See the Sequence Diagram </a>


<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/UML/use_case_diagram.pdf">See the Use Case Diagram </a>


# 3. DDD

Domain Driven Design(DDD): Domain-driven design (DDD) is a software design approach focusing on modelling software to match a domain according to input from that domain's experts. The DDD of my project given below:

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/DDD/DDD.pdf">Click link for the DDD </a>

4. # Metrices:

Python offers some linters and formatters, for example, Flake8, pyflakes, pycodestyle, pylint. Among them I used `pycodestyle` and `pylint`. Those check code base against PEP8 programming style, programming errors(eg, library impoted but not used, undefined names, cyclomatic complexity etc).

The screenshot is given below:

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pycodestyle-classifier.png">Click link for displaying screenshot - using pycodestyle </a>

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pycodestyle-model-train.png">Click link for displaying screenshot - using pycodestyle </a>

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pycodestyle-unit-test.png">Click link for displaying screenshot - using pycodestyle </a>

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pylint-classifier-file.png">Click link for displaying screenshot - using pylint </a>

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pylint-model-train-file.png">Click link for displaying screenshot - using pylint </a>

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/metrics/pylint-unit-test.png">Click link for displaying screenshot - using pylint </a>



# 5. Clean Code Development:

CCD improved usage and readability as well as better maintain the code an appropriate manner:

A:

1. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/appropriate%20indention%20code%20from%20unit-test%20code.png">appropriate indention code </a>


2. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/small%20functions%20and%20methods%20from%20evaluate_metrics%20file.png">small functions and methods</a>


3. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/use%20comments%20and%20doc%20string%20from%20model%20train%20file.png">use comments and doc string </a>

4. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/appropriate%20name%20for%20variable%20and%20method.png">appropriate name for variable and method</a>


5. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/Error%20Handling%20from%20model_train%20file.png">Error Handling and Exception </a>


B:

CCD points(10 more points) that I would like to use in my next project:


<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/CCD/CCD-CHEAT-SHEET.md">Cheat Sheet Points</a>

# 6: Build

Usage of Pybuilder to build Project and have the ability to install and import as a package for usage in other projects.


<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/tree/main/target">Find files here </a>


# 7: UnitTests

UnitTest uncovers errors and accidental changes to functionality within a software. When adding features, unit tests reveal unintended effects on the existing system.

<a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/unittest/unit_test.py">Click the link for unittest code </a>


# 8: Continuous Delivery:

GitHub actions script is designed for Continuous Delivery(CD) of a Python application to an Azure Web App named "irisclassifierwebapp".


-> <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/.github/workflows/main_irisclassifierwebapp.yml">CD link</a>


# 9: IDE

Create my favorite shortcuts and customize for quicker development.

-> own shortcut: `option + r`(run)

->build-in:

. `cmd + f`(find)

. `cmd + r`(replace)

. `option + c/v/x`(copy/paste/cut)

. `shift + ctrl + d`(debugging)


# 10: DSL

DSL class for constructing NN models using PyTorch. The following code provides a simplified interface for building NN architectures and setting training parameters using custom syntax. 


here, the structure of DSL command line:

```bash
layer defination: layer input_features output_features

Activation Function: activation [ReLU/sigmoid/softmax]

Dropout: dropout [probability]

Training parameters: train [epochs] [learning_rate]

```

# 11: Functional Programming:

Some of the functional programming code is given below:

1. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/preprocessing/feature_engineering.py#L38C1-L40C44">mostly side effect free functions</a>

2. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/evaluation/evaluate_metrics.py#L48C1-L55C78">Anonymous functions</a>

3. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/dataset/load_dataset.py#L83-L103">Functions as parameters</a>

4. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/preprocessing/feature_engineering.py#L42C1-L65C68">Functions as return</a>

5. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/preprocessing/feature_engineering.py#L84C1-L93C34">Higher order functions</a>

6. <a href = "https://github.com/atikul-islam-sajib/atikul-islam-sajib-Software_Engineering_Course/blob/main/IrisClassifier/evaluation/evaluate_metrics.py#L69C1-L87C8">Only Final Data Structures</a>

