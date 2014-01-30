#Cortana

Experiments on domain classification problem.

### Status

Benchmark algorithm finished. It got **0.83713331397** accuracy on the data aggregated from all the slot.test.tsv files in each domain.

### Data placement structure

We intend to place the data and code in the data oriented manner. There are four steps of data processing: 

**Featurization**. The raw training data is converted from sentence into a vector of feature. For instance, in 'Bag of Words' model, each sentence will be represented as a huge sparse vector.

**Feature Learning**. We apply various method to refine the data from featurization step. We try combine PCA, LDA, reprensentation learning and other methods to find the best representation of the raw feature. This step will be extremely time consuming.

**Training Models**. We try different models and tune the hyper-parameters for each one.

**Evaluation**. We evaluate the performance of models in testing data. This step is seprated from the last one for fexibility of implementation. Because the same model can be used for differnt latter steps, be it hosting for real testing or testing on different testing data.

--------------------------------------------------

Accordingly, the data can be categoried into the five types(one more for the raw data and some data cleaning). Specifically, five folders will be used to place the data.

**raw_data**. We put the training data obtained from archive here. Inside it, *raw_data/aggregated* will be a single file containing all sentences from each domain, for traning and testing each.

**featurized**. We put featurized data inside here. Each sub-folder will be a way of featurization. For instance, *featurized/bow* will be the bag of words model.

**rep**. Here lives the data transformed from representation learning and/or dimention reduction. As above, each sub-folder will be the output of one learning method.

I suggest we keep *.tsv* extention for the original data and use *.dat* for the data we transformed or generated.TSV files and DAT files are ignored in .gitignore. Also I think we should use *train.dat* for the file name of any training data in the above folders.

**models**. We place trained model here. Any serialization method will be fine but I suggest we use **.model* for the file name. Files suffixed by *.model* are also ignored.

**eval**. We place the test results here.

### Conventions

Each step will fetch traning/input data from a foler in the previous step and write to its own folder. For example, if there's one method that run LDA to reduce dimention and get input from 'Bag of Words' model. It should read input from *featurized/bow/train.dat* and write the transformed data to *rep/lda/train.dat*.

Each folder that contains an algorithm should have a **cmd** file. Inside the cmd file is the command that will run the algorithm and generate the data. We should put the description of the algorithm, assumption of the input data and other information in this cmd file as comments. In case the description get too lengthy, we add a README.md file in the *current* folder.

### Idears worth exploration

1. Categrize rare terms like numbers into class tag. EX, replace all four digit numbers into ...
    "TAG.FOUR-DIGITS".

2. Using the Naive-Bayes model, the accuracy on 8-fold cross validation is around 91%, significatly higher than on testing set, is this because the training data and testing data are heterogeneous?
