# Project Name : NLP ON EMAILS

Text mining

# STEPS TO BE FOLLOWED DURING EDA

1.preprocess the data
2.clean the data
3. plot word cloud for better understanding 

# Remove unsignificant columns

Content and class are being considered as significant columns the rest are removed.

![image](https://user-images.githubusercontent.com/71720761/113475568-21519f00-9494-11eb-9783-63084e484652.png)

# FIRST FEW MAILS FROM DATA

![image](https://user-images.githubusercontent.com/71720761/113475578-30d0e800-9494-11eb-98fb-32d432932ae6.png)

# PREPROCESSING THE DATA

![image](https://user-images.githubusercontent.com/71720761/113475594-49410280-9494-11eb-9f3c-f208c0b041b5.png)

# CHECKING FOR ABUSIVE AND NON-ABUSIVE VALUES

By this we can say that there are lesser abusive mails when compared to non-abusive emails.

![image](https://user-images.githubusercontent.com/71720761/113475611-5eb62c80-9494-11eb-822f-f4925ffd9445.png)

![image](https://user-images.githubusercontent.com/71720761/113475612-62e24a00-9494-11eb-8674-53adee81522a.png)

# Data after preprocessing 

![image](https://user-images.githubusercontent.com/71720761/113475617-742b5680-9494-11eb-8793-03626edee121.png)

# WORD CLOUD FOR EMAIL DATA SET

This represents highly frequent words.
By analyzing data, we can remove or keep these words. 

![image](https://user-images.githubusercontent.com/71720761/113475640-8e653480-9494-11eb-978d-4a41484970ff.png)

![image](https://user-images.githubusercontent.com/71720761/113475643-92915200-9494-11eb-8a4e-3bd366fb0fb0.png)

# After removing high frequency words.

![image](https://user-images.githubusercontent.com/71720761/113475657-a472f500-9494-11eb-9f9a-35d3b837ab6a.png)

![image](https://user-images.githubusercontent.com/71720761/113475659-a76de580-9494-11eb-957c-9ef622bce8a4.png)

# Less frequent words

These are low frequent words we can remove these because they don’t give us much significant information.

![image](https://user-images.githubusercontent.com/71720761/113475680-beacd300-9494-11eb-8136-ff3261844240.png)

# FINAL DATA USED TO TRAIN AND TEST THE MODEL

Word cloud after preprocessing the data.

![image](https://user-images.githubusercontent.com/71720761/113475695-d3896680-9494-11eb-9da2-99e93eaf3339.png)

# MODEL BUILDING

![image](https://user-images.githubusercontent.com/71720761/113475709-e4d27300-9494-11eb-86ce-e61e31b0ea85.png)

# Count vectorizer

There are total 1000 tokens

![image](https://user-images.githubusercontent.com/71720761/113475728-fddb2400-9494-11eb-94b2-32b7f299179a.png)

![image](https://user-images.githubusercontent.com/71720761/113475733-016eab00-9495-11eb-9faf-5fa8d268a099.png)

# SPLITTING DATA INTO TEST AND TRAIN

![image](https://user-images.githubusercontent.com/71720761/113475748-13e8e480-9495-11eb-8306-b49029328843.png)

# Applying naïve bayes classifier and predicting for test data

![image](https://user-images.githubusercontent.com/71720761/113475760-206d3d00-9495-11eb-9135-f0ccd7887620.png)

![image](https://user-images.githubusercontent.com/71720761/113475764-2531f100-9495-11eb-8f21-24e93172b569.png)

# Accuracy , precision and recall scores

![image](https://user-images.githubusercontent.com/71720761/113475778-33800d00-9495-11eb-843b-3c514a580b15.png)

# MODEL DEPLOYMENT USING STREAMLIT

![image](https://user-images.githubusercontent.com/71720761/113475836-4f83ae80-9495-11eb-9d6f-d29609399663.png)

# File structure for model deployment

![image](https://user-images.githubusercontent.com/71720761/113475860-60342480-9495-11eb-9de5-66788a7e246c.png)

![image](https://user-images.githubusercontent.com/71720761/113475862-63c7ab80-9495-11eb-8aa7-47404feb3271.png)

![image](https://user-images.githubusercontent.com/71720761/113475865-66c29c00-9495-11eb-9b86-cba76774cde6.png)

# Details about each folder

Appy.py contains the model and the preprocessing.
Static contains the style template that is CSS sheet which displays the output after user input , it says whether the mail is spam or ham.
Template contains the html sheets  which are designed to take input from the user and to display the output.

# PREDICTIONS FOR CERTAIN MAILS(INPUT)

![image](https://user-images.githubusercontent.com/71720761/113475891-88238800-9495-11eb-8cd4-145a1ab9c7a7.png)

# OUTPUT FOR THE USER INPUT

![image](https://user-images.githubusercontent.com/71720761/113475901-94a7e080-9495-11eb-91b9-bc4f4ad10e97.png)

# Non Abusive predictions

![image](https://user-images.githubusercontent.com/71720761/113475912-a6898380-9495-11eb-8ace-88ebf42b615d.png)

![image](https://user-images.githubusercontent.com/71720761/113475915-a9847400-9495-11eb-8e78-631f3632fd98.png)



![image](https://user-images.githubusercontent.com/71720761/113475925-c325bb80-9495-11eb-93ca-4e4f599c5996.png)

