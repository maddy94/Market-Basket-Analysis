
library("arules")
groceries <- read.transactions(file.choose() , sep=",")

#To visualise the item frequency of different items, use the itemFrequencyPlot function
#following plot would return a barplot of items which have appeared in atleast -> 9835*0.1 = 983.5 transactions
itemFrequencyPlot(groceries , support = 0.1)

#follwing plot would return a barplot of items which are the N-top frequent items
itemFrequencyPlot(groceries , topN = 15)

#the entire transaction sparse matrix can be examined using image function
image(sample(groceries , 100))

