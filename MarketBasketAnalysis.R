#Load arules package
library("arules")

#Read groceries transactional data, read.transactions stores txns in a sparse matrix
groceries <- read.transactions(file.choose() , sep=",")
summary(groceries)
#density refers to proportion of non-zero matrix cells
#total matrix cells = 9835*169,total non-zero matrix cells = 9835*169*density

#find association rules
rules <- apriori(data = groceries, parameter=list(support=0.006,confidence=0.25,minlen=2))
#data = sparse item matrix
#support = minimum support threshold
#condfidence = minimum confidence threshold
#minlen = minimum required rule items
#trial and error to set support and confidence thresholds
#very low threshold => large number of rules,too much time to run operation
#very high threshold => few or very generic rules
#think from a perspective of: How many purchases of an item on a single day will be worth researching about?
#Say the number is 2, so for 30 days worth txn data, item should be purchased 60 times, therefore support = 60/9835 = 0.006


summary(rules)
#lift is another statistical measure that measures how likely will an item be purchased
#as opposed to it's regular purchase , given that another item is already purchased
#lift(X->Y) = confidence(X->Y)/support(Y)

#A strong lift depicts a true connection between items
#get top 5 rules with strongest lift values
inspect(sort(rules, by="lift")[1:5]) 

#use subset of rules to get intersting patterns for a particular item
#general R logical operators like |, & or ! can be used
#can also subset data using support, confidence or lift(e.g. support > 0.7)
herb_rules <- subset(rules , items %in% "herbs")
inspect(herb_rules)
yogurt_rules <- subset(rules , items %in% "yogurt")
inspect(sort(yogurt_rules , by="lift")[1:5])
#To do partial matching of items use pin, fruit_rules will include tropical fruit, citrus fruit etc
fruit_rules <- subset(rules , items %pin% "fruit")
inspect(sort(fruit_rules , by="lift")[1:5])
#To do complete matching of items, e.g. Berries and yogurt both should be present
berry_yogurt_rules <- subset(rules , items %ain% c("berries" , "yogurt"))
inspect(sort(berry_yogurt_rules , by="lift"))
