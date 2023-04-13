# beware-the-for-loop
a comparison of sequential vs parallel group-by data processing

Writing “loops” is a common coding/programming practice.  A loop is a process for iterating and repeating a block of code.  Such a programming tool has many applications.  I find myself frequently using for-loops, and in my job I see many customers using for-loops.  There are a lot of for-loops out there in the world. 

Beware, there is an “insidious” type of for-loop:  one that iterates through subsets of rows in a dataframe, and independently processes each subset.  For example, suppose one column in a dataframe is “geography” which indicates various locations for a retail company.  A common use of a for-loop would be to iterate through each geography, and process the data for each geography separately.  There are many applications for such an approach.  For example, we may want to train machine learning models that are specific to each geography.

But here’s the problem:  for-loops are a serial (not parallel) process.  Why does that matter?  In our brave new world of Big Data, it’s safe to say that parallel processing is paramount.  

There is a common objection that I’ve heard, to the idea of converting existing non-parallelized processing into something that is more “Sparkified”:  when the customer or colleague is using Pandas, and knowing that Pandas is not a distributed-computing package, the objection is a lack of appetite for rewriting their existing code from Python + Pandas into Python + Spark (sans Pandas).  Rest assured, using Pandas does not stand in your way of parallelizing your process.  This is demonstrated in the accompanying code.

The accompanying code demonstrates the insidious type of for-loop (one that iterates through subsets of the data and independently processes each subset), while also demonstrating much faster alternative approaches, thanks to the parallel processing power of Spark.

