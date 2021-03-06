from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
import string

#Line Counter
def process(line):
    v = line
    if v !='\n':
        sub_v = v.split('\n')
        sub_v_num = len(sub_v)
        start_line = 0
        for x in sub_v:
            if "title" in x:
                break
            else:
                start_line = start_line+1
        #---------------
        info_1 = sub_v[start_line].replace('<doc ','').replace('>','').replace('"','').split(' ')
        id = info_1[0].split('=')[1]
        #url =info_1[1].split('=')[1]
        title = info_1[2].split('=')[1]
        
        
        #To remove commas simply use Spaces to separate words
        content = ' '.join(sub_v[start_line+1:sub_v_num+1]).replace(',',' ')
        line_num = content.count(".")
        content=content.strip()
        content=content.split(' ')
        words=set()
        wokey={}
        for word in content: #Words counter
            if word not in string.punctuation:
                words.add(word)
                wokey[word]=content.count(word)

        #The first 10 in reverse order
        wokey_1=sorted(wokey.items(),key=lambda d:d[1],reverse=True)[0:100]
        wokey_2=[]
        for k,v in  wokey_1:
            wokey_2.append(str(k)+'_'+str(v)) 
        
        #------
        return id,title,len(words),line_num,'|'.join(wokey_2)
     
#rdd_data = sc.wholeTextFiles("s3a://zihe-public/articles/AA/wiki_02")
rdd_data = sc.wholeTextFiles("s3a://zihe-public/articles/*")
#rdd_data = sc.wholeTextFiles("s3a://zihe-public/articles/AA/*")

# Show result in dataframe
df1 = rdd_data.flatMap(lambda x: (x[1].split('</doc>')))
df2 = df1.filter(lambda x: x!='\n').map(lambda x: process(x)).toDF(["id", "title" , "words","lines","freq_words"])
df2.show()

#Output
df2.write.format('com.databricks.spark.csv').save('/home/notebook/work/all_result3.csv')
