#=======================================================

#=======================================================

rm(list = ls())

library(dplyr)

library(survival)

library(survminer)

library(randomForestSRC)  

library(readxl)

set.seed(12)

setwd("D:\\DuanWork\\S2Sec\\3OSmodel\\year1\\NodePre")

table2 <- read.csv("Table2_final.csv")
head(table2)
# table2_unique <- unique(table2)
table2_unique <- table2 %>%
  group_by(Regulator) %>%
  summarise(across(everything(), max))
lnc2 <- unique(table2_unique$Regulator)

table(table2_unique$cell.proliferation)
table(table2_unique$cell.invasion)
table(table2_unique$cell.migration)
table(table2_unique$apoptosis.process)

write.csv(table2_unique, file = 'Label.csv')



table1 <- read_excel("Table1_final.xlsx")
table1_unique <- unique(table1)

write.csv(table1_unique, file = 'Net.csv')



table1_unique <- subset(table1_unique, table1_unique$RegulatorType == 'lncRNA')

lnc1 <- unique(table1_unique$Regulator)

table(lnc2 %in% lnc1)
table(lnc1 %in% lnc2)


