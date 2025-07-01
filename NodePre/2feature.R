#=======================================================

#=======================================================


rm(list = ls())
library(Biostrings)
library(dplyr)
library(biomaRt)

final_unique <- read.csv(file = 'lncrna_anno.csv', row.names = 1)
colnames(final_unique)

final_unique <- final_unique %>%
  dplyr::select(hgnc_symbol, cdna)


final_unique <- final_unique %>%
  mutate(
    length = nchar(cdna),
    GC_content = (stringr::str_count(cdna, "G") + stringr::str_count(cdna, "C")) / length,
    A_freq = stringr::str_count(cdna, "A") / length,
    T_freq = stringr::str_count(cdna, "T") / length,
    G_freq = stringr::str_count(cdna, "G") / length,
    C_freq = stringr::str_count(cdna, "C") / length
  )

# k-mer
seqs <- DNAStringSet(final_unique$cdna)

# 2-mer
oligo2 <- as.data.frame(oligonucleotideFrequency(seqs, width = 2))
colnames(oligo2) <- paste0("dimer_", colnames(oligo2))

# 3-mer
oligo3 <- as.data.frame(oligonucleotideFrequency(seqs, width = 3))
colnames(oligo3) <- paste0("trimer_", colnames(oligo3))

# 4-mer
oligo4 <- as.data.frame(oligonucleotideFrequency(seqs, width = 4))
colnames(oligo4) <- paste0("tetramer_", colnames(oligo4))

# 合并所有特征
features <- bind_cols(final_unique, oligo2, oligo3, oligo4)
features$cdna <- NULL
features[1:5,1:5]
names(features)

names(features)[1] <- 'Regulator'

Label <- read.csv("Label.csv", header = T, row.names = 1)
head(Label)

Label_final <- merge(Label, features, by='Regulator')

rownames(Label_final) <- NULL

Net <- read.csv('Net.csv', header = T, row.names = 1)

table(Net$RegulatorType)

Net_final1 <- Net[which(Net$RegulatorType == 'lncRNA'),]
Net_final2 <- Net[which(Net$RegulatorType == 'PCG'),]

Net_final1 <- Net_final1[which(Net_final1$Regulator %in% Label_final$Regulator),]

Net_final <- rbind(Net_final1, Net_final2)


rownames(Net_final) <- NULL

write.csv(Net_final, file = "Net_final.csv")

write.csv(Label_final, file = "Label_final.csv")



